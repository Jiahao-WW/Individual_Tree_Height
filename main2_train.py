from osgeo import gdal
import os
os.environ['PROJ_LIB'] = 'D:/software/Anaconda/envs/deep/Library/share/proj'


import torch
import torch.nn.functional as F
from pytorch_toolbelt import losses as L
import numpy as np
import warnings
from dataload import Mydataset
from metrics import ACC
import time
import csv
from pytorch_lightning import seed_everything  
from progress.bar import Bar
from utils import  Logger, AverageMeter
from SAMUnet import SAMUNet


def loss_fn(seg, target1, bounds):

    diff = F.cross_entropy(seg, target1,reduction='mean')
    bounds = bounds * 4 + 1 
    bound_loss = torch.mean(diff * bounds)
    #diff1 = F.binary_cross_entropy(seg, target1, reduction='mean')

    return bound_loss

def train(model, optimizer, train_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_ave = AverageMeter()
    acc_ave = AverageMeter()
    start = time.time()

    model.train()

    #显示进度条
    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (image, target, bounds) in enumerate(train_loader):

        data_time_single = time.time() - start
        data_time.update(data_time_single)

        image, target, bounds  = image.to(DEVICE), target.to(DEVICE),bounds.to(DEVICE)
        seg = model(image)

        target = torch.tensor(target, dtype = torch.int64)

        target = target.squeeze(1)

        loss= loss_fn(seg,  target, bounds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #计算训练误差

        seg = torch.argmax(seg, axis=1)
        seg = seg.squeeze(1) #4D to 3D

        acc = ACC(seg,target)


        loss_ave.update(loss.item(), image.size(0))
        acc_ave.update(acc.item(), image.size(0))

    
    # scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - start)
        start =time.time()
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | acc: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=loss_ave.avg,
                    top1=acc_ave.avg,
                    )
        bar.next()
    bar.finish()


    return (loss_ave.avg, acc_ave.avg)

def test(model, test_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_ave = AverageMeter()
    acc_ave = AverageMeter()
    start = time.time()
    

    model.eval()
    

    #显示进度条
    bar = Bar('Processing', max=len(test_loader))
    for batch_idx, (val_img, val_mask,  bound) in enumerate(test_loader):

        data_time_single = time.time() - start
        data_time.update(data_time_single)

        val_img, val_mask,  bound = val_img.to(DEVICE), val_mask.to(DEVICE),  bound.to(DEVICE)
        predict= model(val_img)

        val_mask = torch.tensor(val_mask, dtype = torch.int64)
        val_mask = val_mask.squeeze(1)

        valloss = loss_fn(predict, val_mask, bound)

        predict = torch.argmax(predict, axis=1)
        predict = predict.squeeze(1) #4D to 3D
    
        acc = ACC(predict,val_mask)

        loss_ave.update(valloss.item(), val_img.size(0))
        acc_ave.update(acc.item(), val_img.size(0))
        
        # measure elapsed time

        batch_time.update(time.time() - start)
        start =time.time()
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | acc: {acc: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(test_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=loss_ave.avg,
                    acc=acc_ave.avg,
                    )
        bar.next()
    bar.finish()

    return (loss_ave.avg, acc_ave.avg)


if __name__ == '__main__':
    seed_everything(42)
    torch.cuda.empty_cache()
    begintime=time.time()
    EPOCHES = 500
    BATCH_SIZE = 16
    channels = 3
    n_classes = 2 # Classification
    optimizer_name = 'Adam' # optimizer
    lr=0.0001

    data_root =  'D:/dataset/segdataset_npy/'
    model_path = 'D:/dataset/model/segment'

    early_stop = 200
    torch.cuda.empty_cache()

    #dataset
    train_dataset = Mydataset(data_root, mode = 'train')
    val_dataset = Mydataset(data_root, mode = 'val')
    warnings.filterwarnings('ignore')
    torch.backends.cudnn.enable = True

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 4,
        persistent_workers=True)

    val_data_loader = torch.utils.data.DataLoader(
		val_dataset,
		batch_size = BATCH_SIZE,
		shuffle=True,
		num_workers= 4,
        persistent_workers=True)


    #using cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'0'  '1'  '0,1'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  

    #model
    model = SAMUNet(
	encoder_name="resnext50_32x4d",       
	encoder_weights= 'imagenet',     
	in_channels=channels,                
	classes=n_classes,                 
	)

    '''premodel_path = r'D:\dataset\model\segment\samunet0604.pth'
    if premodel_path != None:
        model.load_state_dict(torch.load(premodel_path))'''

    if torch.cuda.device_count()>1:
        print('GPUs')
        model = torch.nn.DataParallel(model)

    model.to(DEVICE)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    params = ([p for p in model.parameters()])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0 = 2,
        T_mult = 2,
        eta_min = 1e-5)

    # log   
    title = 'unet'
    logger_train = Logger(model_path+'_train.txt', title=title)
    logger_train.set_names(['Epoch','Learning Rate', 'Train Loss', 'Train Acc'])
    logger_test = Logger(model_path+'_test.txt', title=title)
    logger_test.set_names(['Epoch', 'Test Loss', 'Test Acc'])


    if os.path.exists(model_path+"_result.csv"):
        f = open(model_path+"_result.csv",'a',encoding='utf-8',newline='')
        csv_writer = csv.writer(f)
    else:
        f = open(model_path+"_result.csv",'w',encoding='utf-8',newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow(['model','dataset','lr','ACC','run_time'])
    

    #training+val
    best_valacc = 0
    for epoch in range(EPOCHES): 
        print('\nEpoch: [%d | %d] LR: %f ' % (epoch + 1, EPOCHES, optimizer.param_groups[0]['lr']))
        #train
        train_loss, train_acc = train(model, optimizer,  train_data_loader)
        logger_train.append([epoch + 1, optimizer.param_groups[0]['lr'], train_loss,  train_acc])
        scheduler.step()

        #val
        if epoch % 2 == 0:
            test_loss, test_acc = test(model, val_data_loader) 


            if best_valacc < test_acc:
                best_valacc = test_acc
                best_epoch = epoch
                if torch.cuda.device_count()>1:
                    torch.save(model.module.state_dict(),  model_path+'_best'+'.pth')
                else:
                    torch.save(model.state_dict(), model_path+'_best'+'.pth')  
                print("valid acc is improved. the model is saved.")
            
            '''if (epoch - best_epoch) >= early_stop:
                break'''
            
            logger_test.append([epoch + 1, test_loss, test_acc])

    endtime=time.time()
    runtime=endtime-begintime
    runtime = np.round(runtime, decimals=4)
    #result
    csv_writer.writerow([title,'train1',lr, '%.4f'%best_valacc, str(runtime)])
    f.close()

    if torch.cuda.device_count()>1:
        torch.save(model.module.state_dict(),  model_path+'.pth')
    else:
        torch.save(model.state_dict(), model_path+'.pth')  
    
