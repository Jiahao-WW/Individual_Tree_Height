import torch
from torch.nn.modules.loss import _Loss

import torch.nn.functional as F
import torch.nn as nn

class JointLoss(_Loss):
    def __init__(self):
        super(JointLoss, self).__init__()

    def forward(self, output1, output2, label1, label2, w1, w2):

        # 计算分类交叉熵损失
      
        #criterion1 = nn.CrossEntropyLoss()
        #ce_loss = criterion1(output1, label1)
        ce_loss = F.cross_entropy(output1, label1, reduction='mean')

        # 计算回归损失
        #criterion2 = nn.MSELoss()
        #mse_loss = criterion2(output2, label2)
        mse_loss = F.mse_loss(output2, label2, reduction='mean')

        # 将两个损失函数连接起来
        joint_loss = w1 * mse_loss + w2 * ce_loss
        return joint_loss