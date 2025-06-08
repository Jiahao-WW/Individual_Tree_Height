#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :SAMUnet.py
@说明        :
@时间        :2024/05/30 16:06:26
@作者        :Jiahao W
'''


from typing import Optional, Union, List
from segment_anything.build_sam_encoder import sam_encoder_model_registry
from importlib import import_module
from encoders import get_encoder
from fusion_module import Fusion
from base import (
    SamSegmentationModel,
    SegmentationHead,
)
from decoders.decoder_unetplusplus import UnetPlusPlusDecoder,UnetDecoder_sam
from decoders.decoder_unet import UnetDecoder
from sam_lora_bias import LoRA_bias_Sam


class SAMUNet(SamSegmentationModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 image_size: int = 256,
                 encoder_depth: int = 4,
                 in_channels: int = 3,
                 classes: int = 1,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (128, 64, 32, 16),
                 decoder_attention_type: Optional[str] = None,
                 activation: Optional[Union[str, callable]] = None, 
        ):
        super().__init__()
        self.cnnencoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        sam_encoder, img_embedding_size = sam_encoder_model_registry['vit_b'](image_size=image_size,
                                                                              num_classes=classes-1,
                                                                              checkpoint='d:/WJH/pythoncode/Chinatree/IndividualTree_single/checkpoints/sam_encoder_vit_b_01ec64.pth',
                                                                              pixel_mean=[0, 0, 0],
                                                                              pixel_std=[1, 1, 1])
        self.img_embedding_size = img_embedding_size

        self.sam_encoder = LoRA_bias_Sam(sam_encoder, encoder_depth)

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.cnnencoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=4,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.fusion = Fusion(self.cnnencoder.out_channels[-1], 256)

        self.initialize()

