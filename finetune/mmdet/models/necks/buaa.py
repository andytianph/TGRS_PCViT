# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from ..utils import ConvModule_Norm

from ..builder import NECKS


@NECKS.register_module()
class buaa(BaseModule):
    def __init__(self,):
        super(buaa, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(2, 2)


    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        x1 = inputs[0]
        x2 = inputs[1]
        x3 = inputs[2]
        x4 = inputs[3]

        x1 = self.maxpool(self.maxpool(x1))
        x2 = self.maxpool(x2)

        x5_avg = torch.mean(x5, dim=1)
        x5_sig = (x5_avg - torch.min(x5_avg)) / (torch.max(x5_avg) - torch.min(x5_avg))

        x3_avg = torch.mean(x3, dim=1)
        x3_sig = (x3_avg - torch.min(x3_avg)) / (torch.max(x3_avg) - torch.min(x3_avg))
      

        x2_avg = torch.mean(x2, dim=1)
        x2_sig = (x2_avg - torch.min(x2_avg)) / (torch.max(x2_avg) - torch.min(x2_avg))


        x1_avg = torch.mean(x1, dim=1)
        x1_sig = (x1_avg - torch.min(x1_avg)) / (torch.max(x1_avg) - torch.min(x1_avg))


        x_sig = torch.where(x5_sig > x3_sig, x5_sig, x3_sig)
        x_sig = torch.where(x_sig > x2_sig, x_sig, x2_sig)
        x_sig = torch.where(x_sig > x1_sig, x_sig, x1_sig)

        x5 = x5 * (1.0 + x_sig)

        return tuple(x5)
