import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from const import *
import math
from torch.nn.parameter import Parameter


class DartsChannelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=1):
        super(DartsChannelConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups

        self.weight_full = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))
        self.weight_opt = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))
        self.weight_light = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))

        # mask
        mask = torch.ones(self.out_channels)
        mask[int(self.out_channels*0.75):] = 0
        self.mask_opt = Parameter(mask.clone().reshape(
            len(mask), 1, 1, 1), requires_grad=False)
        mask[int(self.out_channels*0.5):] = 0
        self.mask_light = Parameter(mask.clone().reshape(
            len(mask), 1, 1, 1), requires_grad=False)

        self.channel_alpha = Parameter(torch.Tensor(3), requires_grad=True)
        self.reset_parameters()
        return

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_full, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_opt, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_light, a=math.sqrt(5))
        init.zeros_(self.channel_alpha)
        return

    def forward(self, x):
        out_full = F.conv2d(x, weight=self.weight_full, stride=self.stride,
                            padding=self.padding, groups=self.groups)
        weight_opt = torch.mul(self.weight_opt, self.mask_opt)
        out_opt = F.conv2d(x, weight=weight_opt, stride=self.stride,
                           padding=self.padding, groups=self.groups)
        weight_light = torch.mul(self.weight_light, self.mask_light)
        out_light = F.conv2d(x, weight=weight_light, stride=self.stride,
                             padding=self.padding, groups=self.groups)
        contributions = self.channel_alpha.softmax(dim=0)
        out = out_full*contributions[0]+out_opt * \
            contributions[1]+out_light*contributions[2]
        return out

    def get_channel(self):
        idx = self.channel_alpha.argmax().item()
        if idx == 0:
            return int(self.out_channels*1)
        if idx == 1:
            return int(self.out_channels*0.75)
        if idx == 2:
            return int(self.out_channels*0.5)
        return self.out_channels


class DartsKernelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=1):
        super(DartsKernelConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups

        self.weight_full = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))
        self.weight_middle = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size-2, kernel_size-2))
        self.weight_light = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size-2*2, kernel_size-2*2))

        self.kernel_alpha = Parameter(torch.Tensor(3))
        self.reset_parameters()
        return

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_full, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_middle, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_light, a=math.sqrt(5))
        init.uniform_(self.kernel_alpha, 0, 0)
        return

    def forward(self, x):
        out_full = F.conv2d(x, weight=self.weight_full, stride=self.stride,
                            padding=self.padding, groups=self.groups)
        out_middle = F.conv2d(x, weight=self.weight_middle, stride=self.stride,
                              padding=self.padding-1, groups=self.groups)
        out_light = F.conv2d(x, weight=self.weight_light, stride=self.stride,
                             padding=self.padding-1*2, groups=self.groups)
        contributions = self.kernel_alpha.softmax(dim=0)
        out = out_full*contributions[0]+out_middle * \
            contributions[1]+out_light*contributions[2]
        return out

    def get_kernel(self):
        idx = self.kernel_alpha.argmax().item()
        if idx == 0:
            return self.kernel_size
        if idx == 1:
            return self.kernel_size-2
        if idx == 2:
            return self.kernel_size-2*2
        return self.kernel_size
