from ast import Param
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from const import *
import torch
import math


class SinglePathChannelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=1):
        super(SinglePathChannelConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups

        self.weight = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))

        self.channel_alpha = Parameter(torch.Tensor(
            1, out_channels), requires_grad=True)
        # init channel indicators
        self.channel_indicators = None
        self.reset_parameters()
        return

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.uniform_(self.channel_alpha, -1, 1)
        return

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        x = torch.mul(x, self.sample_indicator().unsqueeze(2).unsqueeze(3))
        return x

    def sample_indicator(self):
        norm = torch.norm(torch.norm(self.weight, dim=(2, 3)), dim=1)
        indicators = torch.sigmoid(norm-self.channel_alpha)
        for i in range(1, len(indicators)):
            indicators[i] *= indicators[i-1]
        return indicators

    def get_channel(self):
        indicators = self.sample_indicator()
        return (indicators > 0.5).sum().item()


class SinglePathKernelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=1):
        super(SinglePathKernelConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups

        self.weight = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))

        self.kernel_alpha = Parameter(torch.Tensor(
            1, int(kernel_size/2)), requires_grad=True)
        # init kernel indicators
        self.kernel_indicators = None
        self.reset_parameters()
        return

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.uniform_(self.kernel_alpha, -1, 1)
        return

    def forward(self, x):
        indicators = self.sample_indicator()
        mask = torch.ones((self.kernel_size, self.kernel_size))
        mask[1:self.kernel_size-1, 1:self.kernel_size-1] *= indicators[1]
        mask[2:self.kernel_size-2, 2:self.kernel_size-2] *= indicators[2]
        weight = torch.mul(self.weight, mask)
        x = F.conv2d(x, weight=weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        return x

    def sample_indicator(self):
        norm = [torch.norm(self.weight), torch.norm(
            self.weight[:, :, 1:self.kernel_size-1, 1:self.kernel_size-1]), torch.norm(
            self.weight[:, :, 2:self.kernel_size-2, 2:self.kernel_size-2])]
        indicators = [1, torch.sigmoid(norm[1]-self.kernel_alpha[1]), torch.sigmoid(
            norm[1]-self.kernel_alpha[1])*torch.sigmoid(norm[2]-self.kernel_alpha[2])]
        return indicators

    def get_channel(self):
        indicators = self.sample_indicator()
        n = 0
        for indi in indicators:
            if indi > 0.5:
                n += 1
        return 2*n+1
