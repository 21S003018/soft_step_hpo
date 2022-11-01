from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from const import *
import torch
import math


class SoftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=None, groups=1):
        super(SoftConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups

        self.weight = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))

        self.channel_alpha = Parameter(torch.Tensor(1))
        self.kernel_alpha = Parameter(torch.Tensor(1))
        self.expansion = 10
        self.reset_parameters()
        pass

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.uniform_(self.channel_alpha, int(
            self.out_channels*0.5), int(self.out_channels*0.75))
        init.uniform_(self.kernel_alpha, int(self.kernel_size/2)
                      * 0.5, int(self.kernel_size/2)*0.75)
        return

    def sample_indicator(self, alpha, expansion, num):
        indexes = torch.FloatTensor(range(num))
        if torch.cuda.is_available():
            indexes = indexes.cuda(DEVICE)
        return torch.sigmoid(expansion*(alpha-indexes))

    def forward(self, x):
        # tune kernel size
        indicators = self.sample_indicator(
            self.kernel_alpha, self.expansion, int(self.kernel_size/2))
        mask = torch.ones((self.kernel_size, self.kernel_size))
        if torch.cuda.is_available():
            mask = mask.cuda(DEVICE)
        for index, _ in enumerate(indicators):
            mask[:self.kernel_size-index*2, :self.kernel_size -
                 index*2] = indicators[int(self.kernel_size/2)-index-1]
        masked_weight = torch.mul(self.weight, mask)
        # tune channel amount
        indicators = self.sample_indicator(
            self.channel_alpha, self.expansion, self.out_channels)
        masked_weight = torch.mul(
            masked_weight, indicators.reshape((indicators.shape[0], 1, 1, 1)))
        # conv
        x = F.conv2d(x, weight=masked_weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        return x


class SoftChannelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=1):
        super(SoftChannelConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups

        self.weight = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))

        self.channel_alpha = Parameter(torch.Tensor(1))
        self.expansion = 2.5
        self.reset_parameters()
        return

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # init.uniform_(self.channel_alpha, int(
        #     self.out_channels*0.5), int(self.out_channels*0.75))
        init.uniform_(self.channel_alpha, 0.5, 0.75)
        return

    def sample_indicator(self, alpha, expansion, num):
        indexes = torch.FloatTensor(range(num))
        if torch.cuda.is_available():
            indexes = indexes.cuda(DEVICE)
        return torch.sigmoid(expansion*self.out_channels*(alpha-indexes/self.out_channels))

    def forward(self, x):
        indicators = self.sample_indicator(
            self.channel_alpha, self.expansion, self.out_channels)
        masked_weight = torch.mul(
            self.weight, indicators.reshape((indicators.shape[0], 1, 1, 1)))
        x = F.conv2d(x, weight=masked_weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        return x


class SoftKernelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=1):
        super(SoftKernelConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups

        self.weight = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))
        self.kernel_alpha = Parameter(torch.Tensor(1))
        self.expansion = 10
        self.reset_parameters()
        return

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.uniform_(self.kernel_alpha, int(self.kernel_size/2)
                      * 0.5, int(self.kernel_size/2)*0.75)
        return

    def sample_indicator(self, alpha, expansion, num):
        indexes = torch.FloatTensor(range(num))
        if torch.cuda.is_available():
            indexes = indexes.cuda(DEVICE)
        return torch.sigmoid(expansion*(alpha-indexes))

    def forward(self, x):
        indicators = self.sample_indicator(
            self.kernel_alpha, self.expansion, int(self.kernel_size/2))
        mask = torch.ones((self.kernel_size, self.kernel_size))
        if torch.cuda.is_available():
            mask = mask.cuda(DEVICE)
        for index, _ in enumerate(indicators):
            mask[:self.kernel_size-index*2, :self.kernel_size -
                 index*2] = indicators[int(self.kernel_size/2)-index-1]
        masked_weight = torch.mul(self.weight, mask)
        x = F.conv2d(x, weight=masked_weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        return x
