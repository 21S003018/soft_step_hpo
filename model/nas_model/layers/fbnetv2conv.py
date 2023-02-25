from ast import Param
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from const import *
import torch
import math


class FBnetChannelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=1):
        super(FBnetChannelConv2d, self).__init__()
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
        self.up_traingle_for_channel = Parameter(torch.tril(torch.ones(
            self.out_channels, self.out_channels)), requires_grad=False)
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
        return torch.mm(F.softmax(self.channel_alpha, dim=1),
                        self.up_traingle_for_channel)

    def get_channel(self):
        indicators = self.sample_indicator()
        return (indicators > 0.5).sum().item()
