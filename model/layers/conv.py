from ast import Param
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from const import *
import torch
import math


class SoftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False, groups=1):
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
        pass

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        return

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, stride=self.stride,
                     padding=self.padding, bias=self.bias, groups=self.groups)
        return x


class SoftChannelConv2d(SoftConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(SoftChannelConv2d, self).__init__(in_channels,
                                                out_channels, kernel_size, stride, padding, bias, groups)

        self.opt_channel_size = None
        self.opt_channel_size_vector = None

        self.channel_base = Parameter(torch.Tensor(1, out_channels))
        self.channel_controller = Parameter(torch.Tensor(1))

        self.up_traingle_for_channel = Parameter(torch.tril(torch.ones(
            self.out_channels, self.out_channels)), requires_grad=False)
        self.reset_parameters()
        pass

    def reset_parameters(self):
        super().reset_parameters()
        init.uniform_(self.channel_base, -1, 1)
        init.uniform_(self.channel_controller, -1, 1)
        return

    def format_channel_size(self):
        channel_p = F.softmax(self.channel_base, dim=1)
        channel_score = torch.mm(channel_p, self.up_traingle_for_channel)
        channel_score_zeroed = channel_score - \
            torch.sigmoid(self.channel_controller)
        channel_score_expand = channel_score_zeroed * EXPANDTIMES
        channel_size = torch.sigmoid(channel_score_expand)

        self.opt_channel_size = round(float(torch.sum(channel_size)))
        self.opt_channel_size_vector = channel_size
        return channel_size

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        x = torch.mul(x, self.format_channel_size().unsqueeze(2).unsqueeze(3))
        return x


class SoftKernelConv2d(SoftConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(SoftKernelConv2d, self).__init__(in_channels,
                                               out_channels, kernel_size, stride, padding, bias, groups)
        self.opt_kernel_size = None
        self.opt_kernel_size_vector = None

        self.kernel_base = Parameter(torch.Tensor(1, int(self.kernel_size/2)))
        self.kernel_controller = Parameter(torch.Tensor(1))

        self.up_traingle_for_kernel = Parameter(torch.tril(torch.ones(
            int(self.kernel_size/2), int(self.kernel_size/2))), requires_grad=False)

        self.reset_parameters()
        pass

    def reset_parameters(self):
        super().reset_parameters()
        init.uniform_(self.kernel_base, -1, 1)
        init.uniform_(self.kernel_controller, -1, 1)
        return

    def format_kernel_size(self):
        # fit step function
        kernel_p = F.softmax(self.kernel_base, dim=1)
        kernel_score = torch.mm(kernel_p, self.up_traingle_for_kernel)
        kernel_score_zeroed = kernel_score - \
            torch.sigmoid(self.kernel_controller)
        kernel_score_expand = kernel_score_zeroed * EXPANDTIMES
        fack_kernel_size_vector = torch.sigmoid(kernel_score_expand)
        # expand shape to kernel size
        kernel_size_vector = torch.ones(
            (1, self.kernel_size)).to(self.weight.device)
        kernel_size_vector[0, 1:] = torch.stack(
            (fack_kernel_size_vector, fack_kernel_size_vector), dim=1).squeeze(0).T.flatten().unsqueeze(0)
        # construct matrix mask
        kernel_size_mask = torch.ones(
            (self.kernel_size, self.kernel_size)).to(self.weight.device)
        for i in range(self.kernel_size):
            kernel_size_mask[:self.kernel_size-i, :self.kernel_size -
                             i] = kernel_size_vector[0][self.kernel_size-i-1]
        # calculate optimal mask
        self.opt_kernel_size = round(float(torch.sum(kernel_size_vector)))
        self.opt_kernel_size_vector = kernel_size_vector
        return kernel_size_mask

    def forward(self, x):
        x = F.conv2d(x, weight=torch.mul(self.weight, self.format_kernel_size(
        )), stride=self.stride, padding=self.padding, groups=self.groups)
        return x
