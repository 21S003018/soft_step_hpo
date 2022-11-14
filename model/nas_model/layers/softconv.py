from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from const import *
import torch
import math
import utils


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
        self.channel_alpha_expansion = utils.newton_expansion(
            self.out_channels)
        self.kernel_alpha_expansion = utils.newton_expansion(
            int(self.kernel_size/2))
        self.reset_parameters()
        # init channel indicators and kernel mask
        self.channel_indicators = None
        self.mask = None
        pass

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.uniform_(self.channel_alpha, 1-1 /
                      self.out_channels, 1-1/self.out_channels)
        init.uniform_(self.kernel_alpha, 1-1 /
                      int(self.kernel_size/2), 1-1/int(self.kernel_size/2))
        return

    def forward(self, x, arch_opt):
        if arch_opt:
            masked_weight = torch.mul(self.weight, self.mask)
        else:
            masked_weight = torch.mul(self.weight, self.mask.data)
        x = F.conv2d(x, weight=masked_weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        return x

    def sample_channel_indicator(self):
        indexes = torch.FloatTensor(range(1, self.out_channels+1))
        if torch.cuda.is_available():
            indexes = indexes.cuda(DEVICE)
        return torch.sigmoid(self.channel_alpha_expansion*self.out_channels*(self.channel_alpha+self.channel_controller_approx_delta()-indexes/self.out_channels))

    def channel_controller_approx_delta(self):
        if self.channel_alpha.data > 1:
            init.uniform_(self.channel_alpha, 1, 1)
        if self.channel_alpha.data < 1/self.out_channels:
            init.uniform_(self.channel_alpha, 1 /
                          self.out_channels, 1/self.out_channels)
        real_controller = self.channel_alpha.data*self.out_channels
        real_delta = 0.5-(real_controller - real_controller.floor())
        unit_delta = real_delta/self.out_channels
        return unit_delta

    def update_channel_indicators(self):
        self.channel_indicators = self.sample_channel_indicator().reshape(
            (1, self.out_channels, 1, 1))
        return

    def sample_kernel_indicator(self):
        indexes = torch.FloatTensor(range(1, int(self.kernel_size/2)+1))
        if torch.cuda.is_available():
            indexes = indexes.cuda(DEVICE)
        return torch.sigmoid(self.kernel_alpha_expansion*int(self.kernel_size/2)*(self.kernel_alpha+self.kernel_controller_approx_delta()-indexes/int(self.kernel_size/2)))

    def kernel_controller_approx_delta(self):
        if self.kernel_alpha.data > 1:
            init.uniform_(self.kernel_alpha, 1, 1)
        if self.kernel_alpha.data < 1/int(self.kernel_size/2):
            init.uniform_(self.kernel_alpha, 1 /
                          int(self.kernel_size/2), 1/int(self.kernel_size/2))
        real_controller = self.kernel_alpha.data*int(self.kernel_size/2)
        real_delta = 0.5-(real_controller - real_controller.floor())
        unit_delta = real_delta/int(self.kernel_size/2)
        return unit_delta

    def update_kernel_mask(self):
        indicators = self.sample_kernel_indicator()
        self.mask = torch.ones((self.kernel_size, self.kernel_size))
        if torch.cuda.is_available():
            self.mask = self.mask.cuda(DEVICE)
        for index, _ in enumerate(indicators):
            self.mask[index:self.kernel_size-index, index:self.kernel_size -
                      index] = indicators[int(self.kernel_size/2)-index-1]
        return


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

        self.channel_alpha = Parameter(torch.Tensor(1), requires_grad=True)
        self.expansion = utils.newton_expansion(self.out_channels)
        self.reset_parameters()
        # init channel indicators
        self.channel_indicators = None
        return

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.uniform_(self.channel_alpha, 1-1 /
                      self.out_channels, 1-1/self.out_channels)
        return

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        return x

    def sample_indicator(self):
        indexes = torch.FloatTensor(range(1, self.out_channels+1))
        if torch.cuda.is_available():
            indexes = indexes.cuda(DEVICE)
        return torch.sigmoid(self.expansion*self.out_channels*(self.channel_alpha+self.controller_approx_delta()-indexes/self.out_channels))

    def controller_approx_delta(self):
        if self.channel_alpha.data > 1:
            init.uniform_(self.channel_alpha, 1, 1)
        if self.channel_alpha.data < 1/self.out_channels:
            init.uniform_(self.channel_alpha, 1 /
                          self.out_channels, 1/self.out_channels)
        real_controller = self.channel_alpha.data*self.out_channels
        real_delta = 0.5-(real_controller - real_controller.floor())
        unit_delta = real_delta/self.out_channels
        return unit_delta

    def update_channel_indicators(self):
        self.channel_indicators = self.sample_indicator().reshape(
            (1, self.out_channels, 1, 1))
        return


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
        self.expansion = utils.newton_expansion(int(self.kernel_size/2))
        self.reset_parameters()
        # init kernel mask
        self.mask = None
        return

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.uniform_(self.kernel_alpha, 1-1 /
                      int(self.kernel_size/2), 1-1/int(self.kernel_size/2))
        return

    def forward(self, x, arch_opt):
        if arch_opt:
            masked_weight = torch.mul(self.weight, self.mask)
        else:
            masked_weight = torch.mul(self.weight, self.mask.data)
        x = F.conv2d(x, weight=masked_weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        return x

    def sample_indicator(self):
        indexes = torch.FloatTensor(range(1, int(self.kernel_size/2)+1))
        if torch.cuda.is_available():
            indexes = indexes.cuda(DEVICE)
        return torch.sigmoid(self.expansion*int(self.kernel_size/2)*(self.kernel_alpha+self.controller_approx_delta()-indexes/int(self.kernel_size/2)))

    def controller_approx_delta(self):
        if self.kernel_alpha.data > 1:
            init.uniform_(self.kernel_alpha, 1, 1)
        if self.kernel_alpha.data < 1/int(self.kernel_size/2):
            init.uniform_(self.kernel_alpha, 1 /
                          int(self.kernel_size/2), 1/int(self.kernel_size/2))
        real_controller = self.kernel_alpha.data*int(self.kernel_size/2)
        real_delta = 0.5-(real_controller - real_controller.floor())
        unit_delta = real_delta/int(self.kernel_size/2)
        return unit_delta

    def update_kernel_mask(self):
        indicators = self.sample_indicator()
        self.mask = torch.ones((self.kernel_size, self.kernel_size))
        if torch.cuda.is_available():
            self.mask = self.mask.cuda(DEVICE)
        for index, _ in enumerate(indicators):
            self.mask[index:self.kernel_size-index, index:self.kernel_size -
                      index] = indicators[int(self.kernel_size/2)-index-1]
        return
