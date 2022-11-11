from model.blocks.basic_block import SoftResidualBlock
from model.cnn_model.resnet import ResNet
from model.nas_model.soft_step import SoftStep
# from utils import Data
from const import *
import torch
import torch.nn.init as init
# import thop
# from torchstat import stat
# from torchsummary import summary
# from pthflops import count_opss
# from trainers import CNNTrainer, NasTrainer
from model.layers.softconv import SoftConv2d, SoftChannelConv2d, SoftKernelConv2d
from model.cnn_model.mobilenet import MobileNetV2
import json
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd.variable import Variable
import math
import torch.nn.functional as F
from torch import autograd
import utils

# test model's statistic
# train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
# input_channel, inputdim, nclass = 3, 256, 1000
# model = SoftStep(input_channel, inputdim, nclass,
#  path=INVERTEDRESIDUALIMAGENET)
# model = ResNet(input_channel, inputdim, nclass)
# model = MobileNetV2()
# # model = SoftResidualBlock(input_channel, planes=64,
# #   kernel_size=7, stride=2, expansion=1)
# print(stat(model, (3, 256, 256)))
# summary(model.cuda(), input_size=(3, 256, 256), batch_size=-1)
# for name, param in model.arch_parameters():
#     print(name)
#     # break

# # test nas model
# trainer = NasTrainer(SOFTSTEP, CIFAR10, path=RESIDUALCIFAR10)
# trainer.train()

# test json file
# with open("config/residual_cifar10.json", 'r') as f:
#     struc = json.load(f)
# print(type(struc["b0"]["conv_in"]))

# test softstep conv layer
x = torch.rand(1, 3, 32, 32)
# x = Parameter(torch.Tensor(1))
# print(x)
# x.data = x.data.floor()+1.5
# print(x.floor())
# model = SoftKernelConv2d(in_channels=3, out_channels=64, kernel_size=7, groups=1)
# model = SoftChannelConv2d(in_channels=3, out_channels=64, kernel_size=7, groups=1)
# model = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, groups=1)
# model = SoftStep()
# model = ResNet(3, 32, 100,)
# model = MobileNetV2(3, 32, 100,)
# out = model(x)
# stat(model, (3, 32, 32))
# count_ops(model, x)
# flops, params = thop.profile(model, inputs=(x,))
# print(flops)

# model.cuda(DEVICE)
# print(model.format_kernel_size())
# print(model.up_traingle_for_channel)
# preds = model(x)
# print(preds.size(), model.format_channel_size().size())
# print(model.format_kernel_size().size())
# print(model.opt_channel_size)
# print(model.opt_kernel_size)

# model = nn.Conv2d(4, 6, 3, groups=2)
# print(model.weight.size())

# x = torch.rand(1, 3)
# x = x.cuda(DEVICE)
# print(x.device)
# y = torch.ones(1, 7)
# y[0, 1:] = torch.stack((x, x), dim=1).squeeze(0).T.flatten().unsqueeze(0)
# print(y)

# test controller's grad wtr indicator


def newton_expansion(c):
    def f(x):
        return 2*c*x-2-(math.exp(x/2) + math.exp(-x/2))

    def f_derive(x):
        return 2*c - (1/2*math.exp(x/2)-1/2*math.exp(-x/2))
    x = 2*math.log(4*c)*1.1
    for i in range(10):
        # while True:
        x = x - f(x)/f_derive(x)
    return x


out_channels = 32
expansion = newton_expansion(out_channels)
controller = Parameter(torch.Tensor(1))
init.uniform_(controller, 0.5, 0.75)
# init.uniform_(controller, 115.5/192, 115.5/192)

indexes = torch.FloatTensor(range(1, out_channels+1))
indicators = torch.sigmoid(expansion*out_channels *
                           (controller-indexes/out_channels))

# print(controller.item())
# for i, indicator in enumerate(indicators):
#     controller.grad = None
#     indicator.backward(retain_graph=True)
#     if controller.grad.item() == 0:
#         continue
#     print("{}-th, indicator: {}, gradient: {}".format(i,
#           indicator.item(), controller.grad.item()))

# # test indicators' grad
# conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
# bn1 = nn.BatchNorm2d(32)
# conv2 = nn.Conv2d(32, 16, 1, bias=False)
# bn2 = nn.BatchNorm2d(16)
# fc = nn.Linear(16, 10)
# train_loader, test_loader, input_channel, inputdim, nclass = utils.Data().get(CIFAR10)
# for imgs, label in train_loader:
#     x, y = imgs, label
#     break
# n = 100
# # x = torch.rand((n, out_channels, 4, 4))
# x = conv1(x)
# x = bn1(x)
# x = F.relu6(x)
# x = torch.mul(x, indicators.reshape((1, 32, 1, 1)))
# x = conv2(x)
# x = bn2(x)
# x = F.relu6(x)
# x = F.adaptive_avg_pool2d(x, 1)
# x = x.view(x.size(0), -1)
# x = fc(x)
# loss = F.cross_entropy(x, y)
# indicators_grad = autograd.grad(loss, indicators, retain_graph=True)[0]
# # controller_grad = autograd.grad(
# # indicators[0], controller, retain_graph=True)[0]
# print(indicators)
# print(indicators_grad)
# # print(controller_grad)

# x = Parameter(torch.Tensor([1]), requires_grad=True)
# w = x-1+0.1
# F.relu(w, inplace=True)
# y = (w-1)**2 - 1
# y.backward()
# print(x.grad)

# track internal variable's grad
# indicator_grad_8_conv1 = autograd.grad(
#     loss, self.model.blocks[8].conv1.indicators, retain_graph=True)[0]
# controller_grad_8_conv1 = []
# for indicator in self.model.blocks[8].conv1.indicators:
#     controller_grad_8_conv1.append(
#         autograd.grad(indicator, self.model.blocks[8].conv1.channel_alpha, retain_graph=True)[0])
# # print(indicator_grad_8_conv1)
# # print(controller_grad_8_conv1)
# grad = 0
# for j, indicator_grad in enumerate(indicator_grad_8_conv1):
#     print(j, self.model.blocks[8].conv1.indicators[j].item(), indicator_grad.item(),
#           controller_grad_8_conv1[j].item())
#     grad += indicator_grad.item() * \
#         controller_grad_8_conv1[j].item()
# print(grad)

# track params's grad
#    max_grad = 0
#    for name, param in self.model.named_parameters():
#        if not name.__contains__("alpha") and not name.__contains__("indicators"):
#             # tmp = param.grad.max().item()
#             # if tmp > max_grad:
#             #     max_grad = tmp
#             #     max_grad_name = name
#             pass
#         elif name.__contains__("alpha"):
#             print(i+1, name, param.data,
#                   param.grad.data)
#         elif name.__contains__("indica"):
#             print(i+1, name, param.grad)
#     # print("max grad {}: {}. munual grad: {}".format(
#     #     max_grad_name, max_grad, grad))
