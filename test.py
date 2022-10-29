from model.blocks.basic_block import SoftResidualBlock
from model.cnn_model.resnet import ResNet
from model.nas_model.soft_step import SoftStep
# from utils import Data
from const import *
import torch
# import thop
# from torchstat import stat
# from torchsummary import summary
# from pthflops import count_opss
# from trainers import CNNTrainer, NasTrainer
from model.layers.softconv import SoftConv2d, SoftChannelConv2d, SoftKernelConv2d
from model.cnn_model.mobilenet import MobileNetV2
import json
import torch.nn as nn

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
# model = SoftKernelConv2d(in_channels=3, out_channels=64, kernel_size=7, groups=1)
# model = SoftChannelConv2d(in_channels=3, out_channels=64, kernel_size=7, groups=1)
# model = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, groups=1)
model = SoftStep()
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
