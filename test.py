from model.cnn_model.resnet import ResNet
from model.nas_model.soft_step import SoftStep
from utils import Data
from const import *
import torch
from torchstat import stat
from torchsummary import summary
from trainers import CNNTrainer, NasTrainer
from model.nas_model.layers.conv import SoftConv2d, SoftChannelConv2d, SoftKernelConv2d
import json
import torch.nn as nn

# test model's statistic
# # train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
# input_channel, inputdim, nclass = 3, 32, 10
# model = SoftStep(input_channel, inputdim, nclass, path=RESIDUALCIFAR10)
# # model = ResNet(input_channel, inputdim, nclass)
# # print(stat(model, (3, 32, 32)))
# # summary(model.cuda(DEVICE), input_size=(3, 32, 32), batch_size=-1)

# # test nas model
# trainer = NasTrainer(SOFTSTEP, CIFAR10, path=RESIDUALCIFAR10)
# trainer.train()

# test json file
# with open("config/residual_cifar10.json", 'r') as f:
#     struc = json.load(f)
# print(type(struc["b0"]["conv_in"]))

# test dhpo conv layer
x = torch.rand(1, 3, 32, 32)
model = SoftKernelConv2d(
    in_channels=3, out_channels=3, kernel_size=3, groups=1)
model.cuda(DEVICE)
model.format_kernel_size()
print(model.kernel_size_vector.device)
# print(model.up_traingle_for_channel)
# preds = model(x)
# print(preds.size(), model.format_channel_size().size())
# print(model.format_kernel_size().size())
# print(model.opt_channel_size)
# print(model.opt_kernel_size)

# model = nn.Conv2d(4, 6, 3, groups=2)
# print(model.weight.size())

# x = torch.rand(1, 3)
# print(x)
# y = torch.ones(1, 7)
# y[0, 1:] = torch.stack((x, x), dim=1).squeeze(0).T.flatten().unsqueeze(0)
# print(y)
