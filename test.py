from model.cnn_model.resnet import ResNet
from model.nas_model.soft_step import SoftStep
from utils import Data
from const import *
import torch
from torchstat import stat
from torchsummary import summary
from trainers import CNNTrainer, NasTrainer
import json

# test model's statistic
# train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
# input_channel, inputdim, nclass = 3, 32, 10
# model = SoftStep(input_channel, inputdim, nclass,
#                  path="config/residual_cifar10.json")
# model = ResNet(input_channel, inputdim, nclass)
# print(stat(model, (3, 32, 32)))
# summary(model.cuda(DEVICE), input_size=(3, 32, 32), batch_size=-1)

# test nas model
trainer = NasTrainer(SOFTSTEP, CIFAR10, path=RESIDUALCIFAR10)
trainer.train()

# test json file
# with open("config/residual_cifar10.json", 'r') as f:
#     struc = json.load(f)
# print(type(struc["b0"]["conv_in"]))
