from model.nas_model.soft_step import SoftStep
from utils import Data
from const import *
import torch
from torchstat import stat
from torchsummary import summary
from trainers import CNNTrainer

# train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
# input_channel, inputdim, nclass = 3, 32, 10
# model = IRFBlock(in_channels=input_channel, out_channels=3)
# model = SoftStep(input_channel, inputdim, nclass)
# print(stat(model, (3, 32, 32)))
# summary(model.cuda(), input_size=(3, 32, 32), batch_size=-1)
trainer = CNNTrainer(SOFTSTEP, CIFAR10)
trainer.train()
