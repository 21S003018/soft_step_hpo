from model.nas_model.soft_step import SoftStep
from utils import Data
from const import *
import torch

# train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
input_channel, inputdim, nclass = 3, 32, 10
# model = IRFBlock(in_channels=input_channel, out_channels=3)
model = SoftStep(input_channel, inputdim, nclass)
x = torch.rand(1, 3, 32, 32)
preds = model(x)
print(preds.size())
