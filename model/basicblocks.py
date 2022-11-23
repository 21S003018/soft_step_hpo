import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, c1, k1, c2, k2, stride=1, type="normal"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, c1, kernel_size=k1, stride=stride, padding=int(k1/2), bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=k2,
                               stride=1, padding=int(k2/2), bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

        self.in_planes = in_planes
        self.out_planes = c2
        self.stride = stride
        self.block_type = type

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.stride == 1 and self.in_planes == self.out_planes and self.block_type == "skip":
            out += x
        out = F.relu6(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, c1, c2, c3, kernel_size=3, stride=1, type="normal"):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, c1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=kernel_size,
                               stride=stride, padding=int(kernel_size/2), bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c3)

        self.in_planes = in_planes
        self.out_planes = c3
        self.stride = stride
        self.block_type = type

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride == 1 and self.in_planes == self.out_planes and self.block_type == "skip":
            out = out + x
        out = F.relu6(out)
        return out


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expansion=None, type="normal"):
        super(InvertedResidualBlock, self).__init__()
        self.expansion = expansion
        hidden_planes = round(in_planes * self.expansion)

        self.conv1 = nn.Conv2d(in_planes, hidden_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)
        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                               stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)
        self.conv3 = nn.Conv2d(hidden_planes, out_planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.block_type = type

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride == 1 and self.in_planes == self.out_planes and self.block_type == "skip":
            out = out + x
        out = F.relu6(out)
        return out


if __name__ == '__main__':
    # train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
    # model = IRFBlock(in_channels=input_channel, out_channels=3)
    model = nn.Conv2d(3, 3, kernel_size=2,
                      stride=1, padding=0, bias=False, groups=1)
    for name, param in model.named_parameters():
        print(name, param.size(), param.sum(3).sum(2).sum(1))
    input = torch.ones(3, 4, 4)
    print(input.size())
    out = model(input)
    print(out.size(), out)
