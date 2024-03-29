import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, hidden_planes, out_planes, stride=1, type="normal"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, hidden_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)
        self.conv2 = nn.Conv2d(hidden_planes, out_planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.block_type = type
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu6(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, hidden_planes, kernel_size=3, stride=1, expansion=None, type="normal"):
        super(ResidualBlock, self).__init__()
        out_planes = round(hidden_planes * expansion)
        # hidden_planes = round(hidden_planes * 2)
        self.conv1 = nn.Conv2d(in_planes, hidden_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)
        # assert c1 == c2
        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                               stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = nn.Conv2d(hidden_planes, out_planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # if self.stride == 1 and self.in_planes == self.out_planes:
        out += self.shortcut(x)
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
