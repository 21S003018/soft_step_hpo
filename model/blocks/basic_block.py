from torch import nn
import torch.nn.functional as F
import torch
from model.nas_model.layers.conv import SoftChannelConv2d, SoftKernelConv2d


class ResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, expansion=None):
        super(ResidualBlock, self).__init__()
        if not expansion is None:
            self.expansion = expansion
        hidden_planes = round(in_planes / self.expansion)
        self.conv1 = nn.Conv2d(in_planes, hidden_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = nn.Conv2d(hidden_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SoftResidualBlock(ResidualBlock):
    def __init__(self, in_planes, planes, stride=1, expansion=None):
        # super().__init__(in_planes, planes, stride, expansion)
        if not expansion is None:
            self.expansion = expansion
        hidden_planes = round(in_planes / self.expansion)
        self.conv1 = SoftChannelConv2d(in_planes, hidden_planes,
                                       kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = SoftChannelConv2d(
            hidden_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        return


class InvertedResidualBlock(nn.Module):
    expansion = 6

    def __init__(self, inplanes, planes, stride=1, expansion=None):
        super(InvertedResidualBlock, self).__init__()
        if not expansion:
            self.expansion = expansion
        hidden_planes = round(inplanes * self.expansion)

        # pw
        self.conv1 = nn.Conv2d(inplanes, hidden_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)
        # dw
        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=3,
                               stride=stride, padding=1, groups=hidden_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_planes)
        # pw-linear
        self.conv3 = nn.Conv2d(hidden_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != hidden_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu6(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
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
