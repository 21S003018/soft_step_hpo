import torch.nn as nn
import json
import torch.nn.functional as F
from model.layers.softconv import SoftChannelConv2d, SoftKernelConv2d


class Block(nn.Module):
    def __init__(self) -> None:
        super(Block, self).__init__()
        return

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SoftInvertedResidualBlock(Block):
    expansion = 6

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, expansion=None):
        super(SoftInvertedResidualBlock, self).__init__()
        if not expansion:
            self.expansion = expansion
        hidden_planes = round(inplanes * self.expansion)

        # pw
        self.conv1 = SoftChannelConv2d(inplanes, hidden_planes,
                                       kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)
        # dw
        self.conv2 = SoftKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                      stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)
        # pw-linear
        self.conv3 = SoftChannelConv2d(hidden_planes, planes,
                                       kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != hidden_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )


class SoftStep(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None, block=SoftInvertedResidualBlock) -> None:
        super(SoftStep, self).__init__()
        if path is None:
            block_input_channel = 32
            self.conv_in = self.set_conv_in(input_channel, block_input_channel)
            self.blocks, output_channel = self.set_default_blocks(
                block, block_input_channel)
            block_output_channel = output_channel
        else:
            with open(path, 'r') as f:
                struc = json.load(f)
            block_input_channel = struc["b0"]["conv_in"]
            self.conv_in = self.set_conv_in(
                input_channel, block_input_channel, struc["b0"]["stride_in"])
            layers = []
            input_channel = block_input_channel
            for block_config in struc["blocks"]:
                e, c, n, k, s = block_config['e'], block_config['c'], block_config['n'], block_config['k'], block_config['s']
                output_channel = c
                for i in range(n):
                    layers.append(
                        block(input_channel, output_channel, k, s if i == 0 else 1, e))
                    input_channel = output_channel
            self.blocks = nn.Sequential(*layers)
            block_output_channel = struc["b0"]["conv_out"]

        self.conv_out = nn.Sequential(
            nn.Conv2d(output_channel, block_output_channel, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(block_output_channel),
            nn.ReLU6(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block_output_channel, num_classes)
        return

    def set_conv_in(self, input_channel, out_channel, stride=2):
        return nn.Sequential(
            nn.Conv2d(input_channel, out_channels=out_channel,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

    def set_default_blocks(self, block, block_input_channel):
        self.cfgs = [
            # e, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            # [6,  32, 3, 2],
            # [6,  64, 4, 2],
            # [6,  96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]
        layers = []
        input_channel = block_input_channel
        for t, c, n, s in self.cfgs:
            output_channel = c
            for i in range(n):
                layers.append(
                    block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        blocks = nn.Sequential(*layers)
        return blocks, output_channel

    def forward(self, x):
        x = self.conv_in(x)
        print(x.size())
        x = self.blocks(x)
        print(x.size())
        x = self.conv_out(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def generate_struc(self):

        return

    def model_parameters(self):
        for name, param in self.named_parameters():
            if name.__contains__("weight") or name.__contains__("bias"):
                yield param

    def arch_parameters(self):
        for name, param in self.named_parameters():
            if name.__contains__("base") or name.__contains__("controller"):
                yield param


if __name__ == '__main__':
    # train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
    # print(input_channel, inputdim, nclass)
    # # model = IRFBlock(in_channels=input_channel, out_channels=3)
    # model = SoftStep(input_channel, inputdim, nclass)
    # x = torch.rand(3, 32, 32)
    # preds = model(x)
    pass
