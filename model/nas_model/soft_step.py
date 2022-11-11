import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from model.layers.softconv import SoftChannelConv2d, SoftChannelBatchNormConv2d, SoftKernelConv2d


class SoftInvertedResidualBlock(nn.Module):
    expansion = 6

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, expansion=None):
        super(SoftInvertedResidualBlock, self).__init__()
        if not expansion:
            self.expansion = expansion
        hidden_planes = round(inplanes * self.expansion)

        self.conv1 = SoftChannelConv2d(
            inplanes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = SoftKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                      stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

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

    def forward(self, x):
        out = F.relu6(self.conv1(x))
        out = F.relu6(self.conv2(out))
        # out = self.conv1(x)
        # out = self.conv2(out)
        self.hidden_indicators = self.conv1.sample_indicator()
        out = torch.mul(out, self.hidden_indicators.reshape(
            (1, self.hidden_indicators.shape[0], 1, 1)))

        out = self.conv3(out)
        out = out + self.shortcut(x)
        out = F.relu6(out)
        self.outplane_indicators = self.conv3.sample_indicator()
        out = torch.mul(out, self.outplane_indicators.reshape(
            (1, self.outplane_indicators.shape[0], 1, 1)))
        return out


class SoftStep(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None, block=SoftInvertedResidualBlock) -> None:
        super(SoftStep, self).__init__()
        with open(path, 'r') as f:
            struc = json.load(f)
        output_channel = struc["b0"]["conv_in"]
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channel, output_channel,
                      kernel_size=3, stride=struc["b0"]["stride_in"], padding=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU6(inplace=True)
        )

        input_channel = output_channel
        layers = []
        for block_config in struc["blocks"]:
            e, c, n, k, s = block_config['e'], block_config['c'], block_config['n'], block_config['k'], block_config['s']
            output_channel = c
            for i in range(n):
                layers.append(
                    block(input_channel, output_channel, k, s if i == 0 else 1, e))
                input_channel = output_channel
        self.blocks = nn.Sequential(*layers)

        input_channel = output_channel
        output_channel = struc["b0"]["conv_out"]
        self.conv_out = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU6(inplace=True)
        )

        self.fc = nn.Linear(output_channel, num_classes)
        return

    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.conv_out(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def model_parameters(self):
        for name, param in self.named_parameters():
            if name.__contains__("weight") or name.__contains__("bias"):
                yield param

    def arch_parameters(self):
        for name, param in self.named_parameters():
            if name.__contains__("alpha"):
                yield param

    def search_result_list(self):
        for name, _ in self.named_parameters():
            if name.__contains__("channel_alpha"):
                segments = name.split(".")
                layer = eval("self.blocks[int(segments[1])]."+segments[2])
                yield min(layer.channel_alpha.item(), 1)*layer.out_channels
            if name.__contains__("kernel_alpha"):
                segments = name.split(".")
                layer = eval("self.blocks[int(segments[1])]."+segments[2])
                yield min(layer.kernel_alpha.item(), 1)*int(layer.kernel_size/2)


if __name__ == '__main__':
    # train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
    # print(input_channel, inputdim, nclass)
    # # model = IRFBlock(in_channels=input_channel, out_channels=3)
    # model = SoftStep(input_channel, inputdim, nclass)
    # x = torch.rand(3, 32, 32)
    # preds = model(x)
    pass
