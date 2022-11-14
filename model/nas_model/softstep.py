import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from model.nas_model.layers.softconv import SoftConv2d, SoftChannelConv2d, SoftKernelConv2d


class SoftResidualBlock(nn.Module):
    def __init__(self, inplanes, hidden_planes, kernel_size=3, stride=1, expansion=4):
        super(SoftInvertedResidualBlock, self).__init__()
        self.expansion = expansion
        out_planes = self.expansion*hidden_planes
        self.conv1 = SoftChannelConv2d(
            inplanes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = SoftConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                stride=stride, padding=int(kernel_size/2), bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = SoftChannelConv2d(hidden_planes, out_planes,
                                       kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(inplanes, out_planes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.conv1_indicators = self.conv1.sample_indicator()
        out = torch.mul(out, self.conv1_indicators.reshape(
            (1, self.conv1_indicators.shape[0], 1, 1)))

        out = F.relu(self.bn2(self.conv2(out)))
        self.conv2_indicators = self.conv2.sample_channel_indicator()
        out = torch.mul(out, self.conv2_indicators.reshape(
            (1, self.conv2_indicators.shape[0], 1, 1)))

        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        self.conv3_indicators = self.conv3.sample_indicator()
        out = torch.mul(out, self.conv3_indicators.reshape(
            (1, self.conv3_indicators.shape[0], 1, 1)))
        return out


class SoftInvertedResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, expansion=4):
        super(SoftInvertedResidualBlock, self).__init__()
        self.expansion = expansion
        hidden_planes = round(inplanes * self.expansion)
        print(self.expansion)
        self.arch_opt = False

        self.conv1 = SoftChannelConv2d(
            inplanes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = SoftKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                      stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = SoftChannelConv2d(hidden_planes, planes,
                                       kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(inplanes, planes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        if self.arch_opt:
            out = F.relu6(self.bn1(self.conv1(x)))
            out = F.relu6(self.bn2(self.conv2(out)))
            out = torch.mul(out, self.conv1.channel_indicators)

            out = self.bn3(self.conv3(out))
            out = out + self.shortcut(x)
            out = F.relu6(out)
            out = torch.mul(out, self.conv3.channel_indicators)
            self.arch_opt = False
        else:
            out = F.relu6(self.bn1(self.conv1(x)))
            out = F.relu6(self.bn2(self.conv2(out, False)))
            out = torch.mul(out, self.conv1.channel_indicators.data)

            out = self.bn3(self.conv3(out))
            out = out + self.shortcut(x)
            out = F.relu6(out)
            out = torch.mul(out, self.conv3.channel_indicators.data)
        return out


class SoftStep(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None, block=SoftInvertedResidualBlock) -> None:
        super(SoftStep, self).__init__()
        with open(path, 'r') as f:
            struc = json.load(f)
        self.block_type = struc["block_type"]
        block = SoftInvertedResidualBlock if self.block_type == "linear" else SoftResidualBlock
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

    def forward(self, x, arch_opt=False):
        if arch_opt:
            self.update_indicators()
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

    def update_indicators(self):
        for conv_block in self.blocks:
            conv_block.arch_opt = True
            conv_block.conv1.update_channel_indicators()
            conv_block.conv2.update_kernel_mask()
            conv_block.conv3.update_channel_indicators()
            if self.block_type == "bottleneck":
                conv_block.conv2.update_channel_indicators()
        return


if __name__ == '__main__':
    # train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
    # print(input_channel, inputdim, nclass)
    # # model = IRFBlock(in_channels=input_channel, out_channels=3)
    # model = SoftStep(input_channel, inputdim, nclass)
    # x = torch.rand(3, 32, 32)
    # preds = model(x)
    pass
