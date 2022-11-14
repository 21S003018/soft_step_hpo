import torch.nn as nn
import json
import torch.nn.functional as F
from model.basicblocks import ResidualBlock, InvertedResidualBlock, TInvertedResidualBlock


class Eval(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None, block=InvertedResidualBlock) -> None:
        super(Eval, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        block = InvertedResidualBlock if config["type"] == "linear" else ResidualBlock
        output_channel = config["b0"]["conv_in"]
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channel, output_channel,
                      kernel_size=3, stride=config["b0"]["stride_in"], padding=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU6(inplace=True)
        )

        input_channel = output_channel
        layers = []
        for block_config in config["blocks"]:
            e, c, n, k, s = block_config['e'], block_config['c'], block_config['n'], block_config['k'], block_config['s']
            output_channel = c
            for i in range(n):
                layers.append(
                    block(input_channel, output_channel, k, s if i == 0 else 1, e))
                input_channel = output_channel
        self.blocks = nn.Sequential(*layers)

        input_channel = output_channel
        output_channel = config["b0"]["conv_out"]
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

class TEval(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None, block=TInvertedResidualBlock) -> None:
        super(TEval, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        block = TInvertedResidualBlock if config["type"] == "linear" else ResidualBlock
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(output_channel)
        # block
        out_planes = output_channel
        blocks = []
        for block_config in config["block"]:
            t, c1, k, c2, s = block_config['type'], block_config['c1'], block_config['k'], block_config['c2'], block_config['s']
            in_planes = out_planes
            out_planes = c2
            blocks.append(TInvertedResidualBlock(in_planes, out_planes, kernel_size=k,stride=s,expansion=c1/in_planes,type=t))
        self.blocks = nn.Sequential(*blocks)
        # post conv
        input_channel = out_planes
        output_channel = config["layer"]["conv_out"]
        self.conv_out, self.bn_out = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(output_channel)
        # final
        self.fc = nn.Linear(output_channel, num_classes)
        return

    def forward(self, x):
        x = F.relu6(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)
        x = F.relu6(self.bn_out(self.conv_out(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
