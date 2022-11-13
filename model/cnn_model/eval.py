import torch.nn as nn
import json
import torch.nn.functional as F
from model.basicblocks import ResidualBlock, InvertedResidualBlock


class Eval(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None, block=InvertedResidualBlock) -> None:
        super(Eval, self).__init__()
        with open(path, 'r') as f:
            struc = json.load(f)
        block = InvertedResidualBlock if struc["block_type"] == "linear" else ResidualBlock
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
