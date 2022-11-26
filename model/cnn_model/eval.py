import torch.nn as nn
import json
import torch.nn.functional as F
from model.basicblocks import BasicBlock, ResidualBlock, InvertedResidualBlock


class Eval(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None, block=InvertedResidualBlock) -> None:
        super(Eval, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        assert config["type"] == "linear"
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = nn.Conv2d(
            input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(output_channel)
        # block
        out_planes = output_channel
        blocks = []
        for block_config in config["block"]:
            t, c1, k, c2, s = block_config['type'], block_config[
                'c1'], block_config['k'], block_config['c2'], block_config['s']
            in_planes = out_planes
            out_planes = c2
            blocks.append(block(
                in_planes, out_planes, kernel_size=k, stride=s, expansion=c1/in_planes, type=t))
        self.blocks = nn.Sequential(*blocks)
        # post conv
        input_channel = out_planes
        output_channel = config["layer"]["conv_out"]
        self.conv_out, self.bn_out = nn.Conv2d(
            input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(output_channel)
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


class BottleneckEval(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None, block=ResidualBlock) -> None:
        super(BottleneckEval, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        assert config["type"] == "bottleneck"
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = nn.Conv2d(
            input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(output_channel)
        # block
        out_planes = output_channel
        blocks = []
        for block_config in config["block"]:
            t, c1, k, c2, s = block_config['type'], block_config[
                'c1'], block_config['k'], block_config['c2'], block_config['s']
            in_planes = out_planes
            out_planes = c2
            hidden_planes = c1
            blocks.append(block(
                in_planes, hidden_planes, kernel_size=k, stride=s, expansion=out_planes/hidden_planes, type=t))
        self.blocks = nn.Sequential(*blocks)
        # post conv
        input_channel = out_planes
        output_channel = config["layer"]["conv_out"]
        self.conv_out, self.bn_out = nn.Conv2d(
            input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(output_channel)
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


class ShallowEval(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None, block=BasicBlock) -> None:
        super(ShallowEval, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        assert config["type"] == "shallow"
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = nn.Conv2d(
            input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(output_channel)
        # block
        out_planes = output_channel
        blocks = []
        for block_config in config["block"]:
            t, c1, c2, s = block_config['type'], block_config['c1'], block_config['c2'], block_config['s']
            in_planes = out_planes
            hidden_planes = c1
            out_planes = c2
            blocks.append(block(in_planes, hidden_planes, out_planes, s, t))
        self.blocks = nn.Sequential(*blocks)
        # post conv
        input_channel = out_planes
        output_channel = config["layer"]["conv_out"]
        self.conv_out, self.bn_out = nn.Conv2d(
            input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(output_channel)
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


def get_block_type(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["type"]
