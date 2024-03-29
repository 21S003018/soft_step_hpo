import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from const import *
import math
import json
from model.nas_model.layers.darts import DartsChannelConv2d, DartsKernelConv2d


class LinearBlock(nn.Module):  # normal block or reduction block
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expansion=4):
        super(LinearBlock, self).__init__()

        hidden_planes = round(in_planes * expansion)

        self.conv1 = DartsChannelConv2d(
            in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = DartsKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                       stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = DartsChannelConv2d(hidden_planes, out_planes,
                                        kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = F.relu6(self.bn3(self.conv3(out)))
        return out


class SkipLinearBlock(nn.Module):  # skip block
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expansion=4):
        super(SkipLinearBlock, self).__init__()
        assert in_planes == out_planes
        assert stride == 1

        hidden_planes = round(in_planes * expansion)

        self.conv1 = DartsChannelConv2d(
            in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = DartsKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                       stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = nn.Conv2d(hidden_planes, out_planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = F.relu6(self.bn3(self.conv3(out)))
        return out


class LinearStage(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, expansion, n_skip):
        super(LinearStage, self).__init__()
        self.stride = stride
        self.out_planes = out_planes
        self.block = LinearBlock(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, expansion=expansion)
        skips = []
        for _ in range(n_skip):
            skips.append(SkipLinearBlock(
                out_planes, out_planes, kernel_size=kernel_size, stride=1, expansion=expansion))
        self.skips = nn.Sequential(*skips)
        return

    def forward(self, x):
        x = self.block(x)
        output_scale = None
        if self.block.conv3.channel_alpha.argmax() == 0:
            output_scale = 1
        elif self.block.conv3.channel_alpha.argmax() == 1:
            output_scale = 0.75
        elif self.block.conv3.channel_alpha.argmax() == 2:
            output_scale = 0.5
        for skip in self.skips:
            x = F.relu6(skip(x) + x)
            x[:, :int(output_scale*self.out_planes), :, :] = 0
        return x


class DARTS(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None) -> None:
        super(DARTS, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        self.block_type = config["type"]
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = DartsChannelConv2d(
            input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(output_channel)
        # pre block
        block_config = config["block"]
        in_planes = output_channel
        expansion, out_planes, kernel_size, stride = block_config[
            'e'], block_config['c'], block_config['k'], block_config['s']
        self.block_in = LinearBlock(
            in_planes, out_planes, kernel_size, stride, expansion)
        # stages
        stages = []
        for stage_config in config["stage"]:
            e, c, n, k, s = stage_config['e'], stage_config['c'], stage_config['n'], stage_config['k'], stage_config['s']
            in_planes = out_planes
            out_planes = c
            stages.append(LinearStage(
                in_planes, out_planes, kernel_size=k, stride=s, expansion=e, n_skip=n-1))
        self.stages = nn.Sequential(*stages)
        # post conv
        input_channel = out_planes
        output_channel = config["layer"]["conv_out"]
        self.conv_out, self.bn_out = DartsChannelConv2d(
            input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(output_channel)
        # final
        self.fc = nn.Linear(output_channel, num_classes)
        return

    def forward(self, x):
        x = F.relu6(self.bn_in(self.conv_in(x)))
        x = self.block_in(x)
        for stage in self.stages:
            x = stage(x)
        x = F.relu6(self.bn_out(self.conv_out(x)))
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

    def generate_config(self, full=False):
        config = {"type": self.block_type}
        # layer
        config["layer"] = {
            "conv_in": self.conv_in.get_channel(),
            "conv_out": self.conv_out.get_channel()
        }
        # block
        config["block"] = [{
            "type": "normal",
            "c1": self.block_in.conv1.get_channel(),
            "k": self.block_in.conv2.get_kernel(),
            "c2": self.block_in.conv3.get_channel(),
            "s": 1
        }]
        # stage
        for stage in self.stages:
            config["block"].append({
                "type": "normal" if stage.stride == 1 else "reduction",
                "c1": stage.block.conv1.get_channel(),
                "k": stage.block.conv2.get_kernel(),
                "c2": stage.block.conv3.get_channel(),
                "s": stage.stride
            })
            for skip in stage.skips:
                config["block"].append({
                    "type": "skip",
                    "c1": skip.conv1.get_channel(),
                    "k": skip.conv2.get_kernel(),
                    "c2": stage.block.conv3.get_channel(),
                    "s": 1
                })
        return config
