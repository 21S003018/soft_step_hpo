import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from const import *
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math
import utils


class HPOConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=None, groups=1):
        super(HPOConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups

        self.weight = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))
        self.reset_parameters()
        # init channel indicators and kernel mask
        self.channel_indicators = torch.ones(
            self.out_channels).reshape((1, self.out_channels, 1, 1))
        self.kernel_indicators = torch.ones(self.kernel_size)
        self.mask = torch.ones((self.kernel_size, self.kernel_size))
        pass

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        return

    def forward(self, x):
        masked_weight = torch.mul(self.weight, self.mask)
        x = F.conv2d(x, weight=masked_weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        return x

    def update_channel_indicators(self, value: int, full=False) -> None:
        if full:
            value = len(self.channel_indicators[0])
        self.channel_indicators[:, value:, :, :] = 0
        self.channel_indicators[:, :value, :, :] = 1
        if torch.cuda.is_available():
            self.channel_indicators = self.channel_indicators.cuda(
                self.weight.device)
        return

    def update_kernel_mask(self, value: int, full=False) -> None:
        if full:
            value = len(self.kernel_indicators)
        self.kernel_indicators = torch.zeros(int(self.kernel_size/2))
        self.kernel_indicators[:value] = 1
        self.mask = torch.ones((self.kernel_size, self.kernel_size))
        for index, _ in enumerate(self.kernel_indicators):
            self.mask[index:self.kernel_size-index, index:self.kernel_size -
                      index] = self.kernel_indicators[int(self.kernel_size/2)-index-1]
        if torch.cuda.is_available():
            self.mask = self.mask.cuda(self.weight.device)
        return


class HPOChannelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=1):
        super(HPOChannelConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups

        self.weight = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))
        self.reset_parameters()
        # init channel indicators
        self.channel_indicators = torch.ones(
            self.out_channels).reshape((1, self.out_channels, 1, 1))
        return

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        return

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        return x

    def update_channel_indicators(self, value: int, full: False) -> None:
        if full:
            value = len(self.channel_indicators[0])
        self.channel_indicators[:, value:, :, :] = 0
        self.channel_indicators[:, :value, :, :] = 1
        if torch.cuda.is_available():
            self.channel_indicators = self.channel_indicators.cuda(
                self.weight.device)
        return


class HPOKernelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=1):
        super(HPOKernelConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups

        self.weight = Parameter(torch.Tensor(
            out_channels, int(in_channels/self.groups), kernel_size, kernel_size))
        self.reset_parameters()
        # init kernel mask
        self.kernel_indicators = torch.ones(self.kernel_size)
        self.mask = torch.ones((self.kernel_size, self.kernel_size))
        return

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        return

    def forward(self, x):
        masked_weight = torch.mul(self.weight, self.mask)
        x = F.conv2d(x, weight=masked_weight, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        return x

    def update_kernel_mask(self, value: int, full=False) -> None:
        if full:
            value = len(self.kernel_indicators)
        self.kernel_indicators = torch.zeros(int(self.kernel_size/2))
        self.kernel_indicators[:value] = 1
        self.mask = torch.ones((self.kernel_size, self.kernel_size))
        for index, _ in enumerate(self.kernel_indicators):
            self.mask[index:self.kernel_size-index, index:self.kernel_size -
                      index] = self.kernel_indicators[int(self.kernel_size/2)-index-1]
        if torch.cuda.is_available():
            self.kernel_indicators = self.kernel_indicators.cuda(
                self.weight.device)
            self.mask = self.mask.cuda(self.weight.device)
        return


class LinearBlock(nn.Module):  # normal block or reduction block
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expansion=4):
        super(LinearBlock, self).__init__()

        hidden_planes = round(in_planes * expansion)

        self.conv1 = HPOChannelConv2d(
            in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = HPOKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                     stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = HPOChannelConv2d(hidden_planes, out_planes,
                                      kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = torch.mul(out, self.conv1.channel_indicators)
        out = F.relu6(self.bn3(self.conv3(out)))
        out = torch.mul(out, self.conv3.channel_indicators)
        return out

    def update_indicators(self, c1, k, c2, full=False):
        self.conv1.update_channel_indicators(c1, full)
        self.conv2.update_kernel_mask(k, full)
        self.conv3.update_channel_indicators(c2, full)
        return


class SkipLinearBlock(nn.Module):  # skip block
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expansion=4):
        super(SkipLinearBlock, self).__init__()
        assert in_planes == out_planes
        assert stride == 1

        hidden_planes = round(in_planes * expansion)

        self.conv1 = HPOChannelConv2d(
            in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = HPOKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                     stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = nn.Conv2d(hidden_planes, out_planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = torch.mul(out, self.conv1.channel_indicators)
        out = self.bn3(self.conv3(out))
        return out

    def update_indicators(self, c1, k, full=False):
        self.conv1.update_channel_indicators(c1, full)
        self.conv2.update_kernel_mask(k, full)
        return


class LinearStage(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, expansion, n_skip):
        super(LinearStage, self).__init__()
        self.stride = stride
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
        for skip in self.skips:
            x = F.relu6(torch.mul(skip(x),
                        self.block.conv3.channel_indicators) + x)
        return x

    def update_indicators(self, config, full=False):
        self.block.update_indicators(
            config["block"]["c1"], config["block"]["k"], config["block"]["c2"], full)
        for i, skip in enumerate(self.skips):
            skip.update_indicators(
                config["skip"][i]["c1"], config["skip"][i]["k"], full)
        return


class LinearSupernet(nn.Module):
    def __init__(self, input_channel, ndim, num_classes) -> None:
        super(LinearSupernet, self).__init__()
        self.path = LINEARSEARCHSPACE
        with open(self.path, 'r') as f:
            config = json.load(f)
        self.block_type = config["type"]
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = HPOChannelConv2d(
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
        self.conv_out, self.bn_out = HPOChannelConv2d(
            input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(output_channel)
        # final
        self.fc = nn.Linear(output_channel, num_classes)
        return

    def forward(self, x):
        x = F.relu6(self.bn_in(self.conv_in(x)))
        x = torch.mul(x, self.conv_in.channel_indicators)
        x = self.block_in(x)
        for stage in self.stages:
            x = stage(x)
        x = F.relu6(self.bn_out(self.conv_out(x)))
        x = torch.mul(x, self.conv_out.channel_indicators)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def generate_config(self, full=False):
        config = {"type": self.block_type}
        # layer
        config["layer"] = {
            "conv_in": self.conv_in.out_channels if full else self.conv_in.channel_indicators.sum().item(),
            "conv_out": self.conv_out.out_channels if full else self.conv_out.channel_indicators.sum().item()
        }
        # block
        config["block"] = [{
            "type": "normal",
            "c1": self.block_in.conv1.out_channels if full else self.block_in.conv1.channel_indicators.sum().item(),
            "k": self.block_in.conv2.kernel_size if full else self.block_in.conv2.kernel_indicators.sum().item(),
            "c2": self.block_in.conv3.out_channels if full else self.block_in.conv3.channel_indicators.sum().item(),
            "s": 1
        }]
        # stage
        for stage in self.stages:
            config["block"].append({
                "type": "normal" if stage.stride == 1 else "reduction",
                "c1": stage.block.conv1.out_channels if full else stage.block.conv1.channel_indicators.sum().item(),
                "k": stage.block.conv2.kernel_size if full else stage.block.conv2.kernel_indicators.sum().item(),
                "c2": stage.block.conv3.out_channels if full else stage.block.conv3.channel_indicators.sum().item(),
                "s": stage.stride
            })
            for skip in stage.skips:
                config["block"].append({
                    "type": "skip",
                    "c1": skip.conv1.out_channels if full else skip.conv1.channel_indicators.sum().item(),
                    "k": skip.conv2.kernel_size if full else skip.conv2.kernel_indicators.sum().item(),
                    "c2": stage.block.conv3.out_channels if full else stage.block.conv3.channel_indicators.sum().item(),
                    "s": 1
                })
        return config

    def update_indicators(self, config: dict, full=False):
        # layer
        self.conv_in.update_channel_indicators(
            config["layer"]["conv_in"], full)
        self.conv_out.update_channel_indicators(
            config["layer"]["conv_out"], full)
        # block
        self.block_in.update_indicators(
            config["block"]["c1"], config["block"]["k"], config["block"]["c2"], full)
        # stage
        for i, stage in enumerate(self.stages):
            stage.update_indicators(config["stage"][i], full)
        return


class BottleneckSupernet(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None) -> None:
        super(BottleneckSupernet, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        self.block_type = config["type"]
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = HPOChannelConv2d(
            input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(output_channel)
        # stages
        in_planes = output_channel
        stages = []
        for stage_config in config["stage"]:
            e, c, n, k, s = stage_config['e'], stage_config['c'], stage_config['n'], stage_config['k'], stage_config['s']
            hidden_planes = c
            stages.append(BottleneckStage(
                in_planes, hidden_planes, kernel_size=k, stride=s, expansion=e, n_skip=n-1))
            in_planes = hidden_planes*e
        self.stages = nn.Sequential(*stages)
        # post conv
        input_channel = in_planes
        output_channel = config["layer"]["conv_out"]
        self.conv_out, self.bn_out = HPOChannelConv2d(
            input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(output_channel)
        # final
        self.fc = nn.Linear(output_channel, num_classes)
        return

    def forward(self, x, arch_opt=False):
        if arch_opt:
            self.update_indicators()
        x = F.relu6(self.bn_in(self.conv_in(x)))
        if arch_opt:
            x = torch.mul(x, self.conv_in.channel_indicators)
        for stage in self.stages:
            x = stage(x, arch_opt)
        x = F.relu6(self.bn_out(self.conv_out(x)))
        if arch_opt:
            x = torch.mul(x, self.conv_out.channel_indicators)
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
            "conv_in": self.conv_in.out_channels if full else int(self.conv_in.channel_alpha*self.conv_in.out_channels),
            "conv_out": self.conv_out.out_channels if full else int(self.conv_out.channel_alpha*self.conv_out.out_channels)
        }
        # block
        config["block"] = []
        # stage
        for stage in self.stages:
            config["block"].append({
                "type": "normal" if stage.stride == 1 else "reduction",
                "c1": stage.block.conv1.out_channels if full else int(stage.block.conv1.channel_alpha*stage.block.conv1.out_channels),
                "k": stage.block.conv2.kernel_size if full else int(stage.block.conv2.kernel_alpha*int(stage.block.conv2.kernel_size/2))*2+1,
                # "c2": stage.block.conv2.out_channels if full else int(stage.block.conv2.channel_alpha*stage.block.conv2.out_channels),
                # "c3": stage.block.conv3.out_channels if full else int(stage.block.conv3.channel_alpha*stage.block.conv3.out_channels),
                "c2": stage.block.conv3.out_channels if full else int(stage.block.conv3.channel_alpha*stage.block.conv3.out_channels),
                "s": stage.stride
            })
            for skip in stage.skips:
                config["block"].append({
                    "type": "skip",
                    "c1": skip.conv1.out_channels if full else int(skip.conv1.channel_alpha*skip.conv1.out_channels),
                    "k": skip.conv2.kernel_size if full else int(skip.conv2.kernel_alpha*int(skip.conv2.kernel_size/2))*2+1,
                    # "c2": skip.conv2.out_channels if full else int(skip.conv2.channel_alpha*skip.conv2.out_channels),
                    # "c3": stage.block.conv3.out_channels if full else int(stage.block.conv3.channel_alpha*stage.block.conv3.out_channels),
                    "c2": stage.block.conv3.out_channels if full else int(stage.block.conv3.channel_alpha*stage.block.conv3.out_channels),
                    "s": 1
                })
        return config

    def update_indicators(self):
        # layer
        self.conv_in.update_channel_indicators()
        self.conv_out.update_channel_indicators()
        # stage
        for stage in self.stages:
            stage.update_indicators()
        return

    def protect_controller(self):
        self.conv_in.protect_controller()
        for stage in self.stages:
            stage.protect_controller()
        self.conv_out.protect_controller()
        return


class ShallowSupernet(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None) -> None:
        super(ShallowSupernet, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        self.block_type = config["type"]
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = HPOChannelConv2d(
            input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(output_channel)
        # stages
        in_planes = output_channel
        stages = []
        for stage_config in config["stage"]:
            e, c, n, k, s = stage_config['e'], stage_config['c'], stage_config['n'], stage_config['k'], stage_config['s']
            hidden_planes = c
            stages.append(ShallowStage(
                in_planes, hidden_planes, kernel_size=k, stride=s, expansion=e, n_skip=n-1))
            in_planes = hidden_planes*e
        self.stages = nn.Sequential(*stages)
        # post conv
        input_channel = in_planes
        output_channel = config["layer"]["conv_out"]
        self.conv_out, self.bn_out = HPOChannelConv2d(
            input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(output_channel)
        # final
        self.fc = nn.Linear(output_channel, num_classes)
        return

    def forward(self, x, arch_opt=False):
        if arch_opt:
            self.update_indicators()
        x = F.relu6(self.bn_in(self.conv_in(x)))
        if arch_opt:
            x = torch.mul(x, self.conv_in.channel_indicators)
        for stage in self.stages:
            x = stage(x, arch_opt)
        x = F.relu6(self.bn_out(self.conv_out(x)))
        if arch_opt:
            x = torch.mul(x, self.conv_out.channel_indicators)
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
            "conv_in": self.conv_in.out_channels if full else int(self.conv_in.channel_alpha*self.conv_in.out_channels),
            "conv_out": self.conv_out.out_channels if full else int(self.conv_out.channel_alpha*self.conv_out.out_channels)
        }
        # block
        config["block"] = []
        # stage
        for stage in self.stages:
            config["block"].append({
                "type": "normal" if stage.stride == 1 else "reduction",
                "c1": stage.block.conv1.out_channels if full else int(stage.block.conv1.channel_alpha*stage.block.conv1.out_channels),
                "c2": stage.block.conv2.out_channels if full else int(stage.block.conv2.channel_alpha*stage.block.conv2.out_channels),
                "s": stage.stride
            })
            for skip in stage.skips:
                config["block"].append({
                    "type": "skip",
                    "c1": skip.conv1.out_channels if full else int(skip.conv1.channel_alpha*skip.conv1.out_channels),
                    "c2": stage.block.conv2.out_channels if full else int(stage.block.conv2.channel_alpha*stage.block.conv2.out_channels),
                    "s": 1
                })
        return config

    def update_indicators(self):
        # layer
        self.conv_in.update_channel_indicators()
        self.conv_out.update_channel_indicators()
        # stage
        for stage in self.stages:
            stage.update_indicators()
        return

    def protect_controller(self):
        self.conv_in.protect_controller()
        for stage in self.stages:
            stage.protect_controller()
        self.conv_out.protect_controller()
        return
