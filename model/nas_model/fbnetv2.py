import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from model.nas_model.layers.fbnetv2conv import FBnetChannelConv2d
from model.nas_model.layers.darts import DartsKernelConv2d


class LinearBlock(nn.Module):  # normal block or reduction block
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expansion=4):
        super(LinearBlock, self).__init__()

        hidden_planes = round(in_planes * expansion)

        self.conv1 = FBnetChannelConv2d(
            in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = DartsKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                       stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = FBnetChannelConv2d(hidden_planes, out_planes,
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

        self.conv1 = FBnetChannelConv2d(
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
        self.block = LinearBlock(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, expansion=expansion)
        skips = []
        for _ in range(n_skip):
            skips.append(SkipLinearBlock(
                out_planes, out_planes, kernel_size=kernel_size, stride=1, expansion=expansion))
        self.skips = nn.Sequential(*skips)
        return

    def forward(self, x=False):
        x = self.block(x)
        for skip in self.skips:
            x = F.relu6(torch.mul(
                skip(x), self.block.conv3.sample_indicator().unsqueeze(2).unsqueeze(3)) + x)
        return x


class FBnet(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None) -> None:
        super(FBnet, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        self.block_type = config["type"]
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = FBnetChannelConv2d(
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
        self.conv_out, self.bn_out = FBnetChannelConv2d(
            input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(output_channel)
        # final
        self.fc = nn.Linear(output_channel, num_classes)
        return

    def forward(self, x=False):
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


class Bottleneck(nn.Module):  # normal block or reduction block
    def __init__(self, in_planes, hidden_planes, kernel_size=3, stride=1, expansion=4):
        super(Bottleneck, self).__init__()

        out_planes = round(hidden_planes * expansion)

        self.conv1 = FBnetChannelConv2d(
            in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = DartsKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                       stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = FBnetChannelConv2d(hidden_planes, out_planes,
                                        kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.short_cut = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = F.relu6(self.bn3(self.conv3(out))+self.short_cut(x))
        return out


class SkipBottleneck(nn.Module):  # skip block
    def __init__(self, in_planes, hidden_planes, kernel_size=3, stride=1, expansion=4):
        super(SkipBottleneck, self).__init__()
        out_planes = round(hidden_planes * expansion)
        assert in_planes == out_planes
        assert stride == 1

        self.conv1 = FBnetChannelConv2d(
            in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = DartsKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size, stride=stride, padding=int(
            kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = nn.Conv2d(hidden_planes, out_planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = F.relu6(self.bn3(self.conv3(out)))
        return out


class BottleneckStage(nn.Module):
    def __init__(self, in_planes, hidden_planes, kernel_size, stride, expansion, n_skip):
        super(BottleneckStage, self).__init__()
        self.stride = stride
        self.block = Bottleneck(
            in_planes, hidden_planes, kernel_size=kernel_size, stride=stride, expansion=expansion)
        skips = []
        for _ in range(n_skip):
            skips.append(SkipBottleneck(
                hidden_planes*expansion, hidden_planes, kernel_size=kernel_size, stride=1, expansion=expansion))
        self.skips = nn.Sequential(*skips)
        return

    def forward(self, x=False):
        x = self.block(x)
        for skip in self.skips:
            x = F.relu6(torch.mul(
                skip(x), self.block.conv3.sample_indicator().unsqueeze(2).unsqueeze(3)) + x)
        return x


class BottleneckFBnet(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None) -> None:
        super(BottleneckFBnet, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        self.block_type = config["type"]
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = FBnetChannelConv2d(
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
        self.conv_out, self.bn_out = FBnetChannelConv2d(
            input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(output_channel)
        # final
        self.fc = nn.Linear(output_channel, num_classes)
        return

    def forward(self, x=False):
        x = F.relu6(self.bn_in(self.conv_in(x)))
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


class Shallow(nn.Module):  # normal block or reduction block
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expansion=1):
        super(Shallow, self).__init__()
        self.conv1 = FBnetChannelConv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=int(kernel_size/2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = FBnetChannelConv2d(out_planes, out_planes, kernel_size=kernel_size,
                                        stride=1, padding=int(kernel_size/2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out))+self.shortcut(x))
        return out


class SkipShallow(nn.Module):  # skip block
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expansion=1):
        super(SkipShallow, self).__init__()
        assert in_planes == out_planes
        assert stride == 1

        self.conv1 = FBnetChannelConv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=int(kernel_size/2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size,
                               stride=1, padding=int(kernel_size/2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        return out


class ShallowStage(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, expansion, n_skip):
        super(ShallowStage, self).__init__()
        self.stride = stride
        self.block = Shallow(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride)
        skips = []
        for _ in range(n_skip):
            skips.append(SkipShallow(
                out_planes, out_planes, kernel_size=kernel_size, stride=1))
        self.skips = nn.Sequential(*skips)
        return

    def forward(self, x=False):
        x = self.block(x)
        for skip in self.skips:
            x = F.relu6(torch.mul(
                skip(x), self.block.conv2.sample_indicator().unsqueeze(2).unsqueeze(3)) + x)
        return x


class ShallowFBnet(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None) -> None:
        super(ShallowFBnet, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        self.block_type = config["type"]
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = FBnetChannelConv2d(
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
        self.conv_out, self.bn_out = FBnetChannelConv2d(
            input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(output_channel)
        # final
        self.fc = nn.Linear(output_channel, num_classes)
        return

    def forward(self, x=False):
        x = F.relu6(self.bn_in(self.conv_in(x)))
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
        # stage
        for stage in self.stages:
            config["block"].append({
                "type": "normal" if stage.stride == 1 else "reduction",
                "c1": stage.block.conv1.get_channel(),
                "c2": stage.block.conv3.get_channel(),
                "s": stage.stride
            })
            for skip in stage.skips:
                config["block"].append({
                    "type": "skip",
                    "c1": skip.conv1.get_channel(),
                    "c2": stage.block.conv3.get_channel(),
                    "s": 1
                })
        return config


if __name__ == '__main__':
    # train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
    # print(input_channel, inputdim, nclass)
    # # model = IRFBlock(in_channels=input_channel, out_channels=3)
    # model = FBnet(input_channel, inputdim, nclass)
    # x = torch.rand(3, 32, 32)
    # preds = model(x)
    pass
