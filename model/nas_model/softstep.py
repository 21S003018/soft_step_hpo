import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from model.nas_model.layers.softconv import SoftConv2d, SoftChannelConv2d, SoftKernelConv2d


# class SoftResidualBlock(nn.Module):
#     def __init__(self, inplanes, hidden_planes, kernel_size=3, stride=1, expansion=4):
#         super(SoftInvertedResidualBlock, self).__init__()
#         self.expansion = expansion
#         out_planes = self.expansion*hidden_planes
#         self.conv1 = SoftChannelConv2d(
#             inplanes, hidden_planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(hidden_planes)

#         self.conv2 = SoftConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
#                                 stride=stride, padding=int(kernel_size/2), bias=False)
#         self.bn2 = nn.BatchNorm2d(hidden_planes)

#         self.conv3 = SoftChannelConv2d(hidden_planes, out_planes,
#                                        kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_planes)

#         self.shortcut = nn.Sequential(
#             nn.Conv2d(inplanes, out_planes,
#                       kernel_size=1, stride=stride, bias=False),
#             nn.BatchNorm2d(out_planes)
#         )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         self.conv1_indicators = self.conv1.sample_indicator()
#         out = torch.mul(out, self.conv1_indicators.reshape(
#             (1, self.conv1_indicators.shape[0], 1, 1)))

#         out = F.relu(self.bn2(self.conv2(out)))
#         self.conv2_indicators = self.conv2.sample_channel_indicator()
#         out = torch.mul(out, self.conv2_indicators.reshape(
#             (1, self.conv2_indicators.shape[0], 1, 1)))

#         out = self.bn3(self.conv3(out))
#         out = out + self.shortcut(x)
#         out = F.relu(out)
#         self.conv3_indicators = self.conv3.sample_indicator()
#         out = torch.mul(out, self.conv3_indicators.reshape(
#             (1, self.conv3_indicators.shape[0], 1, 1)))
#         return out


# class SoftInvertedResidualBlock(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size=3, stride=1, expansion=4):
#         super(SoftInvertedResidualBlock, self).__init__()
#         self.expansion = expansion
#         hidden_planes = round(inplanes * self.expansion)
#         print(self.expansion)
#         self.arch_opt = False

#         self.conv1 = SoftChannelConv2d(
#             inplanes, hidden_planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(hidden_planes)

#         self.conv2 = SoftKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
#                                       stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
#         self.bn2 = nn.BatchNorm2d(hidden_planes)

#         self.conv3 = SoftChannelConv2d(hidden_planes, planes,
#                                        kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential(
#             nn.Conv2d(inplanes, planes,
#                       kernel_size=1, stride=stride, bias=False),
#             nn.BatchNorm2d(planes)
#         )

#     def forward(self, x):
#         if self.arch_opt:
#             out = F.relu6(self.bn1(self.conv1(x)))
#             out = F.relu6(self.bn2(self.conv2(out)))
#             out = torch.mul(out, self.conv1.channel_indicators)

#             out = self.bn3(self.conv3(out))
#             out = out + self.shortcut(x)
#             out = F.relu6(out)
#             out = torch.mul(out, self.conv3.channel_indicators)
#             self.arch_opt = False
#         else:
#             out = F.relu6(self.bn1(self.conv1(x)))
#             out = F.relu6(self.bn2(self.conv2(out, False)))
#             out = torch.mul(out, self.conv1.channel_indicators.data)

#             out = self.bn3(self.conv3(out))
#             out = out + self.shortcut(x)
#             out = F.relu6(out)
#             out = torch.mul(out, self.conv3.channel_indicators.data)
#         return out


# class SoftStep(nn.Module):
#     def __init__(self, input_channel, ndim, num_classes, path=None, block=SoftInvertedResidualBlock) -> None:
#         super(SoftStep, self).__init__()
#         with open(path, 'r') as f:
#             struc = json.load(f)
#         self.block_type = struc["block_type"]
#         block = SoftInvertedResidualBlock if self.block_type == "linear" else SoftResidualBlock
#         output_channel = struc["b0"]["conv_in"]
#         self.conv_in = nn.Sequential(
#             nn.Conv2d(input_channel, output_channel,
#                       kernel_size=3, stride=struc["b0"]["stride_in"], padding=1, bias=False),
#             nn.BatchNorm2d(output_channel),
#             nn.ReLU6(inplace=True)
#         )

#         input_channel = output_channel
#         layers = []
#         for block_config in struc["blocks"]:
#             e, c, n, k, s = block_config['e'], block_config['c'], block_config['n'], block_config['k'], block_config['s']
#             output_channel = c
#             for i in range(n):
#                 layers.append(
#                     block(input_channel, output_channel, k, s if i == 0 else 1, e))
#                 input_channel = output_channel
#         self.blocks = nn.Sequential(*layers)

#         input_channel = output_channel
#         output_channel = struc["b0"]["conv_out"]
#         self.conv_out = nn.Sequential(
#             nn.Conv2d(input_channel, output_channel, kernel_size=1,
#                       stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(output_channel),
#             nn.ReLU6(inplace=True)
#         )

#         self.fc = nn.Linear(output_channel, num_classes)
#         return

#     def forward(self, x, arch_opt=False):
#         if arch_opt:
#             self.update_indicators()
#         x = self.conv_in(x)
#         x = self.blocks(x)
#         x = self.conv_out(x)
#         x = F.adaptive_avg_pool2d(x, 1)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

#     def model_parameters(self):
#         for name, param in self.named_parameters():
#             if name.__contains__("weight") or name.__contains__("bias"):
#                 yield param

#     def arch_parameters(self):
#         for name, param in self.named_parameters():
#             if name.__contains__("alpha"):
#                 yield param

#     def search_result_list(self):
#         for name, _ in self.named_parameters():
#             if name.__contains__("channel_alpha"):
#                 segments = name.split(".")
#                 layer = eval("self.blocks[int(segments[1])]."+segments[2])
#                 yield min(layer.channel_alpha.item(), 1)*layer.out_channels
#             if name.__contains__("kernel_alpha"):
#                 segments = name.split(".")
#                 layer = eval("self.blocks[int(segments[1])]."+segments[2])
#                 yield min(layer.kernel_alpha.item(), 1)*int(layer.kernel_size/2)

#     def update_indicators(self):
#         for conv_block in self.blocks:
#             conv_block.arch_opt = True
#             conv_block.conv1.update_channel_indicators()
#             conv_block.conv2.update_kernel_mask()
#             conv_block.conv3.update_channel_indicators()
#             if self.block_type == "bottleneck":
#                 conv_block.conv2.update_channel_indicators()
#         return

class LinearBlock(nn.Module):  # normal block or reduction block
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expansion=4):
        super(LinearBlock, self).__init__()

        hidden_planes = round(in_planes * expansion)

        self.conv1 = SoftChannelConv2d(
            in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = SoftKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                      stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = SoftChannelConv2d(hidden_planes, out_planes,
                                       kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x, arch_opt):
        if arch_opt:
            out = F.relu6(self.bn1(self.conv1(x)))
            out = F.relu6(self.bn2(self.conv2(out, arch_opt)))
            out = torch.mul(out, self.conv1.channel_indicators)

            out = F.relu6(self.bn3(self.conv3(out)))
            out = torch.mul(out, self.conv3.channel_indicators)
        else:
            out = F.relu6(self.bn1(self.conv1(x)))
            out = F.relu6(self.bn2(self.conv2(out, arch_opt)))
            out = torch.mul(out, self.conv1.channel_indicators.data)

            out = F.relu6(self.bn3(self.conv3(out)))
            out = torch.mul(out, self.conv3.channel_indicators.data)
        return out

    def update_indicators(self):
        self.conv1.update_channel_indicators()
        self.conv2.update_kernel_mask()
        self.conv3.update_channel_indicators()
        return

    def protect_controller(self):
        self.conv1.protect_controller()
        self.conv2.protect_controller()
        self.conv3.protect_controller()
        return


class SkipLinearBlock(nn.Module):  # skip block
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expansion=4):
        super(SkipLinearBlock, self).__init__()
        assert in_planes == out_planes
        assert stride == 1

        hidden_planes = round(in_planes * expansion)

        self.conv1 = SoftChannelConv2d(
            in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = SoftKernelConv2d(hidden_planes, hidden_planes, kernel_size=kernel_size,
                                      stride=stride, padding=int(kernel_size/2), bias=False, groups=hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = nn.Conv2d(hidden_planes, out_planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x, arch_opt):
        if arch_opt:
            out = F.relu6(self.bn1(self.conv1(x)))
            out = F.relu6(self.bn2(self.conv2(out, arch_opt)))
            out = torch.mul(out, self.conv1.channel_indicators)

            out = self.bn3(self.conv3(out))
        else:
            out = F.relu6(self.bn1(self.conv1(x)))
            out = F.relu6(self.bn2(self.conv2(out, arch_opt)))
            out = torch.mul(out, self.conv1.channel_indicators.data)

            out = self.bn3(self.conv3(out))
        return out

    def update_indicators(self):
        self.conv1.update_channel_indicators()
        self.conv2.update_kernel_mask()
        return

    def protect_controller(self):
        self.conv1.protect_controller()
        self.conv2.protect_controller()
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

    def forward(self, x, arch_opt=False):
        x = self.block(x, arch_opt)
        if arch_opt:
            for skip in self.skips:
                x = F.relu6(torch.mul(skip(x, arch_opt),
                            self.block.conv3.channel_indicators) + x)
        else:
            for skip in self.skips:
                x = F.relu6(torch.mul(skip(x, arch_opt),
                            self.block.conv3.channel_indicators.data) + x)
        return x

    def update_indicators(self):
        self.block.update_indicators()
        for skip in self.skips:
            skip.update_indicators()
        return

    def protect_controller(self):
        self.block.protect_controller()
        for skip in self.skips:
            skip.protect_controller()
        return


class SoftStep(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, path=None) -> None:
        super(SoftStep, self).__init__()
        with open(path, 'r') as f:
            config = json.load(f)
        self.block_type = config["type"]
        # pre conv
        input_channel = input_channel
        output_channel = config["layer"]["conv_in"]
        self.conv_in, self.bn_in = SoftChannelConv2d(
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
        self.conv_out, self.bn_out = SoftChannelConv2d(
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
        x = self.block_in(x, arch_opt)
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
        config["block"] = [{
            "type": "normal",
            "c1": self.block_in.conv1.out_channels if full else int(self.block_in.conv1.channel_alpha*self.block_in.conv1.out_channels),
            "k": self.block_in.conv2.kernel_size if full else int(self.block_in.conv2.kernel_alpha*int(self.block_in.conv2.kernel_size/2))*2+1,
            "c2": self.block_in.conv3.out_channels if full else int(self.block_in.conv3.channel_alpha*self.block_in.conv3.out_channels),
            "s": 1
        }]
        # stage
        for stage in self.stages:
            config["block"].append({
                "type": "normal" if stage.stride == 1 else "reduction",
                "c1": stage.block.conv1.out_channels if full else int(stage.block.conv1.channel_alpha*stage.block.conv1.out_channels),
                "k": stage.block.conv2.kernel_size if full else int(stage.block.conv2.kernel_alpha*int(stage.block.conv2.kernel_size/2))*2+1,
                "c2": stage.block.conv3.out_channels if full else int(stage.block.conv3.channel_alpha*stage.block.conv3.out_channels),
                "s": stage.stride
            })
            for skip in stage.skips:
                config["block"].append({
                    "type": "skip",
                    "c1": skip.conv1.out_channels if full else int(skip.conv1.channel_alpha*skip.conv1.out_channels),
                    "k": skip.conv2.kernel_size if full else int(skip.conv2.kernel_alpha*int(skip.conv2.kernel_size/2))*2+1,
                    "c2": stage.block.conv3.out_channels if full else int(stage.block.conv3.channel_alpha*stage.block.conv3.out_channels),
                    "s": 1
                })
        return config

    def update_indicators(self):
        # layer
        self.conv_in.update_channel_indicators()
        self.conv_out.update_channel_indicators()
        # block
        self.block_in.update_indicators()
        # stage
        for stage in self.stages:
            stage.update_indicators()
        return

    def protect_controller(self):
        self.conv_in.protect_controller()
        self.block_in.protect_controller()
        for stage in self.stages:
            stage.protect_controller()
        self.conv_out.protect_controller()
        return


if __name__ == '__main__':
    # train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
    # print(input_channel, inputdim, nclass)
    # # model = IRFBlock(in_channels=input_channel, out_channels=3)
    # model = SoftStep(input_channel, inputdim, nclass)
    # x = torch.rand(3, 32, 32)
    # preds = model(x)
    pass
