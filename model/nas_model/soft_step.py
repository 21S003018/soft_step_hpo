import torch.nn as nn
from const import CIFAR10, MNIST
from model.blocks.basic_block import ResidualBlock, InvertedResidualBlock


class SoftStep(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, block=ResidualBlock) -> None:
        super(SoftStep, self).__init__()
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            # [6,  32, 3, 2],
            # [6,  64, 4, 2],
            # [6,  96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]
        block_input_channel = 32
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channel, out_channels=block_input_channel,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(block_input_channel),
            nn.ReLU6(inplace=True)
        )
        layers = []
        input_channel = block_input_channel
        for t, c, n, s in self.cfgs:
            output_channel = c
            for i in range(n):
                layers.append(
                    block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.blocks = nn.Sequential(*layers)

        self.conv_out = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU6(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_channel, num_classes)
        return

    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def test_model(self):

        return

    def reset_parameters(self):
        return


if __name__ == '__main__':
    # train_loader, test_loader, input_channel, inputdim, nclass = Data().get(CIFAR10)
    # print(input_channel, inputdim, nclass)
    # # model = IRFBlock(in_channels=input_channel, out_channels=3)
    # model = SoftStep(input_channel, inputdim, nclass)
    # x = torch.rand(3, 32, 32)
    # preds = model(x)
    pass
