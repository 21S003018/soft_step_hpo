import torch.nn as nn


class SoftStep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
