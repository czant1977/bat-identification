import torch
from torch import nn
from torch.nn import functional as F
from utils import (
    get_same_padding_conv2d,
    MemoryEfficientSwish
)

# SE模块
class SE(nn.Module):
    def __init__(self, oup, num_squeezed_channels, input_size):
        super(SE, self).__init__()
        Conv2d = get_same_padding_conv2d(image_size=input_size)

        self._swish = MemoryEfficientSwish()

        self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

    def forward(self, inputs):
        x_squeezed = F.adaptive_avg_pool2d(inputs, 1)
        x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
        x = torch.sigmoid(x_squeezed) * inputs

        return x
