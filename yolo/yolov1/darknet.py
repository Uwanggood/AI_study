from dataclasses import dataclass

import torch.nn as nn
from torch import is_anomaly_check_nan_enabled

from yolo.yolo_layer_util import Squeeze


@dataclass
class ConvData:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 1


@dataclass
class PoolData:
    stride: int
    padding: int = 0
    kernel_size: int = 1


def get_conv_lyr(data: ConvData | PoolData) -> nn.Module:
    if isinstance(data, ConvData):
        return nn.Sequential(
            nn.Conv2d(in_channels=data.in_channels,
                      out_channels=data.out_channels,
                      kernel_size=data.kernel_size,
                      stride=data.stride,
                      padding=data.padding),
            nn.BatchNorm2d(num_features=data.out_channels),
            nn.LeakyReLU(negative_slope=1e-1, inplace=True),
        )
    else:
        return nn.MaxPool2d(kernel_size=data.kernel_size, stride=data.stride, padding=data.padding)


def get_features(data_list: list[ConvData]):
    return nn.Sequential(*[get_conv_lyr(data) for data in data_list])


def get_repeated_conv(list_data: list[ConvData], repeated_cnt: int) -> nn.Module:
    return nn.Sequential(*[get_conv_lyr(data) for data in list_data * repeated_cnt])


class Darknet(nn.Module):
    def __init__(self):
        super().__init__()

        # input size : 448x448
        self.features = get_features(
            [
                # Lyr 1
                ConvData(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),  # 64 x 224 x 224
                PoolData(stride=2, kernel_size=2),  # 64 x 112 x 112

                # Lyr 2
                ConvData(in_channels=64, out_channels=192, kernel_size=3, padding=1),  # 192 x 112 x 112
                PoolData(stride=2, kernel_size=2),  # 192 x 56 x 56

                # Lyr 3
                ConvData(in_channels=192, out_channels=128, kernel_size=1),  # 128 x 56 x 56
                ConvData(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 256 x 56 x 56
                ConvData(in_channels=256, out_channels=256, kernel_size=1),  # 256 x 56 x 56
                ConvData(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 512 x 56 x 56
                PoolData(stride=2, kernel_size=2),  # 28 x 28

                # Lyr 4
                # repeated 1
                ConvData(in_channels=512, out_channels=256, kernel_size=1),  # 256 x 28 x 28
                ConvData(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 512 x 28 x 28

                # repetead 2
                ConvData(in_channels=512, out_channels=256, kernel_size=1),  # 256 x 28 x 28
                ConvData(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 512 x 28 x 28

                # repetead 3
                ConvData(in_channels=512, out_channels=256, kernel_size=1),  # 256 x 28 x 28
                ConvData(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 512 x 28 x 28

                ConvData(in_channels=512, out_channels=1024, kernel_size=3, padding=1),  # 1024 x 28 x 28
                PoolData(kernel_size=2, stride=2),  # 1024 x 14 x 14

                # Lyr 5
                # repeated 1
                ConvData(in_channels=1024, out_channels=512, kernel_size=1),  # 512 x 14 x 14
                ConvData(in_channels=512, out_channels=1024, kernel_size=3, padding=1),  # 1024 x 14 x 14

                # repetead 2
                ConvData(in_channels=1024, out_channels=512, kernel_size=1),  # 512 x 14 x 14
                ConvData(in_channels=512, out_channels=1024, kernel_size=3, padding=1),  # 1024 x 14 x 14

                ConvData(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),  # 1024 x 14 x 14
                PoolData(kernel_size=3, stride=2, padding=1),  # 1024 x 7 x 7

                # Lyr 6
                ConvData(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),  # 1024 x 7 x 7
                ConvData(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),  # 1024 x 7 x 7
            ]
        )

    def forward(self, x):
        return self.features(x)
