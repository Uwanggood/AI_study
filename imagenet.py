import torch.nn as nn


__all__ = ["AlexNet", "AlexNet_Weights", "alexnet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel=11, stride=4), # 55 x 55 x 96
            nn.ReLU(inplace=True),
            nn.Maxpool2d(kernel_size=3, stride=2),  # 55 x 55
            nn.Conv2d(in_channels=96, out_channels=256, kernel=5, )
        )