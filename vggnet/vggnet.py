import torch.nn as nn
import torch


class VGG(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),  # 64 x 222 x 222
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 x 111 x 111
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3
            ),  # 128 x 109 x 109
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 x 54 x 54
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3
            ),  # 256 x 52 x 52
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3
            ),  # 256 x 50 x 50
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256 x 25 x 25
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3
            ),  # 512 x 23 x 23
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3
            ),  # 512 x 21 x 21
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512 x 10 x 10
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),  # 512 x 8 x 8
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),  # 512 x 6 x 6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512 x 3 x 3
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
