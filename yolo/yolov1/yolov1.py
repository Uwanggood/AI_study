import torch.nn as nn

from yolo.yolo_layer_util import Flatten
from yolo.yolov1.darknet import Darknet


class YOLOv1(nn.Module):
    def __init__(self, feature_size: int, num_bboxes: int = 2, num_classes: int = 20):
        super().__init__()
        self.feature_size = feature_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.features = Darknet()
        self.fc_layers = nn.Sequential(
            Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(4096, self.feature_size * self.feature_size * (5 * self.num_bboxes + self.num_classes)),
            nn.Sigmoid()
        )

    def forward(self, x):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        x = self.features(x)
        x = self.fc_layers(x)

        x = x.view(-1, S, S, 5 * B + C)
        return x
