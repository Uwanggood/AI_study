import torch.nn as nn


class Imagenet(nn.Module):
    def __init__(self, num_classes=1000):
        super(Imagenet, self).__init__()
