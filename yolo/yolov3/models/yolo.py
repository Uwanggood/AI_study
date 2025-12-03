from pathlib import Path

import torch
import torch.nn as nn



class Detect(nn.Module):
    stride = None
    dynamic = None
    export = None

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc # number of classes
        self.no = nc + 5 # number of outputs per anchor
        self.nl = len(anchors) # number of detection layers
        self.na = len(anchors[0]) // 2 # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]
