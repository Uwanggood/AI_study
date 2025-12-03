#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead, YOLOXLargeObjectHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, large_object_mode=False):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        
        self.backbone = backbone
        self.large_object_mode = large_object_mode
        
        if large_object_mode:
            if hasattr(backbone, 'single_scale') and not backbone.single_scale:
                import warnings
                warnings.warn("large_object_mode=True but backbone.single_scale=False. Setting backbone to single_scale mode.")
                if hasattr(backbone, 'single_scale'):
                    backbone.single_scale = True
                    if hasattr(backbone, 'backbone'):
                        backbone.backbone.out_indices = [3]
                    if hasattr(backbone, 'out_channels'):
                        dims = [96, 192, 384, 768]
                        if hasattr(backbone, 'backbone') and hasattr(backbone.backbone, 'dims'):
                            dims = backbone.backbone.dims
                        backbone.out_channels = [dims[3]]
        
        if head is None:
            if large_object_mode:
                in_channels = 768
                if hasattr(backbone, 'out_channels') and len(backbone.out_channels) == 1:
                    in_channels = backbone.out_channels[0]
                head = YOLOXLargeObjectHead(80, in_channels=in_channels)
            else:
                in_channels = [256, 512, 1024]
                if hasattr(backbone, 'out_channels'):
                    in_channels = backbone.out_channels
                head = YOLOXHead(80, in_channels=in_channels)
        else:
            if large_object_mode and not isinstance(head, YOLOXLargeObjectHead):
                in_channels = 768
                if hasattr(backbone, 'out_channels') and len(backbone.out_channels) == 1:
                    in_channels = backbone.out_channels[0]
                head = YOLOXLargeObjectHead(
                    head.num_classes if hasattr(head, 'num_classes') else 80,
                    width=head.width if hasattr(head, 'width') else 1.0,
                    in_channels=in_channels,
                    act=head.act if hasattr(head, 'act') else "silu",
                    depthwise=head.depthwise if hasattr(head, 'depthwise') else False,
                )
            elif not large_object_mode and hasattr(backbone, 'out_channels') and hasattr(head, 'in_channels'):
                if head.in_channels != backbone.out_channels:
                    in_channels = backbone.out_channels
                    head = YOLOXHead(
                        head.num_classes,
                        width=head.width if hasattr(head, 'width') else 1.0,
                        strides=head.strides if hasattr(head, 'strides') else [8, 16, 32],
                        in_channels=in_channels,
                        act=head.act if hasattr(head, 'act') else "silu",
                        depthwise=head.depthwise if hasattr(head, 'depthwise') else False,
                    )

        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
