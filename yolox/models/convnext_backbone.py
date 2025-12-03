from functools import partial
from typing import Callable, Literal

import torch.nn as nn 
import torch
import torch.nn.functional as F

from mynet.util import DropPath, trunc_normal_
from loguru import logger


class GRN(nn.Module):
    """Global Response Normalization (GRN) layer
    
    From ConvNeXt V2 paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
    https://arxiv.org/abs/2301.00808
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps
    
    def forward(self, x):
        # x: (N, H, W, C)
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    r""" ConvNeXt V2 Block
    
    Changes from V1:
    - Added GRN (Global Response Normalization) after the first linear layer
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)  # ← V2: GRN added here
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)  # ← V2: GRN applied
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_x + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt V2
        A PyTorch impl of : `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders`
          https://arxiv.org/abs/2301.00808

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        out_indices (list): Output indices for FPN. Default: [0, 1, 2, 3]
    """

    def __init__(self, in_chans=3, depths=None, dims=None,
                 drop_path_rate=0., out_indices=None,
                 ):
        super().__init__()

        if out_indices is None:
            out_indices = [0, 1, 2, 3]
        if dims is None:
            dims = [96, 192, 384, 768]
        if depths is None:
            depths = [3, 3, 9, 3]

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            checkpoint = torch.load(pretrained, map_location='cpu')
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            model_dict = self.state_dict()
            load_dict = {}
            for k, v in state_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        load_dict[k] = v
                    else:
                        logger.warning(f"Shape mismatch for {k}: checkpoint {v.shape} vs model {model_dict[k].shape}")
                else:
                    logger.warning(f"{k} not found in model")
            
            self.load_state_dict(load_dict, strict=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt V2 백본을 YOLOX에 맞게 래핑하는 클래스.
    ConvNeXt의 4개 feature 중 3개를 선택하여 YOLOX가 기대하는 형식으로 반환합니다.
    single_scale=True일 경우 stride 32 feature만 반환합니다.
    """

    def __init__(
        self,
        in_chans=3,
        depths=None,
        dims=None,
        drop_path_rate=0.0,
        pretrained=None,
        single_scale=False,
    ):
        super().__init__()
        self.single_scale = single_scale
        
        if single_scale:
            out_indices = [3]
        else:
            out_indices = [1, 2, 3]
        
        self.backbone = ConvNeXt(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
        )
        
        if pretrained is not None:
            self.backbone.init_weights(pretrained=pretrained)
        
        if dims is None:
            dims = [96, 192, 384, 768]
        
        if single_scale:
            self.out_channels = [dims[3]]
        else:
            self.out_channels = dims[1:4]

    def forward(self, x):
        """
        Args:
            x: 입력 이미지 텐서 (B, C, H, W)
        
        Returns:
            tuple 또는 tensor: single_scale=True일 경우 stride 32 feature만 반환,
                              False일 경우 3개의 feature map (stride 8, 16, 32에 해당)
        """
        features = self.backbone(x)
        if self.single_scale:
            if isinstance(features, (list, tuple)):
                return features[0]
            return features
        return features


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6,
                 data_format: Literal["channels_last", "channels_first"] = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        raise NotImplementedError
