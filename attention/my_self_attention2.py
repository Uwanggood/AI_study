import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(144, 256), patch_size=16, in_chans=3, embed_dim=64):
        super().__init__()

        if isinstance(img_size, int):
            img_h = img_w = img_size
        else:
            assert len(img_size) == 2, "img_size must be either an int or a tuple of length 2."
            img_h, img_w = img_size

        self.img_size = (img_h, img_w)
        self.patch_size = patch_size

        self.n_patches_h = img_h // patch_size
        self.n_patches_w = img_w // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)

        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.w_q(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.w_q(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        out = attn_weights @ V

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=64, num_heads=8, d_ff=256, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x

        x = self.self_attn(x, x, x, mask=mask)
        x = self.dropout1(x)

        x = self.norm1(x + residual)

        residual = x

        x = self.ffn(x)
        x = self.dropout2(x)

        x = self.norm2(x+ residual)
        return x

class SimpleViT(nn.Module):
    def __init__(self, img_size=(144, 256), patch_size=16, d_model=64, in_chans=3, num_heads=8, num_layers=6, num_classes=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff=d_model * 4)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)

        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)

        return self.head(x)

model = SimpleViT()
img = torch.randn(1, 3, 144, 256)
pred = model(img)
print(f'최종 출력 Shape: {pred.shape}')