import math

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 1. Self-Attention 서브 레이어
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)  # 첫 번째 정규화
        self.dropout1 = nn.Dropout(dropout)

        # 2. Feed Forward 서브 레이어
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 확대 (64 -> 256)
            nn.ReLU(),  # 활성화 함수
            nn.Linear(d_ff, d_model)  # 복구 (256 -> 64)
        )
        self.norm2 = nn.LayerNorm(d_model)  # 두 번째 정규화
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- [Phase 1] Attention + Add & Norm ---
        # 1. 원본 저장 (Residual 용)
        residual = x

        # 2. Attention 수행
        x = self.self_attn(x, x, x, mask)  # Q=K=V=x (Self Attention)
        x = self.dropout1(x)

        # 3. Add & Norm
        # (원본 + 어텐션결과)를 정규화
        x = self.norm1(x + residual)

        # --- [Phase 2] FFN + Add & Norm ---
        # 4. 원본 저장 (Residual 용)
        residual = x

        # 5. FFN 수행
        x = self.ffn(x)
        x = self.dropout2(x)

        # 6. Add & Norm
        x = self.norm2(x + residual)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어 떨어져야 합니다."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 1. Q, K, V 생성을 위한 Linear
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 2. 마지막 출력을 위한 Linear
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # -------------------------------------------------------
        # 1. Linear & Split Heads
        # [Batch, Seq, Dim] -> [Batch, Seq, Head, d_head] -> [Batch, Head, Seq, d_head]
        # -------------------------------------------------------
        Q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # -------------------------------------------------------
        # 2. Scaled Dot-Product Attention (핵심 로직)
        # -------------------------------------------------------
        # scores: (Batch, Head, Seq, Seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        # out: (Batch, Head, Seq, d_head)
        out = torch.matmul(attn_weights, V)

        # -------------------------------------------------------
        # 3. Concatenate (Merge Heads)
        # [Batch, Head, Seq, d_head] -> [Batch, Seq, Head, d_head] -> [Batch, Seq, Dim]
        # -------------------------------------------------------
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # -------------------------------------------------------
        # 4. Final Linear (Mix)
        # -------------------------------------------------------
        return self.w_o(out)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_model=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 14*14 = 196개

        # 핵심: 16x16 영역을 한번에 읽어서 64차원 벡터로 만듦
        # 커널크기=16, 스트라이드=16 -> 이미지를 겹치지 않게 조각냄
        self.proj = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (Batch, 3, 224, 224)
        x = self.proj(x)
        # 결과: (Batch, 64, 14, 14) -> 64는 d_model

        # Flatten: (Batch, 64, 196)으로 만듦
        x = x.flatten(2)

        # Transpose: (Batch, 196, 64) -> 트랜스포머 입력 순서 (Batch, Seq, Dim)
        x = x.transpose(1, 2)
        return x


# 1. 가상의 이미지 생성 (배치 1개, RGB, 224x224)
raw_image = torch.randn(1, 3, 224, 224)
print(f"1. 원본 이미지: {raw_image.shape}")

# 2. 패치 임베딩 (이미지를 벡터 시퀀스로 변환)
# d_model = 64로 설정
embedder = PatchEmbedding(img_size=224, patch_size=16, d_model=64)
input_vector = embedder(raw_image)

print(f"2. 패치 임베딩 후 (Input 'x'): {input_vector.shape}")
print("   -> 해석: [배치1, 패치196개, 특징64개]")
print("   -> 이제 이 196개의 패치가 마치 '196개의 단어'처럼 취급됩니다.")

# 3. 트랜스포머 인코더 통과
# 님이 작성한 Encoder Layer 생성-
encoder_layer = TransformerEncoderLayer(d_model=64, num_heads=8, d_ff=256)

# Forward! (여기서 내부적으로 Q, K, V가 만들어집니다)
output = encoder_layer(input_vector)

print(f"3. 인코더 출력: {output.shape}")
