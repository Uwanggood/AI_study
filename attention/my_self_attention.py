import math

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=64, num_heads=8, d_ff=256, dropout=0.1):
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 1. (Seq_Len, d_model) 크기의 0 행렬 생성
        pe = torch.zeros(max_len, d_model)

        # 2. 위치 인덱스 (0, 1, 2, ... 195)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 3. 주파수 계산 (10000^(2i/d_model))
        # 복잡해 보이지만, 그냥 "서로 다른 주기의 파동을 만든다"는 수식입니다.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 4. 짝수 인덱스엔 Sin, 홀수 인덱스엔 Cos 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 5. 배치 차원 추가 (1, Seq, Dim) -> 브로드캐스팅용
        pe = pe.unsqueeze(0)

        # 중요: 학습되는 파라미터가 아니므로 buffer로 등록 (저장은 하되 업데이트 X)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 입력 x에 위치 정보를 더해줌 (x + PE)
        # x.size(1)은 시퀀스 길이 (예: 196)
        return x + self.pe[:, :x.size(1)]


class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, d_model=64, num_heads=8, num_layers=6):
        super().__init__()

        # [1] 전처리: 이미지 -> 패치 벡터 (Batch, 196, 64)
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, d_model)

        # [2] 위치 정보: 벡터에 주소 더하기
        self.pos_encoder = PositionalEncoding(d_model)

        # [3] 인코더 쌓기 (6층 석탑)
        # 님께서 만든 TransformerEncoderLayer를 6번 쌓음
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff=d_model * 4)
            for _ in range(num_layers)
        ])

        # [4] 분류기 (마지막에 개인지 고양이인지 맞추는 놈)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10)  # 예: 10개 클래스 분류

    def forward(self, x):
        # 1. Embedding (이미지 -> 벡터)
        x = self.patch_embed(x)  # [B, 196, 64]

        # 2. Add Position (위치 정보 추가)
        x = self.pos_encoder(x)  # [B, 196, 64] -> 값이 살짝 변함

        # 3. Transformer Encoder Layers (반복)
        for layer in self.layers:
            x = layer(x)

        # 4. Classification
        # 보통 전체 패치의 평균(Global Average Pooling)을 쓰거나
        # 맨 첫 번째 패치(CLS 토큰)만 사용함. 여기선 평균 사용.
        x = self.norm(x)
        x = x.mean(dim=1)  # [B, 196, 64] -> [B, 64] (패치들을 하나로 압축)

        # 5. 최종 예측
        return self.head(x)  # [B, 10]


# --- 실행 ---
model = SimpleViT()
img = torch.randn(1, 3, 224, 224)
pred = model(img)
print(f"최종 출력 Shape: {pred.shape}")  # [1, 10] -> 클래스 확률값
