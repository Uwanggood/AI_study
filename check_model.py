#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from yolox.exp.large_object_exp import LargeObjectExp

# Exp 로드
exp = LargeObjectExp()
model = exp.get_model()

# 체크포인트 로드
ckpt_path = "./YOLOX_outputs/large_object_convnext_256_v2/epoch_40_ckpt.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()
model.cuda()

print("="*80)
print("모델 구조 확인")
print("="*80)

# 두 개의 다른 입력 생성
input1 = torch.randn(1, 3, 256, 256).cuda()
input2 = torch.zeros(1, 3, 256, 256).cuda()  # 완전히 다른 입력

print("\n입력 1 통계:")
print(f"  mean={input1.mean():.4f}, std={input1.std():.4f}, min={input1.min():.4f}, max={input1.max():.4f}")

print("\n입력 2 통계:")
print(f"  mean={input2.mean():.4f}, std={input2.std():.4f}, min={input2.min():.4f}, max={input2.max():.4f}")

# Backbone 출력 확인
print("\n" + "="*80)
print("Backbone 출력 확인")
print("="*80)

with torch.no_grad():
    backbone_out1 = model.backbone(input1)
    backbone_out2 = model.backbone(input2)

print(f"\nBackbone output 1 shape: {backbone_out1[0].shape if isinstance(backbone_out1, tuple) else backbone_out1.shape}")
print(f"Backbone output 2 shape: {backbone_out2[0].shape if isinstance(backbone_out2, tuple) else backbone_out2.shape}")

if isinstance(backbone_out1, tuple):
    feat1 = backbone_out1[0]
    feat2 = backbone_out2[0]
else:
    feat1 = backbone_out1
    feat2 = backbone_out2

print(f"\nBackbone feature 1 통계:")
print(f"  mean={feat1.mean():.4f}, std={feat1.std():.4f}, min={feat1.min():.4f}, max={feat1.max():.4f}")

print(f"\nBackbone feature 2 통계:")
print(f"  mean={feat2.mean():.4f}, std={feat2.std():.4f}, min={feat2.min():.4f}, max={feat2.max():.4f}")

# 두 출력이 다른지 확인
diff = torch.abs(feat1 - feat2).mean().item()
print(f"\n두 backbone 출력의 차이: {diff:.6f}")

if diff < 1e-6:
    print("⚠️  경고: Backbone이 입력과 무관하게 똑같은 출력을 내고 있습니다!")
else:
    print("✅ Backbone은 정상적으로 다른 출력을 냅니다.")

# Head 출력 확인
print("\n" + "="*80)
print("Head 출력 확인")
print("="*80)

with torch.no_grad():
    if isinstance(backbone_out1, tuple):
        head_out1 = model.head(backbone_out1)
        head_out2 = model.head(backbone_out2)
    else:
        head_out1 = model.head([backbone_out1])
        head_out2 = model.head([backbone_out2])

print(f"\nHead output 1 shape: {head_out1.shape}")
print(f"Head output 2 shape: {head_out2.shape}")

print(f"\nHead output 1 통계:")
print(f"  mean={head_out1.mean():.4f}, std={head_out1.std():.4f}, min={head_out1.min():.4f}, max={head_out1.max():.4f}")

print(f"\nHead output 2 통계:")
print(f"  mean={head_out2.mean():.4f}, std={head_out2.std():.4f}, min={head_out2.min():.4f}, max={head_out2.max():.4f}")

# Objectness 확인
obj1 = torch.sigmoid(head_out1[0, :, 4])
obj2 = torch.sigmoid(head_out2[0, :, 4])

print(f"\nObjectness 1: max={obj1.max():.4f}, mean={obj1.mean():.4f}")
print(f"Objectness 2: max={obj2.max():.4f}, mean={obj2.mean():.4f}")

diff_head = torch.abs(head_out1 - head_out2).mean().item()
print(f"\n두 head 출력의 차이: {diff_head:.6f}")

if diff_head < 1e-6:
    print("⚠️  경고: Head가 입력과 무관하게 똑같은 출력을 내고 있습니다!")
else:
    print("✅ Head는 정상적으로 다른 출력을 냅니다.")

# 전체 모델 출력 확인
print("\n" + "="*80)
print("전체 모델 출력 확인")
print("="*80)

with torch.no_grad():
    full_out1 = model(input1)
    full_out2 = model(input2)

diff_full = torch.abs(full_out1 - full_out2).mean().item()
print(f"\n두 전체 모델 출력의 차이: {diff_full:.6f}")

if diff_full < 1e-6:
    print("⚠️  경고: 모델이 입력과 무관하게 똑같은 출력을 내고 있습니다!")
    print("\n가능한 원인:")
    print("  1. BatchNorm/LayerNorm의 running stats가 고정됨")
    print("  2. 모델이 제대로 학습되지 않음")
    print("  3. Pretrained weights 로딩 문제")
    print("  4. GRN 레이어 문제")
else:
    print("✅ 모델은 정상적으로 다른 출력을 냅니다.")

print("\n" + "="*80)

