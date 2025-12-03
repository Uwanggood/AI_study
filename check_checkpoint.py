#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

ckpt_path = "./YOLOX_outputs/large_object_convnext_256_v2/epoch_40_ckpt.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("="*80)
print("체크포인트 분석")
print("="*80)

print(f"\n체크포인트 epoch: {ckpt.get('start_epoch', 'unknown')}")
print(f"\n체크포인트 keys: {list(ckpt.keys())}")

# 모델 weights 확인
model_state = ckpt['model']
print(f"\n모델 파라미터 개수: {len(model_state)}")

# Backbone weights 확인
backbone_keys = [k for k in model_state.keys() if 'backbone' in k]
print(f"\nBackbone 파라미터 개수: {len(backbone_keys)}")
print(f"Backbone 파라미터 예시 (첫 10개):")
for k in backbone_keys[:10]:
    param = model_state[k]
    print(f"  {k}: shape={param.shape}, mean={param.mean():.4f}, std={param.std():.4f}")

# Head weights 확인
head_keys = [k for k in model_state.keys() if 'head' in k]
print(f"\nHead 파라미터 개수: {len(head_keys)}")
print(f"Head 파라미터 예시 (첫 10개):")
for k in head_keys[:10]:
    param = model_state[k]
    print(f"  {k}: shape={param.shape}, mean={param.mean():.4f}, std={param.std():.4f}")

# GRN 파라미터 확인 (V2 특징)
grn_keys = [k for k in model_state.keys() if 'grn' in k]
print(f"\nGRN 파라미터 개수: {len(grn_keys)}")
if len(grn_keys) > 0:
    print(f"GRN 파라미터 예시:")
    for k in grn_keys[:5]:
        param = model_state[k]
        print(f"  {k}: shape={param.shape}, mean={param.mean():.4f}, std={param.std():.4f}")
else:
    print("  ⚠️ GRN 파라미터가 없습니다! V2로 학습되지 않았을 수 있습니다.")

# LayerNorm 파라미터 확인
norm_keys = [k for k in model_state.keys() if 'norm' in k and 'backbone' in k]
print(f"\nBackbone LayerNorm 파라미터 개수: {len(norm_keys)}")
print(f"LayerNorm 파라미터 예시:")
for k in norm_keys[:5]:
    param = model_state[k]
    print(f"  {k}: shape={param.shape}, mean={param.mean():.4f}, std={param.std():.4f}")

print("\n" + "="*80)

