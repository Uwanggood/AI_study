#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from yolox.exp.large_object_exp import LargeObjectExp
from yolox.data.datasets.yolo_format import YOLOFormatDataset

# Exp 로드
exp = LargeObjectExp()
model = exp.get_model()

# 체크포인트 로드
ckpt_path = "./YOLOX_outputs/large_object_convnext_256_v2/epoch_40_ckpt.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.cuda()

# 데이터셋에서 실제 이미지 로드
dataset = YOLOFormatDataset(
    data_dir='/home/uwanggood/문서/workspace/yolo_resize_dataset_256',
    split='val',
    img_size=(256, 256),
    cache=False
)

# 두 개의 다른 이미지
img1, _, _, _ = dataset.pull_item(0)
img2, _, _, _ = dataset.pull_item(1)

input1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
input2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0

print("="*80)
print("모델 모드에 따른 출력 비교")
print("="*80)

print("\n입력 1 통계:")
print(f"  mean={input1.mean():.4f}, std={input1.std():.4f}")
print("\n입력 2 통계:")
print(f"  mean={input2.mean():.4f}, std={input2.std():.4f}")

# EVAL 모드 테스트
print("\n" + "="*80)
print("1. EVAL 모드 (BatchNorm running stats 사용)")
print("="*80)

model.eval()
with torch.no_grad():
    out1_eval = model(input1)
    out2_eval = model(input2)

obj1_eval = torch.sigmoid(out1_eval[0, :, 4])
obj2_eval = torch.sigmoid(out2_eval[0, :, 4])

print(f"\nEVAL 모드 - 출력 1: max_obj={obj1_eval.max():.4f}, mean_obj={obj1_eval.mean():.4f}")
print(f"EVAL 모드 - 출력 2: max_obj={obj2_eval.max():.4f}, mean_obj={obj2_eval.mean():.4f}")

diff_eval = torch.abs(out1_eval - out2_eval).mean().item()
print(f"\n두 출력의 차이 (EVAL): {diff_eval:.6f}")

# TRAIN 모드 테스트
print("\n" + "="*80)
print("2. TRAIN 모드 (BatchNorm current batch stats 사용)")
print("="*80)

model.train()
# Train 모드에서는 targets를 None으로 전달하면 안 되므로, inference 모드로 설정
for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.track_running_stats = False  # 현재 배치 stats만 사용

with torch.no_grad():
    # 배치로 합쳐서 테스트
    inputs_batch = torch.cat([input1, input2], dim=0)
    outs_batch = model.backbone(inputs_batch)
    
    if isinstance(outs_batch, tuple):
        outs_batch = outs_batch[0]
    
    head_outs = model.head([outs_batch])
    
    out1_train = head_outs[0:1]
    out2_train = head_outs[1:2]

obj1_train = torch.sigmoid(out1_train[0, :, 4])
obj2_train = torch.sigmoid(out2_train[0, :, 4])

print(f"\nTRAIN 모드 - 출력 1: max_obj={obj1_train.max():.4f}, mean_obj={obj1_train.mean():.4f}")
print(f"TRAIN 모드 - 출력 2: max_obj={obj2_train.max():.4f}, mean_obj={obj2_train.mean():.4f}")

diff_train = torch.abs(out1_train - out2_train).mean().item()
print(f"\n두 출력의 차이 (TRAIN): {diff_train:.6f}")

# 결론
print("\n" + "="*80)
print("결론")
print("="*80)

if diff_eval < 1e-4 and diff_train > 1e-4:
    print("⚠️  EVAL 모드에서만 문제 발생!")
    print("   → BatchNorm의 running stats가 잘못되었습니다.")
    print("   → 해결책: running stats를 현재 데이터로 업데이트하거나")
    print("              BatchNorm을 GroupNorm으로 교체")
elif diff_eval < 1e-4 and diff_train < 1e-4:
    print("⚠️  두 모드 모두 문제!")
    print("   → 모델 자체가 입력을 무시하고 있습니다.")
    print("   → 해결책: 처음부터 학습 재시작")
else:
    print("✅ 정상!")

print("\n" + "="*80)

