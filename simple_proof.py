#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초간단 테스트: 모델이 정말 고장났는지 확인
"""

import torch
import cv2
import numpy as np
from yolox.exp.large_object_exp import LargeObjectExp
from yolox.data.datasets.yolo_format import YOLOFormatDataset

print("\n" + "="*80)
print("🔍 모델 고장 여부 확인 테스트")
print("="*80)

# 1. 모델 로드
exp = LargeObjectExp()
model = exp.get_model()
ckpt = torch.load("./YOLOX_outputs/large_object_convnext_256_v2/epoch_150_ckpt.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()
model.cuda()

print(f"✅ 체크포인트 로드: epoch {ckpt['start_epoch']}")


# 2. 데이터셋
dataset = YOLOFormatDataset(
    data_dir='/home/uwanggood/문서/workspace/yolo_resize_dataset_256',
    split='val',
    img_size=(256, 256),
    cache=False
)

print("\n📌 테스트 1: 서로 다른 이미지 3개를 모델에 넣어봅니다")
print("-" * 80)

predictions_list = []

for i in range(3):
    img, _, _, _ = dataset.pull_item(i)
    
    # 전처리
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
    
    # 추론
    with torch.no_grad():
        output = model(img_tensor)
    
    # objectness (물체가 있을 확률) 추출
    obj_scores = torch.sigmoid(output[0, :, 4])
    max_obj = obj_scores.max().item()
    mean_obj = obj_scores.mean().item()
    
    print(f"\n이미지 {i}:")
    print(f"  - 이미지 밝기 평균: {img.mean():.1f}/255 (서로 다른 이미지임)")
    print(f"  - 모델 출력 (max objectness): {max_obj:.4f}")
    print(f"  - 모델 출력 (mean objectness): {mean_obj:.4f}")
    
    predictions_list.append((max_obj, mean_obj))

print("\n" + "="*80)
print("📊 결과 분석")
print("="*80)

# 출력이 모두 똑같은지 확인
all_same = True
first_max, first_mean = predictions_list[0]

for i, (max_obj, mean_obj) in enumerate(predictions_list[1:], 1):
    diff = abs(max_obj - first_max)
    if diff > 0.001:  # 0.001 이상 차이나면 다른 것
        all_same = False
        break

if all_same:
    print("❌ 문제 발견!")
    print(f"   → 3개의 다른 이미지에서 모두 똑같은 출력: {first_max:.4f}")
    print(f"   → 이것은 모델이 입력을 보지 않는다는 뜻입니다.")
    print(f"\n💡 원인: ImageNet으로 학습된 Pretrained 모델의 BatchNorm이")
    print(f"         우리 데이터(트럭/그래플 이미지)와 안 맞습니다.")
    print(f"\n✅ 해결책: Pretrained 없이 처음부터 학습")
else:
    print("✅ 정상!")
    print("   → 다른 이미지에서 다른 출력이 나옵니다.")

print("\n" + "="*80)
print("🎓 이해를 위해 공부할 것")
print("="*80)
print("""
1. **BatchNorm** (가장 중요!)
   - 뭐하는 거냐? 신경망의 숫자들을 정규화(normalize)
   - 왜 문제? 학습할 때 본 데이터와 다른 데이터가 들어오면 이상하게 작동
   - 키워드: "Batch Normalization", "running_mean", "running_var"
   
2. **Pretrained Weight**
   - 뭐하는 거냐? 다른 데이터로 미리 학습된 모델
   - 왜 문제? ImageNet(일반 사진)으로 학습했는데, 우리는 트럭 사진
   - 키워드: "Transfer Learning", "Fine-tuning"
   
3. **eval() vs train() 모드**
   - eval(): 학습할 때 저장한 통계 사용 (고정)
   - train(): 현재 데이터의 통계 사용 (변함)
   - 키워드: "PyTorch eval mode", "BatchNorm inference"

📚 추천 학습 순서:
   1) PyTorch BatchNorm 튜토리얼
   2) Transfer Learning 개념
   3) Object Detection (YOLO) 기초
""")

print("\n" + "="*80)

