#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import cv2
from yolox.data.datasets.yolo_format import YOLOFormatDataset

# 데이터셋
dataset = YOLOFormatDataset(
    data_dir='/home/uwanggood/문서/workspace/yolo_resize_dataset_256',
    split='val',
    img_size=(256, 256),
    cache=False
)

print("="*80)
print("Dataset 확인 - 정말 다른 이미지를 로드하는가?")
print("="*80)

for i in range(3):
    img, target, img_info, img_id = dataset.pull_item(i)
    
    print(f"\nSample {i}:")
    print(f"  - img_id: {img_id}")
    print(f"  - img shape: {img.shape}")
    print(f"  - img stats: mean={img.mean():.2f}, std={img.std():.2f}")
    print(f"  - 첫 10 픽셀: {img[0, 0, :10]}")
    print(f"  - img 해시: {hash(img.tobytes())}")
    
    # 이미지 저장해서 눈으로 확인
    cv2.imwrite(f"./dataset_check_{i}.png", img)
    print(f"  - 저장: ./dataset_check_{i}.png")

print("\n" + "="*80)
print("✅ dataset_check_*.png 파일들을 열어서 정말 다른 이미지인지 확인하세요!")

