#!/usr/bin/env python3
# -*- coding: utf-8 -*-

USE_RANDOM_SAMPLES = True

import os
import random

import cv2
import torch

from yolox.data.datasets.yolo_format import YOLOFormatDataset
from yolox.exp.large_object_exp import LargeObjectExp
from yolox.utils import postprocess

# Exp 로드
exp = LargeObjectExp()
model = exp.get_model()

# 최신 체크포인트 로드
ckpt_path = "./YOLOX_outputs/large_object_convnext_256/epoch_200_ckpt.pth"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# EMA 모델 사용 (있으면)
if "ema_model" in ckpt:
    print("EMA 모델 사용!")
    model.load_state_dict(ckpt["ema_model"])
else:
    print("일반 모델 사용")
    model.load_state_dict(ckpt["model"])
    
# eval 모드로 설정
model.eval()
model.cuda()

# BatchNorm만 train 모드로 유지 (running stats 대신 batch stats 사용)
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.train()

print("BatchNorm을 train 모드로 설정!")

# decode_in_inference 확인
print(f"decode_in_inference: {model.head.decode_in_inference}")

# 데이터셋
dataset = YOLOFormatDataset(
    data_dir='/home/uwanggood/문서/workspace/yolo_resize_dataset_256',
    split='val',
    img_size=(256, 256),
    cache=False
)

print(f"체크포인트 로드: {ckpt_path}")
print(f"Epoch: {ckpt.get('start_epoch', 'unknown')}")
print()

# 색상 정의
color_gt = (0, 255, 0)      # 초록 - Ground Truth
color_pred = (0, 0, 255)    # 빨강 - Prediction
class_names = dataset._classes

# 출력 디렉토리 생성
output_dir = "./inference_results"
os.makedirs(output_dir, exist_ok=True)

# 5개 샘플만 테스트 (디버깅용)
num_samples = 5

# 샘플 인덱스 선택
if USE_RANDOM_SAMPLES:
    dataset_size = len(dataset)
    sample_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
    sample_indices.sort()
    print(f"랜덤으로 선택된 {num_samples}개 샘플: {sample_indices}")
else:
    sample_indices = list(range(num_samples))
    print(f"첫 {num_samples}개 샘플로 테스트: {sample_indices}")
print()

# 선택된 샘플로 테스트
for idx, i in enumerate(sample_indices):
    img, target, img_info, img_id = dataset.pull_item(i)
    
    # 원본 이미지 복사 (시각화용)
    img_vis = img.copy()
    
    print(f"\n{'='*80}")
    print(f"Sample {i}:")
    print(f"  - 원본 이미지 통계 (전처리 전):")
    print(f"    mean={img.mean():.2f}, std={img.std():.2f}, min={img.min():.2f}, max={img.max():.2f}")
    
    # 전처리
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda()
    img_tensor = img_tensor / 255.0
    
    print(f"  - 전처리 후 통계:")
    print(f"    mean={img_tensor.mean():.4f}, std={img_tensor.std():.4f}, min={img_tensor.min():.4f}, max={img_tensor.max():.4f}")
    print(f"  - Input shape: {img_tensor.shape}")
    print(f"  - Ground truth objects: {len(target)}")
    
    # 추론
    with torch.no_grad():
        outputs_raw = model(img_tensor)
    
    # Postprocessing (NMS)
    if outputs_raw is not None:
        print(f"  - Raw output shape: {outputs_raw.shape}")
        
        # Raw output 통계 (이미 sigmoid 적용됨)
        obj_conf = outputs_raw[0, :, 4]  # sigmoid 이미 적용됨
        max_obj = obj_conf.max().item()
        mean_obj = obj_conf.mean().item()
        print(f"  - Objectness stats: max={max_obj:.4f}, mean={mean_obj:.4f}")
        
        # 첫 5개 bbox 좌표 확인
        if len(outputs_raw.shape) == 3 and outputs_raw.shape[2] >= 4:
            bboxes_raw = outputs_raw[0, :5, :4]
            print(f"  - First 5 bbox coords (raw):")
            for j, bbox in enumerate(bboxes_raw):
                print(f"    [{j}] x1={bbox[0]:.2f}, y1={bbox[1]:.2f}, x2={bbox[2]:.2f}, y2={bbox[3]:.2f}, obj_conf={obj_conf[j]:.4f}")
        
        # YOLOX postprocess (NMS, confidence filtering)
        outputs = postprocess(
            outputs_raw,
            num_classes=exp.num_classes,
            conf_thre=0.2,  # 0.001 -> 0.3 (적절한 threshold)
            nms_thre=exp.nmsthre,
            class_agnostic=False,
        )
        
        if outputs[0] is not None:
            predictions = outputs[0].cpu().numpy()
            print(f"  - Predictions after NMS: {len(predictions)}")
            
            # 모든 예측 출력
            for j, pred in enumerate(predictions):
                x1, y1, x2, y2, obj_conf, cls_conf, cls_id = pred
                print(f"    [{j}] bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), "
                      f"obj_conf={obj_conf:.3f}, cls_conf={cls_conf:.3f}, cls_id={int(cls_id)}")
            
            # 예측 박스 그리기 (빨강)
            for pred in predictions:
                x1, y1, x2, y2, obj_conf, cls_conf, cls_id = pred
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(cls_id)
                
                # 바운딩 박스
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color_pred, 2)
                
                # 라벨
                label = f"{class_names[cls_id]}: {obj_conf*cls_conf:.2f}"
                cv2.putText(img_vis, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pred, 2)
        else:
            print(f"  - No predictions after NMS")
    
    # Ground truth 박스 그리기 (초록)
    print(f"  - Ground Truth:")
    if target is not None and len(target) > 0:
        for j, obj in enumerate(target):
            x1, y1, x2, y2, cls_id = obj
            print(f"    [{j}] bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), cls_id={int(cls_id)}")
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(cls_id)
            
            # 바운딩 박스
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color_gt, 2)
            
            # 라벨
            label = f"GT: {class_names[cls_id]}"
            cv2.putText(img_vis, label, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_gt, 2)
    
    # 이미지 저장
    output_path = os.path.join(output_dir, f"inference_{idx}.png")
    cv2.imwrite(output_path, img_vis)
    print(f"  - 저장: {output_path} (dataset index: {i})")

print(f"\n{'='*80}")
print(f"✅ 완료! {output_dir} 폴더를 확인하세요.")
print()
print("범례:")
print("  🟢 초록 박스 = Ground Truth")
print("  🔴 빨강 박스 = Model Prediction")

