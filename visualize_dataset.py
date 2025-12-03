#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
데이터셋 시각화 스크립트
- Ground truth 바운딩 박스 확인
- 라벨링이 제대로 되었는지 검증
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from yolox.data.datasets.yolo_format import YOLOFormatDataset


def visualize_sample(dataset, num_samples=9, save_dir="./visualization"):
    """
    데이터셋에서 샘플을 가져와 바운딩 박스를 그려서 저장합니다.
    
    Args:
        dataset: YOLOFormatDataset 인스턴스
        num_samples: 시각화할 샘플 수
        save_dir: 저장할 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 색상 정의 (클래스별)
    colors = [
        (0, 255, 0),    # 녹색 - grapple
        (255, 0, 0),    # 빨강 - truck
        (0, 0, 255),    # 파랑
        (255, 255, 0),  # 노랑
    ]
    
    # 클래스 이름
    class_names = dataset._classes
    
    print(f"데이터셋 정보:")
    print(f"- 총 이미지 수: {len(dataset)}")
    print(f"- 클래스: {class_names}")
    print(f"- 이미지 크기: {dataset.img_size}")
    print()
    
    # Figure 생성 (3x3 그리드)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # 랜덤 샘플 선택
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        if i >= num_samples:
            break
            
        # 이미지와 라벨 가져오기
        img, target, img_info, img_id = dataset.pull_item(idx)
        
        # 이미지가 normalized 되어있다면 복원
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # BGR to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width = img.shape[:2]
        
        # 바운딩 박스 그리기
        img_with_boxes = img.copy()
        
        if target is not None and len(target) > 0:
            for obj in target:
                # YOLOX format: [x1, y1, x2, y2, class_id]
                x1, y1, x2, y2, cls_id = obj
                
                # 정수형으로 변환
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 색상 선택
                color = colors[int(cls_id) % len(colors)]
                
                # 바운딩 박스 그리기
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                # 클래스 이름 표시
                cls_idx = int(cls_id)
                if cls_idx < len(class_names):
                    label = f"{class_names[cls_idx]}"
                else:
                    label = f"Class {cls_idx}"
                cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # subplot에 표시
        axes[i].imshow(img_with_boxes)
        axes[i].set_title(f"Sample {idx}: {len(target) if target is not None else 0} objects")
        axes[i].axis('off')
    
    # 남은 subplot 숨기기
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 저장
    output_path = os.path.join(save_dir, "dataset_samples.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 시각화 결과 저장: {output_path}")
    
    # 개별 이미지도 저장
    print("\n개별 이미지 저장 중...")
    for i, idx in enumerate(indices[:3]):  # 처음 3개만
        img, target, img_info, img_id = dataset.pull_item(idx)
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width = img.shape[:2]
        img_with_boxes = img.copy()
        
        if target is not None and len(target) > 0:
            for obj in target:
                # YOLOX format: [x1, y1, x2, y2, class_id]
                x1, y1, x2, y2, cls_id = obj
                
                # 정수형으로 변환
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                color = colors[int(cls_id) % len(colors)]
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
                
                cls_idx = int(cls_id)
                if cls_idx < len(class_names):
                    label = f"{class_names[cls_idx]}"
                else:
                    label = f"Class {cls_idx}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_with_boxes, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # RGB to BGR for cv2.imwrite
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        
        output_path = os.path.join(save_dir, f"sample_{idx:03d}.png")
        cv2.imwrite(output_path, img_with_boxes)
        print(f"  - {output_path}")
    
    print(f"\n✅ 완료! {save_dir} 폴더를 확인하세요.")
    plt.show()


if __name__ == "__main__":
    # 데이터셋 로드
    data_dir = "/home/uwanggood/문서/workspace/yolo_resize_dataset_256"
    
    print("=" * 60)
    print("YOLO 데이터셋 시각화")
    print("=" * 60)
    print()
    
    # Train 데이터셋
    print("📁 Train 데이터셋 로드 중...")
    train_dataset = YOLOFormatDataset(
        data_dir=data_dir,
        img_dir="images",
        label_dir="labels",
        split="train",
        img_size=(256, 256),
        preproc=None,  # 전처리 없이 원본 이미지
        cache=False
    )
    
    print(f"✅ Train 데이터셋 로드 완료: {len(train_dataset)} 이미지\n")
    
    # 시각화
    visualize_sample(train_dataset, num_samples=9, save_dir="./visualization/train")
    
    print("\n" + "=" * 60)
    print("📁 Val 데이터셋 로드 중...")
    
    # Val 데이터셋
    val_dataset = YOLOFormatDataset(
        data_dir=data_dir,
        img_dir="images",
        label_dir="labels",
        split="val",
        img_size=(256, 256),
        preproc=None,
        cache=False
    )
    
    print(f"✅ Val 데이터셋 로드 완료: {len(val_dataset)} 이미지\n")
    
    # 시각화
    visualize_sample(val_dataset, num_samples=9, save_dir="./visualization/val")

