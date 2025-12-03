#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

data_dir = '/home/uwanggood/문서/workspace/yolo_resize_dataset_256'
train_labels_dir = os.path.join(data_dir, 'labels', 'train')

print("="*80)
print("라벨 파일 검사")
print("="*80)

label_files = glob.glob(os.path.join(train_labels_dir, '*.txt'))
print(f"\n총 라벨 파일 개수: {len(label_files)}")

problems = []

for i, label_file in enumerate(label_files[:50]):  # 처음 50개만 체크
    with open(label_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            try:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # 문제 체크
                if class_id < 0 or class_id >= 2:  # num_classes=2
                    problems.append(f"{os.path.basename(label_file)}:{line_num} - class_id={class_id} (범위: 0-1)")
                
                if not (0 <= cx <= 1):
                    problems.append(f"{os.path.basename(label_file)}:{line_num} - cx={cx} (범위: 0-1)")
                
                if not (0 <= cy <= 1):
                    problems.append(f"{os.path.basename(label_file)}:{line_num} - cy={cy} (범위: 0-1)")
                
                if not (0 <= w <= 1):
                    problems.append(f"{os.path.basename(label_file)}:{line_num} - w={w} (범위: 0-1)")
                
                if not (0 <= h <= 1):
                    problems.append(f"{os.path.basename(label_file)}:{line_num} - h={h} (범위: 0-1)")
                    
            except ValueError as e:
                problems.append(f"{os.path.basename(label_file)}:{line_num} - 파싱 에러: {e}")

print("\n" + "="*80)
if problems:
    print(f"❌ 문제 발견! (총 {len(problems)}개)")
    print("\n처음 20개 문제:")
    for p in problems[:20]:
        print(f"  - {p}")
else:
    print("✅ 처음 50개 파일은 정상입니다!")

print("\n" + "="*80)

