#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import copy

import cv2
import numpy as np
from pathlib import Path

from .datasets_wrapper import CacheDataset, cache_read_img
from ..dataloading import get_yolox_datadir


class YOLOFormatDataset(CacheDataset):
    """
    YOLO 형식 데이터셋 클래스.
    라벨 형식: class_id center_x center_y width height (정규화된 좌표)
    """

    def __init__(
            self,
            data_dir=None,
            img_dir="images",
            label_dir="labels",
            split="train",
            img_size=(416, 416),
            preproc=None,
            cache=False,
            cache_type="ram",
            classes=None,
    ):
        """
        Args:
            data_dir (str): 데이터셋 루트 디렉토리
            img_dir (str): 이미지 디렉토리 이름 (기본: "images")
            label_dir (str): 라벨 디렉토리 이름 (기본: "labels")
            split (str): train 또는 val
            img_size (tuple): 타겟 이미지 크기
            preproc: 데이터 전처리 함수
            cache (bool): 캐시 사용 여부
            cache_type (str): 캐시 타입 ("ram" 또는 "disk")
            classes (list): 클래스 이름 리스트 (None이면 자동으로 카테고리에서 로드)
        """
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "YOLO")
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, img_dir, split)
        self.label_dir = os.path.join(data_dir, label_dir, split)
        self.split = split
        self.img_size = img_size
        self.preproc = preproc

        if classes is None:
            categories_file = os.path.join(data_dir, "annotations", "categories.json")
            if os.path.exists(categories_file):
                import json
                with open(categories_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "categories" in data:
                        categories = data["categories"]
                    elif isinstance(data, list):
                        categories = data
                    else:
                        categories = []
                if categories:
                    self._classes = tuple([cat["name"] for cat in sorted(categories, key=lambda x: x["id"])])
                    self.class_ids = [cat["id"] for cat in sorted(categories, key=lambda x: x["id"])]
                else:
                    self._classes = ("grapple", "truck")
                    self.class_ids = [0, 1]
            else:
                self._classes = ("grapple", "truck")
                self.class_ids = [0, 1]
        else:
            self._classes = tuple(classes)
            self.class_ids = list(range(len(classes)))

        self.cats = [
            {"id": idx, "name": val} for idx, val in enumerate(self._classes)
        ]

        self.ids = self._get_image_ids()
        self.num_imgs = len(self.ids)
        self.annotations = self._load_annotations()

        # Create COCO API compatible object for evaluation
        self._create_coco_api()

        path_filename = [os.path.join(img_dir, split, f"{img_id}.png")
                         if not img_id.endswith('.png') and not img_id.endswith('.jpg')
                         else os.path.join(img_dir, split, img_id)
                         for img_id in self.ids]

        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{split}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return self.num_imgs

    def pull_item(self, idx):
        label, img_info, resized_info, img_file = self.annotations[idx]
        img = self.read_img(idx)
        
        import copy
        return img, copy.deepcopy(label), img_info, np.array([idx])


    def _get_image_ids(self):
        """이미지 ID 리스트를 가져옵니다."""
        img_files = []
        if os.path.exists(self.img_dir):
            for ext in ['.png', '.jpg', '.jpeg']:
                img_files.extend(Path(self.img_dir).glob(f"*{ext}"))
        img_ids = [f.stem for f in img_files]
        return sorted(img_ids)

    def _load_annotations(self):
        """모든 어노테이션을 로드합니다."""
        annotations = []
        for img_id in self.ids:
            annotations.append(self.load_anno_from_id(img_id))
        return annotations

    def load_anno_from_id(self, img_id):
        """특정 이미지 ID의 어노테이션을 로드합니다."""
        label_file = os.path.join(self.label_dir, f"{img_id}.txt")

        if not os.path.exists(label_file):
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.img_dir, f"{img_id}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                return (np.zeros((0, 5)), (256, 256), (256, 256), f"{img_id}.png")

            img = cv2.imread(img_path)
            if img is None:
                return (np.zeros((0, 5)), (256, 256), (256, 256), f"{img_id}.png")

            height, width = img.shape[:2]
            return (np.zeros((0, 5)), (height, width), (height, width), os.path.basename(img_path))

        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(self.img_dir, f"{img_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            return (np.zeros((0, 5)), (256, 256), (256, 256), f"{img_id}.png")

        img = cv2.imread(img_path)
        if img is None:
            return (np.zeros((0, 5)), (256, 256), (256, 256), f"{img_id}.png")

        height, width = img.shape[:2]

        objs = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                # Convert normalized coords to pixel coords
                x1 = (center_x - w / 2) * width
                y1 = (center_y - h / 2) * height
                x2 = (center_x + w / 2) * width
                y2 = (center_y + h / 2) * height

                # Clamp to image boundaries BEFORE resize
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 > x1 + 1 and y2 > y1 + 1:  # 최소 크기 1x1
                    # YOLOX format: [x1, y1, x2, y2, class_id]
                    objs.append([x1, y1, x2, y2, class_id])

        num_objs = len(objs)
        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            res[ix] = obj

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        
        # Ensure bbox coordinates are valid after resizing
        resized_h = int(height * r)
        resized_w = int(width * r)
        res[:, 0] = np.clip(res[:, 0], 0, resized_w - 1)  # x1
        res[:, 1] = np.clip(res[:, 1], 0, resized_h - 1)  # y1
        res[:, 2] = np.clip(res[:, 2], res[:, 0] + 1, resized_w)  # x2 > x1
        res[:, 3] = np.clip(res[:, 3], res[:, 1] + 1, resized_h)  # y2 > y1
        
        # Validate class IDs
        res[:, 4] = np.clip(res[:, 4], 0, len(self._classes) - 1)
        
        # Remove invalid boxes (너무 작거나 좌표가 잘못된 것)
        valid_mask = (res[:, 2] > res[:, 0] + 1) & (res[:, 3] > res[:, 1] + 1)
        res = res[valid_mask]

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = os.path.basename(img_path)

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.img_dir, file_name)
        img = cv2.imread(img_file)
        assert img is not None, f"Failed to load image: {img_file}"
        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        인덱스에 해당하는 이미지/라벨 페어를 가져와 전처리합니다.
        
        Args:
            index (int): 데이터 인덱스
        
        Returns:
            img: 전처리된 이미지
            padded_labels: 전처리된 라벨
            img_info: (height, width) 튜플
            img_id: 이미지 ID
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        
        return img, target, img_info, img_id

    def _create_coco_api(self):
        """
        Create a COCO API compatible object for evaluation.
        This is a minimal implementation for COCOEvaluator to work.
        """
        from pycocotools.coco import COCO
        
        # Create COCO-style annotation dict
        coco_dict = {
            "info": {
                "description": "YOLO Format Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "",
                "date_created": "2025/11/29"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": self.cats
        }
        
        ann_id = 0
        for img_idx, img_id in enumerate(self.ids):
            label, img_info, _, img_file = self.annotations[img_idx]
            height, width = img_info
            
            # Add image info
            coco_dict["images"].append({
                "id": img_idx,
                "file_name": f"{img_id}.png",
                "width": int(width),
                "height": int(height),
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })
            
            # Add annotations (ground truth boxes)
            if label is not None and len(label) > 0:
                for obj in label:
                    cls_id, x, y, w, h = obj
                    # Convert from center format to xywh (top-left corner)
                    x1 = float(x - w / 2)
                    y1 = float(y - h / 2)
                    
                    coco_dict["annotations"].append({
                        "id": ann_id,
                        "image_id": img_idx,
                        "category_id": int(cls_id),
                        "bbox": [x1, y1, float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    ann_id += 1
        
        # Create COCO object from dict
        import io
        import contextlib
        
        # Suppress COCO initialization output
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO()
            self.coco.dataset = coco_dict
            self.coco.createIndex()


