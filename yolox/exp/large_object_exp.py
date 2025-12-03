#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp

__all__ = ["LargeObjectExp"]


class LargeObjectExp(BaseExp):
    """
    큰 물체 전용 학습을 위한 실험 설정.
    256x256 입력 크기로 ConvNeXt 백본과 단일 스케일 헤드를 사용합니다.
    """

    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 2
        self.width = 1.0
        self.act = "silu"

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 4
        self.input_size = (256, 256)
        self.multiscale_range = 0
        self.data_dir = "/home/uwanggood/문서/workspace/yolo_resize_dataset_256"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.test_ann = "instances_test.json"
        self.use_yolo_format = True

        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.0  # 0.5 -> 0.0 (Mosaic 끄기)
        self.mixup_prob = 0.0   # 0.5 -> 0.0 (Mixup 끄기)
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.enable_mixup = False  # True -> False
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # 학습률을 낮게 설정 (backbone collapse 방지)
        self.basic_lr_per_img = 0.0001  # 0.001 -> 0.0001 (10배 낮춤)
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        # Gradient clipping for stability
        self.max_grad_norm = 5.0  # 10.0 -> 5.0 (더 엄격하게)
        self.print_interval = 10
        self.eval_interval = 10
        self.save_history_ckpt = True
        self.exp_name = "large_object_convnext"

        # -----------------  testing config ------------------ #
        self.test_size = (256, 256)
        self.test_conf = 0.01
        self.nmsthre = 0.65

    def get_model(self):
        from yolox.models import YOLOX, ConvNeXtBackbone, YOLOXLargeObjectHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            # ConvNeXt V2 백본 (Pretrained 제거 - 처음부터 학습)
            backbone = ConvNeXtBackbone(
                single_scale=True,
                pretrained=None,  # Pretrained 제거
            )
            head = YOLOXLargeObjectHead(
                num_classes=self.num_classes,
                width=self.width,
                in_channels=768,
                act=self.act,
            )
            self.model = YOLOX(backbone=backbone, head=head, large_object_mode=True)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import TrainTransform
        from yolox.data.datasets import YOLOFormatDataset

        if self.use_yolo_format:
            return YOLOFormatDataset(
                data_dir=self.data_dir,
                img_dir="images",
                label_dir="labels",
                split="train",
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob
                ),
                cache=cache,
                cache_type=cache_type,
            )
        else:
            from yolox.data import COCODataset
            return COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob
                ),
                cache=cache,
                cache_type=cache_type,
            )

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img: str = None
    ):
        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, \
                    "cache_img must be None if you didn't create self.dataset before launch"
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        self.dataset = MosaicDetection(
            self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_val_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def get_val_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import (
            ValTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )
        from yolox.data.datasets import YOLOFormatDataset

        if self.use_yolo_format:
            split = "val" if not testdev else "test"
            valdataset = YOLOFormatDataset(
                data_dir=self.data_dir,
                img_dir="images",
                label_dir="labels",
                split=split,
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy),
            )
        else:
            from yolox.data.datasets import COCODataset
            valdataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.val_ann if not testdev else self.test_ann,
                name="val2017" if not testdev else "test2017",
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy),
            )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        val_loader = DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def eval(self, model, evaluator, weights=None, return_outputs=False):
        """
        Evaluate the model using the evaluator.
        
        Args:
            model: model to evaluate
            evaluator: evaluator instance (COCOEvaluator)
            weights: (unused, for compatibility)
            return_outputs: whether to return prediction outputs
        """
        return evaluator.evaluate(
            model, 
            distributed=False,
            half=True,  # Use FP16 for evaluation (faster, less memory)
            return_outputs=return_outputs
        )

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = torch.nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[:, :, 1::2] *= scale_x
            targets[:, :, 2::2] *= scale_y
        return inputs, targets

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        """
        Random resize for multiscale training.
        Since we have multiscale_range=0, we just return the original size.
        """
        tensor = torch.LongTensor(2).to(rank)
        if rank == 0:
            # No random resizing, use fixed size
            size = self.input_size
            tensor[0] = size[0]
            tensor[1] = size[1]
        if is_distributed:
            import torch.distributed as dist
            dist.barrier()
            dist.broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

