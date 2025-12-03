#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp
from yolox.exp.large_object_exp import LargeObjectExp
from yolox.utils import configure_nccl, configure_omp, get_num_devices


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Large Object Training")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-b", "--batch-size", type=int, default=64, help="batch size"
    )
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used, \
        support `tensorboard` and `wandb`",
        default="tensorboard"
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="caching imgs to ram/disk",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "-f16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    parser = make_parser()
    args = parser.parse_args()

    if args.exp_file is None:
        exp = LargeObjectExp()
    else:
        exp = get_exp(args.exp_file, args.name)
        exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.num_machines > 1 else None
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend="nccl",
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )

