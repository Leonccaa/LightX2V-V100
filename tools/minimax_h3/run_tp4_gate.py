#!/usr/bin/env python3
"""Minimal distributed runner for the isolated MiniMax-H3 TP4 gate."""

import argparse
import json
import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.common.ops import *  # noqa: F401,F403
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.utils.input_info import T2AVInputInfo
from lightx2v.utils.lockable_dict import LockableDict
from lightx2v.utils.set_config import set_parallel_config
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def load_config(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    transformer_config_path = os.path.join(args.model_path, "transformer", "config.json")
    with open(transformer_config_path, "r", encoding="utf-8") as handle:
        transformer_config = json.load(handle)

    merged = {
        "model_cls": "minimax_h3",
        "task": "t2av",
        "model_path": args.model_path,
        "dit_original_ckpt": os.path.join(args.model_path, "transformer"),
        "use_prompt_enhancer": False,
        "warmup": False,
        "cfg_parallel": False,
        "seq_parallel": False,
        "enable_cfg": False,
        "dit_quantized": False,
        "dit_quant_scheme": "Default",
    }
    merged.update(config)
    merged.update(transformer_config)
    # Runtime choices are authoritative over similarly named model metadata.
    merged.update(config)
    return LockableDict(merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    config = load_config(args)
    platform_device = PLATFORM_DEVICE_REGISTER[os.getenv("PLATFORM", "cuda")]
    platform_device.init_parallel_env()
    set_parallel_config(config)
    torch.set_grad_enabled(False)

    if dist.get_rank() == 0:
        logger.info(
            "MiniMax-H3 isolated gate: TP={}, dtype={}, shape={}x{}x{}f, steps={}",
            config["parallel"]["tensor_p_size"],
            os.getenv("DTYPE"),
            config["target_width"],
            config["target_height"],
            config["target_video_length"],
            len(config["h3_base_sigmas"]) - 1,
        )

    runner = MiniMaxH3Runner(config)
    runner.init_modules()
    input_info = T2AVInputInfo(
        seed=args.seed,
        prompt=args.prompt,
        target_shape=[int(config["target_height"]), int(config["target_width"])],
        target_video_length=int(config["target_video_length"]),
    )
    runner.run_pipeline(input_info)

    dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
