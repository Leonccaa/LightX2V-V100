#!/usr/bin/env python3
"""Reproducible MiniMax-H3 TP4 benchmark with per-rank load/runtime telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from lightx2v.common.ops import *  # noqa: F401,F403
from lightx2v.models.networks.minimax_h3.packing import validate_t2av_geometry
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.utils.input_info import T2AVInputInfo
from lightx2v.utils.lockable_dict import LockableDict
from lightx2v.utils.set_config import set_parallel_config
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _process_memory() -> dict[str, int]:
    result: dict[str, int] = {}
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmSize"}:
                result[f"{key.lower()}_kib"] = int(value.strip().split()[0])
    return result


def _cuda_memory() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
    }


def _load_config(args: argparse.Namespace) -> tuple[LockableDict, dict]:
    with args.config.open("r", encoding="utf-8") as handle:
        runtime_config = json.load(handle)
    runtime_config.update(
        {
            "target_video_length": args.frames,
            "target_height": args.height,
            "target_width": args.width,
            "h3_dit_only": True,
            "h3_finite_check": bool(args.block_finite_check),
            "h3_benchmark_load_telemetry": True,
            "h3_latent_output_path": str(args.evidence_dir / f"{args.case_id}-trial1-latents.safetensors"),
        }
    )
    with (args.model_path / "transformer" / "config.json").open("r", encoding="utf-8") as handle:
        transformer_config = json.load(handle)

    merged = {
        "model_cls": "minimax_h3",
        "task": "t2av",
        "model_path": str(args.model_path),
        "dit_original_ckpt": str(args.model_path / "transformer"),
        "use_prompt_enhancer": False,
        "warmup": False,
        "cfg_parallel": False,
        "seq_parallel": False,
        "enable_cfg": False,
        "dit_quantized": False,
        "dit_quant_scheme": "Default",
    }
    merged.update(runtime_config)
    merged.update(transformer_config)
    merged.update(runtime_config)
    return LockableDict(merged), runtime_config


def _timed_cuda_call(function, *args, **kwargs):
    torch.cuda.synchronize()
    started_epoch = time.time()
    started = time.perf_counter()
    result = function(*args, **kwargs)
    torch.cuda.synchronize()
    return result, {
        "started_epoch": started_epoch,
        "finished_epoch": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--nominal-seconds", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--effective-config", required=True, type=Path)
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--block-finite-check", action="store_true")
    args = parser.parse_args()

    validate_t2av_geometry(args.frames, args.height, args.width)
    if args.trials < 1:
        raise ValueError("--trials must be positive")

    config, runtime_config = _load_config(args)
    platform_device = PLATFORM_DEVICE_REGISTER[os.getenv("PLATFORM", "cuda")]
    process_started_epoch = time.time()
    parallel_started = time.perf_counter()
    platform_device.init_parallel_env()
    parallel_seconds = time.perf_counter() - parallel_started
    set_parallel_config(config)
    torch.set_grad_enabled(False)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if rank == 0:
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.effective_config.parent.mkdir(parents=True, exist_ok=True)
        with args.effective_config.open("w", encoding="utf-8") as handle:
            json.dump(runtime_config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        logger.info(
            "MiniMax-H3 benchmark {}: TP={}, dtype={}, shape={}x{}x{}f, trials={}, block_finite_check={}",
            args.case_id,
            config["parallel"]["tensor_p_size"],
            os.getenv("DTYPE"),
            args.width,
            args.height,
            args.frames,
            args.trials,
            args.block_finite_check,
        )
    dist.barrier(device_ids=[torch.cuda.current_device()])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started_epoch = time.time()
    load_started = time.perf_counter()
    runner = MiniMaxH3Runner(config)
    runner.init_modules()
    torch.cuda.synchronize()
    load_finished_epoch = time.time()
    load_seconds = time.perf_counter() - load_started

    load_report = {
        "started_epoch": load_started_epoch,
        "finished_epoch": load_finished_epoch,
        "elapsed_seconds": load_seconds,
        "parallel_init_seconds": parallel_seconds,
        "cuda_memory_after_load": _cuda_memory(),
        "process_memory_after_load": _process_memory(),
        "component_timing": getattr(runner.model, "h3_load_telemetry", {}),
        "checkpoint_files": getattr(runner.model, "h3_checkpoint_file_stats", []),
        "lora_files": getattr(runner.model, "h3_lora_file_stats", []),
    }

    active: dict = {"trial": None}
    original_input_encoder = runner.run_input_encoder
    original_init_run = runner.init_run
    original_run_segment = runner.run_segment
    original_model_infer = runner.model.infer

    def timed_input_encoder():
        result, timing = _timed_cuda_call(original_input_encoder)
        active["trial"]["conditioning"] = timing
        return result

    def timed_init_run():
        result, timing = _timed_cuda_call(original_init_run)
        active["trial"]["prepare_dit"] = timing
        active["trial"]["packed_sequence"] = {
            "text_rows": int(runner.inputs["text_encoder_output"]["prompt_embeds"].shape[0]),
            "audio_rows": int(runner.scheduler.audio_latents.shape[0]),
            "video_rows": int(runner.scheduler.video_latents.shape[0]),
            "total_rows": int(runner.scheduler.layout.sequence_length),
            "latent_frames": int(runner.scheduler.num_latent_frames),
            "latent_height": int(runner.scheduler.latent_height),
            "latent_width": int(runner.scheduler.latent_width),
        }
        return result

    def timed_model_infer(inputs):
        result, timing = _timed_cuda_call(original_model_infer, inputs)
        active["trial"]["model_evaluations"].append(timing)
        return result

    def timed_run_segment(segment_idx=0):
        result, timing = _timed_cuda_call(original_run_segment, segment_idx)
        active["trial"]["dit_segment"] = timing
        return result

    runner.run_input_encoder = timed_input_encoder
    runner.init_run = timed_init_run
    runner.model.infer = timed_model_infer
    runner.run_segment = timed_run_segment

    trials = []
    for trial_index in range(1, args.trials + 1):
        output_path = args.evidence_dir / f"{args.case_id}-trial{trial_index}-latents.safetensors"
        runner.set_config({"h3_latent_output_path": str(output_path)})
        input_info = T2AVInputInfo(
            seed=args.seed,
            prompt=args.prompt,
            target_shape=[args.height, args.width],
            target_video_length=args.frames,
        )
        trial = {
            "trial": trial_index,
            "seed": args.seed,
            "output": str(output_path),
            "model_evaluations": [],
        }
        active["trial"] = trial
        dist.barrier(device_ids=[torch.cuda.current_device()])
        torch.cuda.reset_peak_memory_stats()
        pipeline_started_epoch = time.time()
        pipeline_started = time.perf_counter()
        runner.run_pipeline(input_info)
        torch.cuda.synchronize()
        trial["pipeline"] = {
            "started_epoch": pipeline_started_epoch,
            "finished_epoch": time.time(),
            "elapsed_seconds": time.perf_counter() - pipeline_started,
        }
        trial["cuda_memory"] = _cuda_memory()
        trial["process_memory"] = _process_memory()
        if rank == 0:
            with safe_open(output_path, framework="pt", device="cpu") as handle:
                trial["latent_metadata"] = dict(handle.metadata() or {})
            trial["output_size_bytes"] = output_path.stat().st_size
            trial["output_sha256"] = _sha256(output_path)
        trials.append(trial)
        dist.barrier(device_ids=[torch.cuda.current_device()])

    device_properties = torch.cuda.get_device_properties(local_rank)
    rank_report = {
        "rank": rank,
        "local_rank": local_rank,
        "device": {
            "name": device_properties.name,
            "total_memory_bytes": device_properties.total_memory,
            "compute_capability": [device_properties.major, device_properties.minor],
        },
        "load": load_report,
        "trials": trials,
    }
    all_ranks = [None for _ in range(world_size)]
    dist.all_gather_object(all_ranks, rank_report)

    if rank == 0:
        report = {
            "status": "pass",
            "case_id": args.case_id,
            "geometry": {
                "nominal_duration_seconds": args.nominal_seconds,
                "encoded_duration_seconds": args.frames / 24.0,
                "frames": args.frames,
                "height": args.height,
                "width": args.width,
                "fps": 24,
            },
            "benchmark": {
                "trials": args.trials,
                "block_finite_check": bool(args.block_finite_check),
                "feature_caching": config.get("feature_caching", "NoCaching"),
                "attention": config.get("attn_type"),
                "rms": config.get("rms_type"),
                "rope": config.get("rope_type"),
                "dtype": os.getenv("DTYPE"),
                "sensitive_layer_dtype": os.getenv("SENSITIVE_LAYER_DTYPE"),
                "tensor_parallel": int(config["parallel"]["tensor_p_size"]),
                "model_evaluations": len(config["h3_base_sigmas"]) - 1,
            },
            "fixture": {
                "prompt": args.prompt,
                "seed": args.seed,
                "precomputed_condition_path": config.get("precomputed_condition_path"),
            },
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": list(torch.cuda.nccl.version()),
                "world_size": world_size,
                "source_revision": args.source_revision,
                "process_started_epoch": process_started_epoch,
                "process_finished_epoch": time.time(),
                "distributed_environment": {
                    name: os.getenv(name)
                    for name in (
                        "CUDA_VISIBLE_DEVICES",
                        "NCCL_P2P_DISABLE",
                        "NCCL_SHM_DISABLE",
                        "NCCL_ALGO",
                        "NCCL_PROTO",
                    )
                },
            },
            "ranks": all_ranks,
        }
        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("H3_BENCHMARK_REPORT " + json.dumps(report, sort_keys=True))

    dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
