#!/usr/bin/env python3
"""Decode a MiniMax-H3 latent queue while keeping both official VAEs resident."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from decode_h3_latents import _load_config, _load_rows, _sha256

from lightx2v.models.audio_encoders.hf.minimax_h3 import MiniMaxH3AudioVAE
from lightx2v.models.networks.minimax_h3.packing import (
    audio_latent_num_frames,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from lightx2v.models.video_encoders.hf.ltx2.audio_vae.ops import Audio
from lightx2v.models.video_encoders.hf.minimax_h3 import MiniMaxH3VideoVAE
from lightx2v.utils.ltx2_media_io import encode_video


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


def _load_jobs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("--jobs-manifest must contain a non-empty JSON list")
    jobs: list[dict] = []
    for index, raw_job in enumerate(payload, start=1):
        if not isinstance(raw_job, dict):
            raise TypeError(f"job {index} must be an object")
        job = dict(raw_job)
        job_id = str(job.get("id", f"job-{index:02d}"))
        if not job.get("output_path") or not job.get("decoded_output_path"):
            raise ValueError(f"job {job_id!r} requires output_path and decoded_output_path")
        job["id"] = job_id
        jobs.append(job)
    return jobs


def _wait_for_file(path: Path, timeout_seconds: float) -> dict[str, float]:
    started_epoch = time.time()
    started = time.perf_counter()
    deadline = started + timeout_seconds
    while not path.is_file():
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"timed out waiting for DiT latents: {path}")
        time.sleep(0.1)
    return {
        "started_epoch": started_epoch,
        "finished_epoch": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--jobs-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--input-wait-seconds", type=float, default=1800.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MiniMax-H3 resident VAE worker")
    if args.input_wait_seconds <= 0:
        raise ValueError("--input-wait-seconds must be positive")

    process_started_epoch = time.time()
    process_started = time.perf_counter()
    config = _load_config(args.config, args.model_path)
    jobs = _load_jobs(args.jobs_manifest)

    num_frames = int(config["target_video_length"])
    height = int(config["target_height"])
    width = int(config["target_width"])
    spatial_scale = int(config.get("vae_spatial_scale_factor", 16))
    latent_frames = video_latent_num_frames(num_frames)
    latent_height = height // spatial_scale
    latent_width = width // spatial_scale
    num_audio_latents = audio_latent_num_frames(num_frames)
    patch_size = tuple(int(value) for value in config.get("patch_size", (1, 2, 2)))
    in_channels = int(config.get("in_channels", 24))
    target_video_rows = (
        latent_frames
        * (latent_height // patch_size[1])
        * (latent_width // patch_size[2])
    )
    target_audio_rows = num_audio_latents * int(config.get("audio_channels", 2))

    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    video_load_started = time.perf_counter()
    video_vae = MiniMaxH3VideoVAE.from_pretrained(args.model_path, device="cuda", cpu_offload=False)
    video_load_seconds = time.perf_counter() - video_load_started
    audio_load_started = time.perf_counter()
    audio_vae = MiniMaxH3AudioVAE.from_pretrained(args.model_path, device="cuda", cpu_offload=False)
    audio_load_seconds = time.perf_counter() - audio_load_started
    sample_rate = int(audio_vae.sampling_rate)
    load_memory = _cuda_memory()

    results = []
    for job in jobs:
        latents_path = Path(job["output_path"])
        output_path = Path(job["decoded_output_path"])
        if not latents_path.is_absolute():
            latents_path = args.jobs_manifest.parent / latents_path
        if not output_path.is_absolute():
            output_path = args.jobs_manifest.parent / output_path
        input_wait = _wait_for_file(latents_path, args.input_wait_seconds)
        rows, latent_metadata = _load_rows(latents_path)
        condition_video_rows = int(rows["video_rows"].shape[0]) - target_video_rows
        condition_audio_rows = int(rows["audio_rows"].shape[0]) - target_audio_rows
        if condition_video_rows < 0 or condition_audio_rows < 0:
            raise ValueError(
                f"saved latent rows are shorter than the target geometry for {job['id']}: "
                f"video={rows['video_rows'].shape[0]}/{target_video_rows}, "
                f"audio={rows['audio_rows'].shape[0]}/{target_audio_rows}"
            )
        source_layout = {
            "video_rows": int(rows["video_rows"].shape[0]),
            "audio_rows": int(rows["audio_rows"].shape[0]),
            "condition_video_rows": condition_video_rows,
            "condition_audio_rows": condition_audio_rows,
            "target_video_rows": target_video_rows,
            "target_audio_rows": target_audio_rows,
        }
        video_latents = unpatchify_video_tokens(
            rows["video_rows"][condition_video_rows:],
            latent_frames,
            latent_height,
            latent_width,
            channels=in_channels,
            patch_size=patch_size,
        )
        audio_latents = unpack_audio_tokens(rows["audio_rows"][condition_audio_rows:], num_audio_latents)

        torch.cuda.reset_peak_memory_stats()
        job_started_epoch = time.time()
        job_started = time.perf_counter()
        video_started = time.perf_counter()
        video = video_vae.decode(video_latents, return_cpu=True)
        video_seconds = time.perf_counter() - video_started
        del video_latents
        torch.cuda.empty_cache()

        audio_started = time.perf_counter()
        audio = audio_vae.decode(audio_latents, return_cpu=True)
        audio_seconds = time.perf_counter() - audio_started
        del audio_latents, rows
        torch.cuda.empty_cache()

        if video.shape != (1, 3, num_frames, height, width):
            raise ValueError(
                f"decoded video shape mismatch for {job['id']}: "
                f"expected {(1, 3, num_frames, height, width)}, got {tuple(video.shape)}"
            )
        if audio.ndim != 3 or audio.shape[:2] != (1, 2):
            raise ValueError(f"decoded audio for {job['id']} must be [1,2,samples], got {tuple(audio.shape)}")
        if not bool(torch.isfinite(video).all()) or not bool(torch.isfinite(audio).all()):
            raise FloatingPointError(f"official VAE decode produced NaN or Inf for {job['id']}")

        frames = (video[0].permute(1, 2, 3, 0).float() * 255.0).round().clamp_(0, 255).to(torch.uint8)
        waveform = audio[0].float()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_started = time.perf_counter()
        encode_video(
            video=frames,
            fps=int(config.get("fps", 24)),
            audio=Audio(waveform=waveform, sampling_rate=sample_rate),
            output_path=str(output_path),
            video_chunks_number=1,
        )
        encode_seconds = time.perf_counter() - encode_started
        job_memory = _cuda_memory()
        job_finished_epoch = time.time()
        results.append(
            {
                "id": job["id"],
                "source_latents": str(latents_path),
                "source_latents_sha256": _sha256(latents_path),
                "source_metadata": latent_metadata,
                "source_layout": source_layout,
                "output": str(output_path),
                "output_sha256": _sha256(output_path),
                "input_wait": input_wait,
                "started_epoch": job_started_epoch,
                "finished_epoch": job_finished_epoch,
                "timing_seconds": {
                    "video_vae_decode": video_seconds,
                    "audio_vae_decode": audio_seconds,
                    "encode_mp4": encode_seconds,
                    "job_total": time.perf_counter() - job_started,
                },
                "cuda_memory": job_memory,
                "decoded": {
                    "video_shape": list(video.shape),
                    "audio_shape": list(audio.shape),
                    "audio_rms": float(audio.square().mean().sqrt()),
                },
            }
        )
        del video, audio, frames, waveform
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "status": "pass",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "resident": {
            "video_vae": next(video_vae.parameters()).device.type == "cuda",
            "audio_vae": next(audio_vae.parameters()).device.type == "cuda",
            "video_vae_load_seconds": video_load_seconds,
            "audio_vae_load_seconds": audio_load_seconds,
            "cuda_memory_after_load": load_memory,
        },
        "geometry": {
            "frames": num_frames,
            "height": height,
            "width": width,
            "fps": int(config.get("fps", 24)),
        },
        "jobs": results,
        "process_started_epoch": process_started_epoch,
        "process_finished_epoch": time.time(),
        "process_elapsed_seconds": time.perf_counter() - process_started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
