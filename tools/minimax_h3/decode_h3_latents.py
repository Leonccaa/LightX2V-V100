#!/usr/bin/env python3
"""Decode an isolated MiniMax-H3 DiT gate artifact with the official VAEs.

The TP ranks all hold the same final video/audio rows.  Decoding the rank-0
artifact in a separate single-GPU phase avoids loading four redundant copies
of the 10+ GiB VAE while preserving the exact TP4 denoising result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(config_path: Path, model_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with (model_path / "transformer" / "config.json").open("r", encoding="utf-8") as handle:
        transformer_config = json.load(handle)
    merged = dict(transformer_config)
    merged.update(config)
    return merged


def _load_rows(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        rows = {name: handle.get_tensor(name) for name in handle.keys()}
    expected = {"video_rows", "audio_rows"}
    if set(rows) != expected:
        raise ValueError(f"expected latent keys {sorted(expected)}, got {sorted(rows)}")
    if not bool(torch.isfinite(rows["video_rows"]).all()) or not bool(torch.isfinite(rows["audio_rows"]).all()):
        raise FloatingPointError("saved H3 rows contain NaN or Inf")
    return rows, metadata


def _cuda_peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MiniMax-H3 VAE acceptance gate")

    started = time.perf_counter()
    config = _load_config(args.config, args.model_path)
    rows, latent_metadata = _load_rows(args.latents)

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

    video_latents = unpatchify_video_tokens(
        rows["video_rows"],
        latent_frames,
        latent_height,
        latent_width,
        channels=in_channels,
        patch_size=patch_size,
    )
    audio_latents = unpack_audio_tokens(rows["audio_rows"], num_audio_latents)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    video_load_started = time.perf_counter()
    video_vae = MiniMaxH3VideoVAE.from_pretrained(
        args.model_path,
        device="cuda",
        cpu_offload=True,
    )
    video_load_seconds = time.perf_counter() - video_load_started
    video_decode_started = time.perf_counter()
    video = video_vae.decode(video_latents, return_cpu=True)
    video_decode_seconds = time.perf_counter() - video_decode_started
    video_peak_mib = _cuda_peak_mib()
    del video_vae, video_latents
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    audio_load_started = time.perf_counter()
    audio_vae = MiniMaxH3AudioVAE.from_pretrained(
        args.model_path,
        device="cuda",
        cpu_offload=True,
    )
    sample_rate = int(audio_vae.sampling_rate)
    audio_load_seconds = time.perf_counter() - audio_load_started
    audio_decode_started = time.perf_counter()
    audio = audio_vae.decode(audio_latents, return_cpu=True)
    audio_decode_seconds = time.perf_counter() - audio_decode_started
    audio_peak_mib = _cuda_peak_mib()
    del audio_vae, audio_latents, rows
    gc.collect()
    torch.cuda.empty_cache()

    if video.shape != (1, 3, num_frames, height, width):
        raise ValueError(f"decoded video shape mismatch: expected {(1, 3, num_frames, height, width)}, got {tuple(video.shape)}")
    if audio.ndim != 3 or audio.shape[:2] != (1, 2):
        raise ValueError(f"decoded audio must be [1,2,samples], got {tuple(audio.shape)}")
    if not bool(torch.isfinite(video).all()) or not bool(torch.isfinite(audio).all()):
        raise FloatingPointError("official VAE decode produced NaN or Inf")

    frames = (video[0].permute(1, 2, 3, 0).float() * 255.0).round().clamp_(0, 255).to(torch.uint8)
    waveform = audio[0].float()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_started = time.perf_counter()
    encode_video(
        video=frames,
        fps=int(config.get("fps", 24)),
        audio=Audio(waveform=waveform, sampling_rate=sample_rate),
        output_path=str(args.output),
        video_chunks_number=1,
    )
    encode_seconds = time.perf_counter() - encode_started

    report = {
        "status": "pass",
        "source_latents": str(args.latents),
        "source_latents_sha256": _sha256(args.latents),
        "source_metadata": latent_metadata,
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "geometry": {
            "frames": num_frames,
            "height": height,
            "width": width,
            "fps": int(config.get("fps", 24)),
            "latent_frames": latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "audio_latents": num_audio_latents,
        },
        "decoded": {
            "video_shape": list(video.shape),
            "video_min": float(video.min()),
            "video_max": float(video.max()),
            "video_mean": float(video.mean()),
            "video_std": float(video.std()),
            "audio_shape": list(audio.shape),
            "audio_min": float(audio.min()),
            "audio_max": float(audio.max()),
            "audio_rms": float(audio.square().mean().sqrt()),
            "sample_rate": sample_rate,
        },
        "timing_seconds": {
            "video_vae_load": video_load_seconds,
            "video_vae_decode": video_decode_seconds,
            "audio_vae_load": audio_load_seconds,
            "audio_vae_decode": audio_decode_seconds,
            "encode_mp4": encode_seconds,
            "decode_process_total": time.perf_counter() - started,
        },
        "peak_cuda_allocated_mib": {
            "video_vae": video_peak_mib,
            "audio_vae": audio_peak_mib,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
