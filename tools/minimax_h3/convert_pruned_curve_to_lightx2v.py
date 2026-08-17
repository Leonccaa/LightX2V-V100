#!/usr/bin/env python3
"""Convert a Comfy-Org MiniMax-H3 pruned BF16 single file for LightX2V.

The Comfy checkpoint already contains Q/K/V in contiguous groups. This tool
therefore splits the fused projection directly, swaps the fused SwiGLU halves
to LightX2V's ``[value; gate]`` layout, and retains the AdaLN curve table.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def convert_key(source_key: str, tensor: torch.Tensor, heads: int, head_dim: int):
    if source_key == "rope.inv_freq":
        return []
    if source_key == "adaln_t_table":
        return [(source_key, tensor)]

    target_key = source_key
    if target_key.startswith("token_refiner.blocks."):
        target_key = target_key.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif target_key.startswith("blocks."):
        target_key = target_key.replace("blocks.", "transformer_blocks.", 1)

    replacements = (
        ("video_patch_proj.", "proj_in."),
        ("audio_patch_proj.", "audio_proj_in."),
        ("condition_proj.", "context_embedder."),
        ("final_layer.norm.", "norm_out.norm."),
        ("final_layer.adaln_proj.linear.", "norm_out.linear."),
        ("final_layer.video_out.", "proj_out."),
        ("final_layer.audio_out.", "audio_proj_out."),
        (".attn.q_norm.", ".attn.norm_q."),
        (".attn.k_norm.", ".attn.norm_k."),
        (".attn.out_proj.", ".attn.to_out.0."),
    )
    for old, new in replacements:
        target_key = target_key.replace(old, new)

    if target_key.endswith(".attn.qkv_proj.weight"):
        inner_dim = heads * head_dim
        if tensor.shape[0] != 3 * inner_dim:
            raise ValueError(f"{source_key}: expected {3 * inner_dim} QKV rows, got {tensor.shape[0]}")
        query, key, value = tensor.split(inner_dim, dim=0)
        prefix = target_key.removesuffix("qkv_proj.weight")
        return [
            (f"{prefix}to_q.weight", query.contiguous()),
            (f"{prefix}to_k.weight", key.contiguous()),
            (f"{prefix}to_v.weight", value.contiguous()),
        ]

    if target_key.endswith(".mlp.fc1.weight"):
        gate, value = tensor.chunk(2, dim=0)
        target_key = target_key.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight")
        return [(target_key, torch.cat((value, gate), dim=0).contiguous())]

    target_key = target_key.replace(".mlp.fc2.", ".ff.net.2.")
    return [(target_key, tensor)]


def convert(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    with args.base_config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    written: list[Path] = []
    weight_map: dict[str, Path] = {}
    buffer: dict[str, torch.Tensor] = {}
    buffer_size = 0
    total_size = 0
    source_keys = 0
    target_keys = 0
    curve_shape: tuple[int, int] | None = None

    def flush() -> None:
        nonlocal buffer, buffer_size
        if not buffer:
            return
        path = output / f".tmp-shard-{len(written):05d}.safetensors"
        save_file(buffer, path, metadata={"format": "pt"})
        for key in buffer:
            weight_map[key] = path
        written.append(path)
        buffer = {}
        buffer_size = 0

    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        for source_key in checkpoint.keys():
            if source_key.endswith(".weight_scale"):
                raise ValueError(f"quantized checkpoint is not valid for the BF16 curve conversion: {source_key}")
            source_keys += 1
            source_tensor = checkpoint.get_tensor(source_key)
            if source_key == "adaln_t_table":
                curve_shape = tuple(source_tensor.shape)
            for target_key, target_tensor in convert_key(
                source_key,
                source_tensor,
                heads=int(config.get("num_attention_heads", 56)),
                head_dim=int(config.get("attention_head_dim", 128)),
            ):
                if target_key in weight_map or target_key in buffer:
                    raise KeyError(f"duplicate converted key: {target_key}")
                buffer[target_key] = target_tensor
                size = tensor_nbytes(target_tensor)
                buffer_size += size
                total_size += size
                target_keys += 1
            if buffer_size >= args.max_shard_size:
                flush()
    flush()

    if curve_shape is None or len(curve_shape) != 2 or curve_shape[0] < 2 or curve_shape[1] < 1:
        raise ValueError(f"invalid or missing adaln_t_table: {curve_shape}")
    if any(key.startswith("time_embedder.") for key in weight_map):
        raise ValueError("curve checkpoint unexpectedly retained a timestep MLP")
    expected_adaln = int(config.get("num_layers", 50)) * 2 + 2
    adaln_keys = [key for key in weight_map if ".adaln_proj.linear." in key or key.startswith("norm_out.linear.")]
    if len(adaln_keys) != expected_adaln:
        raise ValueError(f"expected {expected_adaln} curve AdaLN tensors, got {len(adaln_keys)}")

    final_names = [f"diffusion_pytorch_model-{index + 1:05d}-of-{len(written):05d}.safetensors" for index in range(len(written))]
    renames = {old: output / final for old, final in zip(written, final_names)}
    for old, new in renames.items():
        os.rename(old, new)

    index = {
        "metadata": {
            "total_size": total_size,
            "source_file": source.name,
            "source_keys": source_keys,
            "target_keys": target_keys,
            "adaln_curve_grid": curve_shape[0],
            "adaln_curve_basis": curve_shape[1],
        },
        "weight_map": {key: renames[path].name for key, path in sorted(weight_map.items())},
    }
    with (output / "diffusion_pytorch_model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    config.update(
        {
            "h3_adaln_curve": True,
            "adaln_curve_grid": curve_shape[0],
            "time_embed_dim": curve_shape[1],
        }
    )
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "source_keys": source_keys,
                "target_keys": target_keys,
                "shards": len(written),
                "total_size": total_size,
                "curve_shape": curve_shape,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-size", type=int, default=4 * 1024**3)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
