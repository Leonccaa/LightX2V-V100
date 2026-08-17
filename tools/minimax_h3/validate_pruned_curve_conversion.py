#!/usr/bin/env python3
"""Validate a converted Comfy-Org MiniMax-H3 AdaLN-curve checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


def target_tensor(target: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(target / weight_map[key], framework="pt", device="cpu") as checkpoint:
        return checkpoint.get_tensor(key)


def assert_equal(source_tensor: torch.Tensor, target_tensor_value: torch.Tensor, label: str) -> None:
    if source_tensor.dtype != target_tensor_value.dtype:
        raise AssertionError(f"{label}: dtype mismatch {source_tensor.dtype} != {target_tensor_value.dtype}")
    if source_tensor.shape != target_tensor_value.shape:
        raise AssertionError(f"{label}: shape mismatch {tuple(source_tensor.shape)} != {tuple(target_tensor_value.shape)}")
    if not torch.equal(source_tensor, target_tensor_value):
        raise AssertionError(f"{label}: values differ")


def validate(args: argparse.Namespace) -> None:
    source_path = args.source.resolve()
    target_path = args.target.resolve()
    with args.full_index.open("r", encoding="utf-8") as handle:
        full_index = json.load(handle)
    with (target_path / "diffusion_pytorch_model.safetensors.index.json").open("r", encoding="utf-8") as handle:
        converted_index = json.load(handle)

    full_keys = set(full_index["weight_map"])
    expected_keys = {key for key in full_keys if not key.startswith("time_embedder.")}
    expected_keys.add("adaln_t_table")
    weight_map = converted_index["weight_map"]
    converted_keys = set(weight_map)
    if converted_keys != expected_keys:
        missing = sorted(expected_keys - converted_keys)
        extra = sorted(converted_keys - expected_keys)
        raise AssertionError(f"converted key contract mismatch: missing={missing[:8]}, extra={extra[:8]}")

    target_dtype_counts: Counter[str] = Counter()
    target_tensor_bytes = 0
    for key, shard_name in weight_map.items():
        with safe_open(target_path / shard_name, framework="pt", device="cpu") as checkpoint:
            view = checkpoint.get_slice(key)
            dtype = view.get_dtype()
            shape = view.get_shape()
        target_dtype_counts[dtype] += 1
        element_size = {"F16": 2, "BF16": 2, "F32": 4}[dtype]
        target_tensor_bytes += int(torch.tensor(shape).prod().item()) * element_size

    exact_checks = 0
    with safe_open(source_path, framework="pt", device="cpu") as source:
        assert_equal(
            source.get_tensor("adaln_t_table"),
            target_tensor(target_path, weight_map, "adaln_t_table"),
            "adaln_t_table",
        )
        exact_checks += 1

        curve_pairs = [
            ("final_layer.adaln_proj.linear.weight", "norm_out.linear.weight"),
            ("final_layer.adaln_proj.linear.bias", "norm_out.linear.bias"),
        ]
        for index in range(50):
            for suffix in ("weight", "bias"):
                curve_pairs.append(
                    (
                        f"blocks.{index}.adaln_proj.linear.{suffix}",
                        f"transformer_blocks.{index}.adaln_proj.linear.{suffix}",
                    )
                )
        for source_key, target_key in curve_pairs:
            assert_equal(
                source.get_tensor(source_key),
                target_tensor(target_path, weight_map, target_key),
                target_key,
            )
            exact_checks += 1

        for source_prefix, target_prefix in (
            ("blocks.0", "transformer_blocks.0"),
            ("blocks.25", "transformer_blocks.25"),
            ("blocks.49", "transformer_blocks.49"),
            ("token_refiner.blocks.0", "token_refiner.refiner_blocks.0"),
            ("token_refiner.blocks.1", "token_refiner.refiner_blocks.1"),
        ):
            fused_qkv = source.get_tensor(f"{source_prefix}.attn.qkv_proj.weight")
            query, key, value = fused_qkv.chunk(3, dim=0)
            for name, expected in (("to_q", query), ("to_k", key), ("to_v", value)):
                target_key = f"{target_prefix}.attn.{name}.weight"
                assert_equal(expected, target_tensor(target_path, weight_map, target_key), target_key)
                exact_checks += 1
            del fused_qkv, query, key, value

            fused_fc1 = source.get_tensor(f"{source_prefix}.mlp.fc1.weight")
            gate, value = fused_fc1.chunk(2, dim=0)
            target_key = f"{target_prefix}.ff.net.0.proj.weight"
            assert_equal(
                torch.cat((value, gate), dim=0),
                target_tensor(target_path, weight_map, target_key),
                target_key,
            )
            exact_checks += 1
            del fused_fc1, gate, value
            gc.collect()

    metadata_total = int(converted_index["metadata"]["total_size"])
    if metadata_total != target_tensor_bytes:
        raise AssertionError(f"index total_size mismatch: {metadata_total} != {target_tensor_bytes}")

    report = {
        "source": str(source_path),
        "target": str(target_path),
        "target_keys": len(converted_keys),
        "target_dtype_counts": dict(sorted(target_dtype_counts.items())),
        "target_tensor_bytes": target_tensor_bytes,
        "exact_tensor_checks": exact_checks,
        "key_contract": "full LightX2V keys minus four time_embedder tensors plus adaln_t_table",
        "status": "pass",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--full-index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
