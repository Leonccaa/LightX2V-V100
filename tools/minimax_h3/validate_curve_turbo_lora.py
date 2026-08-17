#!/usr/bin/env python3
"""Validate the curve-compatible MiniMax-H3 Turbo LoRA conversion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


def assert_equal(expected: torch.Tensor, actual: torch.Tensor, label: str) -> None:
    if expected.dtype != actual.dtype or expected.shape != actual.shape or not torch.equal(expected, actual):
        raise AssertionError(f"LoRA conversion mismatch: {label}")


def validate(args: argparse.Namespace) -> None:
    with safe_open(args.source, framework="pt", device="cpu") as source, safe_open(
        args.target, framework="pt", device="cpu"
    ) as target:
        source_keys = set(source.keys())
        target_keys = set(target.keys())
        if len(source_keys) != 416 or len(target_keys) != 624:
            raise AssertionError(f"unexpected tensor counts: source={len(source_keys)}, target={len(target_keys)}")
        if any("adaln" in key for key in source_keys | target_keys):
            raise AssertionError("curve-compatible Turbo LoRA must not contain AdaLN adapters")

        exact_checks = 0
        mappings = {
            "attn.out_proj": "attn.to_out.0",
            "mlp.fc1": "ff.net.0.proj",
            "mlp.fc2": "ff.net.2",
        }
        module_pairs = [
            (f"diffusion_model.blocks.{index}", f"transformer_blocks.{index}")
            for index in range(50)
        ]
        module_pairs.extend(
            (
                f"diffusion_model.token_refiner.blocks.{index}",
                f"token_refiner.refiner_blocks.{index}",
            )
            for index in range(2)
        )
        for source_prefix, target_prefix in module_pairs:
            qkv_a = source.get_tensor(f"{source_prefix}.attn.qkv_proj.lora_A.weight")
            qkv_b = source.get_tensor(f"{source_prefix}.attn.qkv_proj.lora_B.weight")
            for name, b_shard in zip(("to_q", "to_k", "to_v"), qkv_b.chunk(3, dim=0), strict=True):
                assert_equal(qkv_a, target.get_tensor(f"{target_prefix}.attn.{name}.lora_A.weight"), f"{target_prefix}.{name}.A")
                assert_equal(b_shard, target.get_tensor(f"{target_prefix}.attn.{name}.lora_B.weight"), f"{target_prefix}.{name}.B")
                exact_checks += 2
            for source_suffix, target_suffix in mappings.items():
                for factor in ("A", "B"):
                    assert_equal(
                        source.get_tensor(f"{source_prefix}.{source_suffix}.lora_{factor}.weight"),
                        target.get_tensor(f"{target_prefix}.{target_suffix}.lora_{factor}.weight"),
                        f"{target_prefix}.{target_suffix}.{factor}",
                    )
                    exact_checks += 1

        if exact_checks != len(target_keys):
            raise AssertionError(f"not every target tensor was checked: {exact_checks} != {len(target_keys)}")
        report = {
            "source": str(args.source),
            "target": str(args.target),
            "source_tensors": len(source_keys),
            "target_tensors": len(target_keys),
            "target_dtype_counts": dict(Counter(str(target.get_slice(key).get_dtype()) for key in target_keys)),
            "exact_tensor_checks": exact_checks,
            "adaln_pairs_removed_upstream": 51,
            "retained_source_pairs": len(source_keys) // 2,
            "status": "pass",
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
