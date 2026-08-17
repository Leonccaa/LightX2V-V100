#!/usr/bin/env python3
"""Convert the fused ComfyUI MiniMax-H3 Turbo v4 LoRA for LightX2V."""

import argparse
import os

from safetensors import safe_open
from safetensors.torch import save_file


def _pair(source, prefix):
    return source.get_tensor(f"{prefix}.lora_A.weight"), source.get_tensor(f"{prefix}.lora_B.weight")


def _store_pair(output, prefix, lora_a, lora_b):
    # Q/K/V share A in the fused source, and B is split as views. Safetensors
    # intentionally rejects aliased storage, so materialize independent files.
    output[f"{prefix}.lora_A.weight"] = lora_a.clone().contiguous()
    output[f"{prefix}.lora_B.weight"] = lora_b.clone().contiguous()


def _convert_block(source, output, source_prefix, target_prefix):
    lora_a, lora_b = _pair(source, f"{source_prefix}.attn.qkv_proj")
    if lora_b.shape[0] % 3:
        raise ValueError(f"QKV LoRA up tensor is not divisible by three: {tuple(lora_b.shape)}")
    for name, shard in zip(("to_q", "to_k", "to_v"), lora_b.chunk(3, dim=0), strict=True):
        _store_pair(output, f"{target_prefix}.attn.{name}", lora_a, shard)

    mappings = {
        "attn.out_proj": "attn.to_out.0",
        "mlp.fc1": "ff.net.0.proj",
        "mlp.fc2": "ff.net.2",
    }
    for source_suffix, target_suffix in mappings.items():
        lora_a, lora_b = _pair(source, f"{source_prefix}.{source_suffix}")
        _store_pair(output, f"{target_prefix}.{target_suffix}", lora_a, lora_b)


def convert(source_path, output_path):
    output = {}
    with safe_open(source_path, framework="pt", device="cpu") as source:
        keys = set(source.keys())
        source_namespace = "diffusion_model." if all(key.startswith("diffusion_model.") for key in keys) else ""
        curve_compatible = not any(".adaln_proj.linear." in key for key in keys)
        for index in range(50):
            source_prefix = f"{source_namespace}blocks.{index}"
            target_prefix = f"transformer_blocks.{index}"
            _convert_block(source, output, source_prefix, target_prefix)
            if not curve_compatible:
                lora_a, lora_b = _pair(source, f"{source_prefix}.adaln_proj.linear")
                _store_pair(output, f"{target_prefix}.adaln_proj.linear", lora_a, lora_b)

        for index in range(2):
            _convert_block(
                source,
                output,
                f"{source_namespace}token_refiner.blocks.{index}",
                f"token_refiner.refiner_blocks.{index}",
            )

        if not curve_compatible:
            lora_a, lora_b = _pair(source, f"{source_namespace}final_layer.adaln_proj.linear")
            _store_pair(output, "norm_out.linear", lora_a, lora_b)

        for key in output:
            # Converted keys cannot be used to reconstruct fused QKV source
            # names, so verify the source contract independently below.
            if key.endswith(".alpha"):
                raise AssertionError("Turbo v4 conversion must not synthesize alpha tensors")
        expected_pairs = 50 * (4 if curve_compatible else 5) + 2 * 4 + (0 if curve_compatible else 1)
        source_pairs = len(keys) // 2
        if source_pairs != expected_pairs or any(key.endswith(".alpha") for key in keys):
            raise ValueError(
                f"Unexpected Turbo v4 source contract: tensors={len(keys)}, pairs={source_pairs}, expected_pairs={expected_pairs}"
            )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(
        output,
        output_path,
        metadata={
            "format": "LightX2V MiniMax-H3 LoRA",
            "source": os.path.basename(source_path),
            "scale": "1.0 (source has no alpha tensors)",
            "qkv": "split q/k/v with shared lora_A",
            "curve_compatible": str(curve_compatible).lower(),
            "source_namespace": source_namespace or "none",
        },
    )
    print(f"converted {len(output)} tensors -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
