# MiniMax-H3 on 4x Tesla V100 with LightX2V TP4

## Status and scope

This downstream branch demonstrates MiniMax-H3 DiT inference on four
Tesla V100 PCIe 32 GB GPUs with LightX2V tensor parallelism. It was developed
from `ModelTC/LightX2V` commit
`60631ff15310d645128226a98f138245bd59eba0`.

The validated path uses:

- FP16 Tensor Core computation across TP4;
- FP32 for numerically sensitive text-refiner, normalization, residual, and
  overflow-protection paths;
- PyTorch SDPA attention and native PyTorch RMSNorm/RoPE;
- optional official pruned AdaLN-curve FL2VA or Ref2VA checkpoints;
- precomputed T2V, I2V, or R2V conditioning bundles;
- a DiT-only TP4 stage followed by a separate official-VAE decode stage.

This is not an upstream-supported LightX2V configuration. It does not include
model, LoRA, text-encoder, or VAE weights, and it does not bypass any upstream
license or access terms.

## Why these changes were needed

V100 is an SM70 GPU without native BF16 Tensor Core execution. Loading the H3
checkpoint as ordinary BF16 or retaining all temporary paths in FP32 either
fails or leaves insufficient memory for useful video shapes.

The implementation makes four main changes:

1. **V100 mixed precision.** Main DiT matrix operations run in FP16, while
   validated sensitive paths remain FP32.
2. **Real TP shard ownership.** Each rank explicitly materializes its own
   tensor-parallel shard. A contiguous view had retained the complete source
   tensor storage and raised resident memory to about 30.6 GiB per rank.
3. **SM70-compatible primitives.** The validated backend uses PyTorch SDPA,
   native RMSNorm, and real-valued native RoPE. A tested TileLang attention
   kernel was slower than SDPA and was not selected.
4. **Separated pipeline stages.** Four V100s run only the DiT. Official video
   and audio VAEs decode saved finite latents separately, avoiding four
   redundant VAE copies.

The pruned checkpoint path additionally implements linear interpolation over
the released AdaLN curve table and its projection path.

## Data flow

```text
official H3 conditioner
        |
        v
conditioning.safetensors
  prompt embeddings + token tags
  optional keyframe/reference latents
        |
        v
4x V100 FP16 TP4 DiT
  FP32 sensitive paths
  finite checks per transformer block
        |
        v
video/audio latent rows
        |
        v
single-device official video/audio VAE
        |
        v
H.264 + AAC output
```

## Relevant files

- `lightx2v/models/networks/minimax_h3/`: V100 precision and pruned
  AdaLN-curve support.
- `lightx2v/models/runners/minimax_h3/minimax_h3_runner.py`: conditioning
  bundle loading, task restoration, DiT-only output, and finite gates.
- `tools/minimax_h3/run_tp4_gate.py`: minimal single-evaluation gate.
- `tools/minimax_h3/run_tp4_benchmark.py`: repeatable TP4 runner and
  structured measurements.
- `tools/minimax_h3/convert_pruned_curve_to_lightx2v.py`: lossless key
  remap, QKV/FC split, and sharding conversion.
- `tools/minimax_h3/validate_pruned_curve_conversion.py`: exact conversion
  checks.
- `tools/minimax_h3/decode_h3_latents.py`: separate official-VAE decoder.
- `examples/minimax_h3/H3ConditionExport/`: ComfyUI conditioning export
  node.

## Measured environment

The accepted runs used four Tesla V100 PCIe 32 GB GPUs, PyTorch
`2.10.0+cu126`, CUDA 12.6, and NCCL 2.27.5. Before inference, all 12 directed
CUDA P2P paths and TP4 NCCL all-reduce were tested, and volatile uncorrected
ECC was zero.

These versions describe the measured environment; they are not a claim that
other combinations cannot work.

## Prepare a pruned checkpoint

Obtain the checkpoint under its upstream terms. No checkpoint is redistributed
here.

```bash
python tools/minimax_h3/convert_pruned_curve_to_lightx2v.py \
  --source /path/to/minimax_h3_ref2va_pruned_bf16.safetensors \
  --base-config /path/to/full-lightx2v-transformer/config.json \
  --output /path/to/converted-pruned-ref2va
```

Validate the conversion against the corresponding full LightX2V key contract:

```bash
python tools/minimax_h3/validate_pruned_curve_conversion.py \
  --source /path/to/minimax_h3_ref2va_pruned_bf16.safetensors \
  --target /path/to/converted-pruned-ref2va \
  --full-index /path/to/full-lightx2v-transformer/diffusion_pytorch_model.safetensors.index.json \
  --report conversion-validation.json
```

The accepted conversion contained 635 tensors, 40,225,668,128 tensor bytes,
and passed 123 exact tensor checks. Conversion is a layout transformation, not
quantization or calibration.

## Export conditioning

Copy `examples/minimax_h3/H3ConditionExport` into the ComfyUI
`custom_nodes` directory and restart ComfyUI. The included
`export_ref2av_conditioning_api.json` shows the R2V graph. Replace the
placeholder text-encoder filename, prompt, image, and output path.

The resulting safetensors bundle contains text embeddings and token tags,
plus keyframe or reference latents when the selected H3 conditioning node
produces them. It contains no raw image bytes.

## Run the TP4 DiT stage

```bash
export DTYPE=FP16
export SENSITIVE_LAYER_DTYPE=FP32
export LIGHTX2V_MINIMAL_IMPORT=1

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29542 \
  tools/minimax_h3/run_tp4_benchmark.py \
  --config configs/minimax_h3/minimax_h3_ref2av_v100_tp4.json \
  --model-path /path/to/minimax-h3-lightx2v-model-root \
  --transformer-path /path/to/converted-pruned-ref2va \
  --task ref2av \
  --condition-path /path/to/minimax-h3-ref2av-conditioning.safetensors \
  --prompt "the same prompt used to create the conditioning bundle" \
  --seed 2026081603 \
  --case-id h3-v100-ref2av \
  --width 864 --height 480 --frames 124 --nominal-seconds 5 \
  --trials 1 --block-finite-check \
  --evidence-dir ./output/h3-v100-ref2av \
  --report ./output/h3-v100-ref2av/benchmark.json \
  --effective-config ./output/h3-v100-ref2av/effective-config.json
```

The tool writes finite video/audio latent rows. Decode them with:

```bash
python tools/minimax_h3/decode_h3_latents.py \
  --config configs/minimax_h3/minimax_h3_ref2av_v100_tp4.json \
  --model-path /path/to/minimax-h3-lightx2v-model-root \
  --latents ./output/h3-v100-ref2av/h3-v100-ref2av-trial1-latents.safetensors \
  --output ./output/h3-v100-ref2av/output.mp4 \
  --report ./output/h3-v100-ref2av/decode.json
```

## Validation boundary

The original validation covered finite tensors, deterministic repeated
latents where reported, media structure, GPU memory, P2P/NCCL, ECC, and visual
inspection. Runtime success is not a universal model-quality claim. See
[the benchmark report](minimax_h3_v100_tp4_benchmark.md) for measured
resolution, duration, memory, timing, and the 32 GB capacity boundary.

## License and publication boundary

LightX2V code remains under Apache-2.0. MiniMax-H3 checkpoints and related
artifacts remain under their own upstream terms. This branch deliberately
contains no model weights, private service configuration, internal network
names, or raw operational logs.
