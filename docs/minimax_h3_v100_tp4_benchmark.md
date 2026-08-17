# MiniMax-H3 on 4x Tesla V100 PCIe 32 GB: native-FP16 LightX2V TP4 benchmark

Benchmark date: 2026-08-10 America/Vancouver (raw evidence continues after midnight UTC on 2026-08-11).

## Headline result

Native-FP16 MiniMax-H3 is viable on four 32 GB V100 PCIe cards with LightX2V tensor parallelism. Eight of nine requested resolution/duration points completed two deterministic DiT trials plus official audio/video VAE decode. The practical high end is 960x544 at 15 seconds or 1344x768 at 5 seconds. The 1344x768 / 10-second point also runs, but uses 32,072 MiB per card by NVML and takes 681.83 seconds for the hot DiT phase. The 1344x768 / 15-second point fails with a clear CUDA capacity OOM.

The fastest tested point, 864x480 / 124 frames, takes:

- 25.11 seconds to start a fresh four-rank process and load/merge the model;
- 49.73 seconds for a same-process hot six-evaluation DiT run;
- 43.93 seconds for the separate official single-V100 video/audio VAE decode and MP4 encode;
- 93.66 seconds for hot DiT plus fresh VAE decode, excluding live text encoding.

This is a DiT/load/LoRA/VAE benchmark using one precomputed H3 text-conditioning tensor. It is not an end-to-end arbitrary-prompt text-encoder benchmark.

## Hardware and software

- Four NVIDIA Tesla V100 PCIe 32 GB cards, 250 W power limit, PCIe Gen3 x16.
- GPU0-GPU1 topology is PIX; every other pair is PXB. There is no NVLink.
- Dual-socket AMD EPYC host.
- 256 GiB system RAM; no swap.
- Ubuntu 24.04 container environment.
- NVIDIA driver `580.159.04`.
- PyTorch `2.10.0+cu126`; NCCL `2.27.5`.
- LightX2V upstream base `60631ff15310d645128226a98f138245bd59eba0` plus the local MiniMax-H3 V100 FP16/TP/SP implementation.
- An upstream MiniMax-H3 snapshot obtained under its separate license.
- MiniMax-H3 Turbo LoRA v4 merged at load time.
- Model files were read from warm shared storage; this was not a cold-storage benchmark.

## P2P gate before benchmarking

All inter-GPU paths were verified before the matrix started.

- `nvidia-smi topo -p2p r/w`: all 12 directed off-diagonal paths reported OK.
- CUDA peer copies: 256 MiB FP16, three warmups, ten measured copies per direction, with sample-value validation.
- All 12 CUDA directions passed at 12.254-12.270 GiB/s.
- NCCL TP4 all-reduce: 256 MiB FP16, three warmups, ten measured iterations, exact expected sum of 10.0 on every rank.
- NCCL bus bandwidth was 10.522-10.525 GiB/s per rank.
- NCCL logs identified every ring edge as `P2P/CUMEM`.

## Fixed inference path

- Transformer precision: FP16.
- Sensitive layers and validated accumulation/residual paths: FP32.
- Tensor parallel size: four.
- Attention: PyTorch SDPA.
- RMSNorm and RoPE: native PyTorch paths.
- Turbo v4: six model evaluations.
- No INT8 weights or kernels.
- No EasyCache, TeaCache, feature cache, or block skipping.
- Official video and audio VAEs run afterward on one V100.
- Seed `20260810`; identical red-paper-boat prompt conditioning for every point.
- 24 fps; H3-compatible frame counts 124, 243, and 362, corresponding to 5.167, 10.125, and 15.083 seconds.

Each point starts a fresh four-rank process, loads the model, then runs the same seed twice without reloading. Trial 2 is the reported hot DiT result.

## Speed matrix

`Fresh two-phase` is fresh load + Trial 1 DiT + fresh single-GPU VAE decode. It excludes live text encoding and orchestration.

| Resolution / frames | Packed sequence | Load | Trial 1 DiT | Hot DiT | Mean / eval | x realtime | VAE + encode | Fresh two-phase |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 864x480 / 124 | 15,463 | 25.11 s | 50.06 s | 49.73 s | 8.25 s | 9.63x | 43.93 s | 119.10 s |
| 960x544 / 124 | 19,348 | 25.38 s | 68.44 s | 68.03 s | 11.27 s | 13.17x | 43.21 s | 137.03 s |
| 1344x768 / 124 | 37,774 | 25.21 s | 191.73 s | 191.63 s | 31.85 s | 37.09x | 65.61 s | 282.56 s |
| 864x480 / 243 | 30,034 | 24.89 s | 132.38 s | 132.12 s | 21.94 s | 13.05x | 68.33 s | 225.59 s |
| 960x544 / 243 | 37,594 | 25.67 s | 190.32 s | 190.25 s | 31.62 s | 18.79x | 69.22 s | 285.22 s |
| 1344x768 / 243 | 73,450 | 25.45 s | 681.98 s | 681.83 s | 113.45 s | 67.34x | 113.98 s | 821.42 s |
| 864x480 / 362 | 44,605 | 25.02 s | 255.29 s | 255.90 s | 42.50 s | 16.97x | 94.17 s | 374.47 s |
| 960x544 / 362 | 55,840 | 25.08 s | 389.84 s | 389.86 s | 64.82 s | 25.85x | 92.89 s | 507.82 s |
| 1344x768 / 362 | 109,126 planned | 25 s-class load | CUDA OOM in first evaluation | n/a | n/a | n/a | n/a | n/a |

Trial 1 and Trial 2 differ by less than about 0.7% at every passing point. In other words, the approximately 25-second model load is reusable, but the DiT calculation itself receives almost no first-run-to-hot-run speedup.

## DiT memory, power, and temperature

Power is the measured total of all four boards and includes pre-existing idle
CUDA contexts, approximately 312 MiB per card.

| Resolution / frames | Peak PyTorch allocated / GPU | Peak NVML / GPU | Four-board average | Peak sampled total | DiT energy | Max DiT temperature |
|---|---:|---:|---:|---:|---:|---:|
| 864x480 / 124 | 18.58 GiB | 20,430 MiB | 766 W | 961 W | 10.59 Wh | 65 C |
| 960x544 / 124 | 19.24 GiB | 21,210 MiB | 738 W | 958 W | 13.94 Wh | 66 C |
| 1344x768 / 124 | 22.34 GiB | 24,912 MiB | 867 W | 1,009 W | 46.13 Wh | 68 C |
| 864x480 / 243 | 21.04 GiB | 23,350 MiB | 842 W | 1,031 W | 30.89 Wh | 67 C |
| 960x544 / 243 | 22.31 GiB | 24,892 MiB | 860 W | 1,009 W | 45.45 Wh | 68 C |
| 1344x768 / 243 | 28.35 GiB | 32,072 MiB | 807 W | 1,017 W | 152.91 Wh | 68 C |
| 864x480 / 362 | 23.49 GiB | 26,290 MiB | 832 W | 1,009 W | 59.17 Wh | 68 C |
| 960x544 / 362 | 25.38 GiB | 28,532 MiB | 910 W | 998 W | 98.57 Wh | 68 C |

The lower average power at 1344x768 / 243 frames, despite its much longer runtime, is consistent with long attention/communication or memory-bound phases in which utilization remains reported as 100% while board power falls well below the 250 W cap.

## VAE resource data

The VAE phase uses one V100 while the other three cards retain only their idle LAN worker contexts. Average power is still the sum of all four boards.

| Resolution / frames | Decode total | Video-VAE peak allocated | Peak NVML on active GPU | Four-board average | Decode energy | Max decode temperature |
|---|---:|---:|---:|---:|---:|---:|
| 864x480 / 124 | 43.93 s | 11,713 MiB | 12,422 MiB | 261 W | 3.34 Wh | 67 C |
| 960x544 / 124 | 43.21 s | 12,172 MiB | 12,882 MiB | 267 W | 3.34 Wh | 69 C |
| 1344x768 / 124 | 65.61 s | 14,350 MiB | 15,082 MiB | 290 W | 5.49 Wh | 70 C |
| 864x480 / 243 | 68.33 s | 13,413 MiB | 14,562 MiB | 291 W | 5.75 Wh | 70 C |
| 960x544 / 243 | 69.22 s | 14,312 MiB | 15,402 MiB | 292 W | 5.78 Wh | 71 C |
| 1344x768 / 243 | 113.98 s | 18,580 MiB | 19,762 MiB | 310 W | 10.03 Wh | 71 C |
| 864x480 / 362 | 94.17 s | 15,112 MiB | 16,302 MiB | 306 W | 8.24 Wh | 71 C |
| 960x544 / 362 | 92.89 s | 16,453 MiB | 17,442 MiB | 306 W | 8.14 Wh | 71 C |

## Model loading details

The loader instrumentation exposes both logical safetensors work and actual retained TP data.

- 14 transformer safetensor files.
- 61.73 GiB of logical checkpoint tensor data touched by each rank.
- 15.97 GiB of resident model tensors per rank after TP4 selection:
  - 15.09 GiB FP16 across 603 tensors;
  - 0.88 GiB FP32 across 35 tensors.
- 0.793 GiB LoRA source per rank; 0.351 GiB of locally relevant sharded factors.
- Fresh load range 24.89-25.67 seconds; mean 25.23 seconds.
- Mean checkpoint/model phase 24.08 seconds; mean LoRA merge 1.17 seconds.
- Model-load CUDA peak 16.58 GiB per rank.
- Process high-water memory about 6.54 GiB per rank; measured CT cgroup peak 6.70-7.00 GiB.
- Four-board load power averaged about 192 W and used about 1.35 Wh.
- Warm page-cache NFS receive traffic was below 0.001-0.215 GiB per load. This is not a cold-disk benchmark.

The current loader calls `get_tensor` for every source tensor before selecting the local TP shard. A safetensors slice-based loader is a plausible startup/CPU-memory optimization, but it will not reduce the hot DiT time.

## The 1344x768 / 15-second OOM

This point did not time out and the container was not killed by the host. It failed in the first transformer evaluation at the FFN scale restoration:

`return projected.float() * 256.0`

The first reported rank had:

- 31.73 GiB physical CUDA capacity;
- 30.70 GiB process memory in use;
- 29.94 GiB allocated by PyTorch;
- 288.71 MiB reserved but unallocated;
- 742.25 MiB free;
- a failed 2.19 GiB allocation request.

The immediate memory optimization target is therefore not speculative: it is the full-sequence FP32 FFN/residual temporary. A row-chunked or numerically equivalent in-place accumulation should be tested before changing model precision.

## Determinism and output validation

- All eight passing cases produced finite latents in both runs.
- Trial 1 and Trial 2 `video_rows` and `audio_rows` were exactly equal element by element, with zero maximum absolute error.
- The safetensors file hashes can differ because metadata/header order can differ; file SHA is not used as the determinism gate.
- Every passing Trial 2 decoded with the official H3 video/audio VAEs.
- Every MP4 passed expected frame count, geometry, 24 fps timing, H.264 video, AAC stereo 32 kHz audio, and non-silent audio checks.
- Combined and per-case contact sheets were visually inspected. The red paper boat, rain, reflections, and camera progression remained coherent; no black/corrupt frames or obvious subject break was observed.

## Thermal and closeout

A host-side fan-control guard remained enabled and active with zero restarts.
Under load it reached 75% commanded PWM while the hottest active VAE GPU
reached 71 C. No GPU reached the guard's emergency threshold or the benchmark's
80 C stop threshold.

After the matrix:

- unrelated worker services remained healthy and idle;
- all four V100s were idle at 312 MiB, 0% utilization, 44-47 C;
- corrected and uncorrected volatile ECC counts were zero on all four cards;
- no TP4/decode benchmark container remained running.

## What I would optimize next

1. Replace the full-sequence FP32 FFN scale-restoration temporary with a numerically equivalent row-chunked or in-place path, then retry 1344x768 / 362 frames.
2. Benchmark attention using the actual long shapes. The tested SM70 TileLang kernel was 7.6% slower than PyTorch SDPA at the 864x480 / 124-frame shape, so that specific kernel is not useful.
3. Add a safetensors slice loader and a resident service. This targets approximately 25 seconds of reload work, not DiT compute.
4. Integrate live text conditioning and keep the official VAE as a separate phase before claiming a production end-to-end API time.
5. For more than four V100s, test TP4+SP2. LightX2V's TP and sequence-parallel dimensions multiply, so that configuration needs eight ranks. On only four 32 GB cards, TP2+SP2 is unlikely to fit in native FP16: the measured replicated/sharded weight split predicts roughly 31.2 GiB of resident tensors per rank before activations.

## Public evidence boundary

This report contains sanitized aggregate measurements only. Raw logs, media,
conditioning data, latent tensors, model artifacts, internal service details,
and machine-specific paths are intentionally not included in this branch.
