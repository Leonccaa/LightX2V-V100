#!/usr/bin/env python3
"""Benchmark TileLang attention at the exact MiniMax-H3 TP4 shape on V100.

Kernel structure is adapted from TileLang v0.1.9's official
``examples/flash_attention/example_mha_fwd_bhsd.py``.  This benchmark is
deliberately attention-only; it does not claim an end-to-end gain unless the
measured kernel can later be integrated into LightX2V.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F


@tilelang.jit(
    out_idx=[3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def flash_attention(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    dim: int,
    block_m: int = 64,
    block_n: int = 64,
    num_stages: int = 0,
    threads: int = 128,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        q: T.Tensor(q_shape, dtype),
        k: T.Tensor(kv_shape, dtype),
        v: T.Tensor(kv_shape, dtype),
        output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_m), heads, batch, threads=threads) as (bx, by, bz):
            q_shared = T.alloc_shared([block_m, dim], dtype)
            k_shared = T.alloc_shared([block_n, dim], dtype)
            v_shared = T.alloc_shared([block_n, dim], dtype)
            o_shared = T.alloc_shared([block_m, dim], dtype)
            p_shared = T.alloc_shared([block_m, block_n], dtype)
            acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
            acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_m], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
            scores_scale = T.alloc_fragment([block_m], accum_dtype)
            scores_sum = T.alloc_fragment([block_m], accum_dtype)
            logsum = T.alloc_fragment([block_m], accum_dtype)

            T.copy(q[bz, by, bx * block_m : (bx + 1) * block_m, :], q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for block_k in T.Pipelined(T.ceildiv(seq_kv, block_n), num_stages=num_stages):
                T.copy(k[bz, by, block_k * block_n : (block_k + 1) * block_n, :], k_shared)
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.if_then_else(
                        block_k * block_n + j >= seq_kv,
                        -T.infinity(accum_dtype),
                        0,
                    )
                T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_m):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_m):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                # TileLang 0.1.9's SM70 layout inference cannot reconcile the
                # FP32 score fragment with the FP16 fragment layout expected by
                # the second MMA.  A shared-memory handoff is explicit, valid
                # on Volta, and keeps this probe on Tensor Cores.
                T.copy(acc_s, p_shared)

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(v[bz, by, block_k * block_n : (block_k + 1) * block_n, :], v_shared)
                T.gemm(p_shared, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_m, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, o_shared)
            T.copy(o_shared, output[bz, by, bx * block_m : (bx + 1) * block_m, :])

    return main


def _bench(fn, warmup: int, iterations: int) -> tuple[list[float], torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    assert output is not None
    return timings, output


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--sequence", type=int, default=15463)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=0)
    parser.add_argument("--threads", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(20260810)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    shape = (args.batch, args.heads, args.sequence, args.dim)
    q = torch.randn(shape, device=device, dtype=torch.float16)
    k = torch.randn(shape, device=device, dtype=torch.float16)
    v = torch.randn(shape, device=device, dtype=torch.float16)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sdpa_timings, sdpa_output = _bench(
        lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False),
        args.warmup,
        args.iterations,
    )
    sdpa_peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)

    compile_started = time.perf_counter()
    kernel = flash_attention(
        args.batch,
        args.heads,
        args.sequence,
        args.sequence,
        args.dim,
        block_m=args.block_m,
        block_n=args.block_n,
        num_stages=args.num_stages,
        threads=args.threads,
    )
    compile_seconds = time.perf_counter() - compile_started
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tilelang_timings, tilelang_output = _bench(
        lambda: kernel(q, k, v),
        args.warmup,
        args.iterations,
    )
    tilelang_peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)

    delta = tilelang_output.float() - sdpa_output.float()
    relative_rmse = float(delta.square().mean().sqrt() / sdpa_output.float().square().mean().sqrt())
    max_abs_delta = float(delta.abs().max())
    correlation = float(torch.corrcoef(torch.stack((tilelang_output.float().flatten(), sdpa_output.float().flatten())))[0, 1])
    sdpa_median = statistics.median(sdpa_timings)
    tilelang_median = statistics.median(tilelang_timings)
    total_flops = 4.0 * args.batch * args.heads * args.sequence * args.sequence * args.dim

    result = {
        "status": "pass" if math.isfinite(relative_rmse) and relative_rmse < 0.02 else "fail",
        "shape_source": "MiniMax-H3 864x480/124f TP4 packed attention: B=1,H=56/4,S=64+414+14985,D=128",
        "shape": {"batch": args.batch, "heads": args.heads, "sequence": args.sequence, "head_dim": args.dim},
        "device": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "tilelang": tilelang.__version__,
        },
        "tilelang_config": {
            "block_m": args.block_m,
            "block_n": args.block_n,
            "num_stages": args.num_stages,
            "threads": args.threads,
            "compile_seconds": compile_seconds,
            "sm70_mma_in_source": "mma_sync_sm70" in kernel.get_kernel_source(),
        },
        "torch_sdpa": {
            **_summary(sdpa_timings),
            "effective_tflops": total_flops / sdpa_median * 1e-9,
            "peak_allocated_mib": sdpa_peak_mib,
        },
        "tilelang": {
            **_summary(tilelang_timings),
            "effective_tflops": total_flops / tilelang_median * 1e-9,
            "peak_allocated_mib": tilelang_peak_mib,
        },
        "tilelang_over_sdpa_speedup": sdpa_median / tilelang_median,
        "output_comparison": {
            "relative_rmse": relative_rmse,
            "max_abs_delta": max_abs_delta,
            "correlation": correlation,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
