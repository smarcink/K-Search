#!/usr/bin/env python3
"""
Swin block eager vs torch.compile vs fused Triton kernel benchmark on Intel XPU.

Measures latency, throughput, and correctness for:
  1. Eager (no compilation)
  2. torch.compile(backend="inductor")                  — default
  3. torch.compile(backend="inductor", mode="reduce-overhead")
  4. torch.compile(backend="inductor", mode="max-autotune")
  5. Fused Triton kernel (entire Swin block in one kernel per window)

Also includes roofline analysis against Arc Pro B70 peak specs.
"""

import os
import sys
import time
import subprocess
import statistics

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
DEVICE = "xpu" if torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Swin model from parent dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swin_block_fp16_simplified import Model, get_init_inputs
# Fused Triton kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "triton_opus46_copy"))
from model_new import ModelNew

WARMUP = 10
REPEAT = 50


def make_input(B, H, W, C, device=DEVICE):
    return torch.randn(B, H, W, C, dtype=torch.float16, device=device)


def bench(fn, inp, warmup=WARMUP, repeat=REPEAT, label=""):
    """Benchmark a function, return median/min/max/std in ms."""
    # warmup
    for _ in range(warmup):
        fn(inp)
    torch.xpu.synchronize() if DEVICE == "xpu" else torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        if DEVICE == "xpu":
            torch.xpu.synchronize()
        else:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(inp)
        if DEVICE == "xpu":
            torch.xpu.synchronize()
        else:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    med = statistics.median(times)
    mn = min(times)
    mx = max(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return {"median": med, "min": mn, "max": mx, "std": std, "times": times}


def check_correctness(ref_out, test_out, name):
    """Check numerical correctness."""
    diff = (ref_out - test_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel = (diff / (ref_out.abs() + 1e-8)).max().item()
    return {"max_abs": max_diff, "mean_abs": mean_diff, "max_rel": rel}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Swin block: eager vs torch.compile on Intel XPU")
    print("=" * 80)
    print(f"Device:   {DEVICE} — {torch.xpu.get_device_name(0) if DEVICE == 'xpu' else 'N/A'}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"Warmup:   {WARMUP}   Repeat: {REPEAT}")

    # Check ocloc/IGC
    try:
        out = subprocess.check_output(
            "dpkg -l | grep -E 'intel-ocloc |libigc2 ' | awk '{printf \"  %-22s %s\\n\", $2, $3}'",
            shell=True, text=True
        )
        print(f"Packages:\n{out.strip()}")
    except Exception:
        pass

    spv_env = os.environ.get("TORCHINDUCTOR_XPU_KERNEL_FORMAT", "(default: zebin)")
    print(f"TORCHINDUCTOR_XPU_KERNEL_FORMAT: {spv_env}")
    print()

    init_inputs = get_init_inputs()  # [32, 1, [4,4], [0,0], 4]

    # -------------------------------------------------------------------
    # Part 1: Single-size deep analysis (the K-Search target size)
    # -------------------------------------------------------------------
    B, H, W, C = 1, 720, 1280, 32
    print(f"{'='*80}")
    print(f"PART 1 — Deep analysis: input ({B}, {H}, {W}, {C}) fp16")
    print(f"{'='*80}")

    model = Model(*init_inputs).to(DEVICE).eval()
    inp = make_input(B, H, W, C)

    # Eager
    with torch.no_grad():
        ref_out = model(inp)
        eager_res = bench(lambda x: model(x), inp, label="eager")
    print(f"\n  Eager:              {eager_res['median']:7.2f} ms  (min={eager_res['min']:.2f}, max={eager_res['max']:.2f}, std={eager_res['std']:.2f})")

    # Compile variants
    compile_modes = [
        ("compile-default",        {}),
        ("compile-reduce-overhead", {"mode": "reduce-overhead"}),
        ("compile-max-autotune",    {"mode": "max-autotune"}),
    ]

    compile_results = {}
    for name, kwargs in compile_modes:
        # Fresh model copy to avoid caching interactions
        model_c = Model(*init_inputs).to(DEVICE).eval()
        model_c.load_state_dict(model.state_dict())
        compiled = torch.compile(model_c, backend="inductor", **kwargs)

        try:
            with torch.no_grad():
                # Compilation warmup — first call triggers compile
                compiled(inp)
                if DEVICE == "xpu":
                    torch.xpu.synchronize()

                test_out = compiled(inp)
                corr = check_correctness(ref_out, test_out, name)
                res = bench(lambda x: compiled(x), inp, label=name)

            speedup = eager_res["median"] / res["median"]
            compile_results[name] = {"bench": res, "corr": corr, "speedup": speedup}
            print(f"  {name:25s} {res['median']:7.2f} ms  (min={res['min']:.2f}, max={res['max']:.2f}, std={res['std']:.2f})  "
                  f"speedup={speedup:.2f}x  max_diff={corr['max_abs']:.2e}")
        except Exception as e:
            print(f"  {name:25s} FAILED: {e}")
            compile_results[name] = None

    # -------------------------------------------------------------------
    # Part 1b: Fused Triton kernel
    # -------------------------------------------------------------------
    triton_model = ModelNew(*init_inputs).to(DEVICE).eval()
    triton_model.load_state_dict(model.state_dict(), strict=False)
    # Trigger cache_splits
    with torch.no_grad():
        triton_model(inp)

    with torch.no_grad():
        triton_out = triton_model(inp)
        triton_corr = check_correctness(ref_out, triton_out, "triton-fused")
        triton_res = bench(lambda x: triton_model(x), inp, label="triton-fused")
    triton_speedup = eager_res["median"] / triton_res["median"]
    print(f"  {'triton-fused':25s} {triton_res['median']:7.2f} ms  (min={triton_res['min']:.2f}, max={triton_res['max']:.2f}, std={triton_res['std']:.2f})  "
          f"speedup={triton_speedup:.2f}x  max_diff={triton_corr['max_abs']:.2e}")

    # -------------------------------------------------------------------
    # Part 2: Sweep across input sizes (eager vs compile vs triton)
    # -------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"PART 2 — Size sweep: eager vs compile-default vs triton-fused")
    print(f"{'='*80}")

    sizes = [
        (1, 180, 320, 32),    # small: 1/16 of target
        (1, 360, 640, 32),    # medium: 1/4 of target
        (1, 720, 1280, 32),   # target (repeat)
        (4, 720, 1280, 32),   # batch=4
    ]

    print(f"\n  {'Size':>25s}  {'Eager(ms)':>10s}  {'Compile(ms)':>12s}  {'Triton(ms)':>11s}  {'E→C':>6s}  {'E→T':>6s}  {'max_diff':>10s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*12}  {'-'*11}  {'-'*6}  {'-'*6}  {'-'*10}")

    for B, H, W, C in sizes:
        label = f"({B},{H},{W},{C})"
        inp_s = make_input(B, H, W, C)

        # Eager
        model_e = Model(*init_inputs).to(DEVICE).eval()
        with torch.no_grad():
            ref = model_e(inp_s)
            e_res = bench(lambda x: model_e(x), inp_s, warmup=5, repeat=30)

        # Compile
        model_c = Model(*init_inputs).to(DEVICE).eval()
        model_c.load_state_dict(model_e.state_dict())
        compiled = torch.compile(model_c, backend="inductor")
        try:
            with torch.no_grad():
                compiled(inp_s)
                if DEVICE == "xpu":
                    torch.xpu.synchronize()
                c_res = bench(lambda x: compiled(x), inp_s, warmup=5, repeat=30)
            c_sp = e_res["median"] / c_res["median"]
            c_str = f"{c_res['median']:12.2f}"
            c_sp_str = f"{c_sp:5.2f}x"
        except Exception:
            c_str = f"{'FAILED':>12s}"
            c_sp_str = "  N/A"

        # Triton fused
        model_t = ModelNew(*init_inputs).to(DEVICE).eval()
        model_t.load_state_dict(model_e.state_dict(), strict=False)
        try:
            with torch.no_grad():
                model_t(inp_s)
                t_out = model_t(inp_s)
                corr = check_correctness(ref, t_out, label)
                t_res = bench(lambda x: model_t(x), inp_s, warmup=5, repeat=30)
            t_sp = e_res["median"] / t_res["median"]
            t_str = f"{t_res['median']:11.2f}"
            t_sp_str = f"{t_sp:5.2f}x"
            corr_str = f"{corr['max_abs']:10.2e}"
        except Exception as e:
            t_str = f"{'FAILED':>11s}"
            t_sp_str = "  N/A"
            corr_str = str(e)[:10]

        print(f"  {label:>25s}  {e_res['median']:10.2f}  {c_str}  {t_str}  {c_sp_str}  {t_sp_str}  {corr_str}")

        del model_e, model_t, inp_s
        try:
            del model_c, compiled
        except Exception:
            pass
        if DEVICE == "xpu":
            torch.xpu.empty_cache()

    # -------------------------------------------------------------------
    # Part 3: Per-layer breakdown (where does compile help?)
    # -------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"PART 3 — Per-layer breakdown: which ops benefit from compile?")
    print(f"{'='*80}")

    model_p = Model(*init_inputs).to(DEVICE).eval()
    inp_p = make_input(1, 720, 1280, 32)

    # Break down into sub-operations
    ops = {}
    with torch.no_grad():
        # 1. LayerNorm
        def ln1(x):
            return model_p.norm1(x)
        ops["norm1 (LayerNorm)"] = (ln1, inp_p)

        x_ln1 = ln1(inp_p)
        # 2. Attention
        def attn(x):
            return model_p._attn(x)
        ops["_attn (W-MSA)"] = (attn, x_ln1)

        x_attn = attn(x_ln1)
        x_res1 = inp_p + x_attn

        # 3. LayerNorm 2
        def ln2(x):
            return model_p.norm2(x)
        ops["norm2 (LayerNorm)"] = (ln2, x_res1)

        x_ln2 = ln2(x_res1)
        # 4. MLP (fc1 + GELU + fc2)
        def mlp(x):
            return model_p.fc2(model_p.act(model_p.fc1(x)))
        ops["MLP (fc1+GELU+fc2)"] = (mlp, x_ln2)

    print(f"\n  {'Operation':>25s}  {'Eager(ms)':>10s}  {'Compile(ms)':>12s}  {'Speedup':>8s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*12}  {'-'*8}")

    for name, (fn, fn_inp) in ops.items():
        with torch.no_grad():
            e_res = bench(fn, fn_inp, warmup=5, repeat=30)
            try:
                fn_c = torch.compile(fn, backend="inductor")
                fn_c(fn_inp)
                if DEVICE == "xpu":
                    torch.xpu.synchronize()
                c_res = bench(fn_c, fn_inp, warmup=5, repeat=30)
                sp = e_res["median"] / c_res["median"]
                print(f"  {name:>25s}  {e_res['median']:10.2f}  {c_res['median']:12.2f}  {sp:7.2f}x")
            except Exception as e:
                print(f"  {name:>25s}  {e_res['median']:10.2f}  {'FAILED':>12s}  {str(e)[:40]}")

    # -------------------------------------------------------------------
    # Part 4: Roofline analysis
    # -------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"PART 4 — Roofline analysis (Arc Pro B70: 608 GB/s BW, 160 TFLOPS FP16)")
    print(f"{'='*80}")

    # Hardware specs
    PEAK_BW_GBs = 608.0
    PEAK_FLOPS = 160e12  # FP16 DPAS

    # Workload: (1, 720, 1280, 32) fp16, window=4x4
    num_windows = 57600  # = 1 * 180 * 320
    N, Cdim, HIDDEN = 16, 32, 128

    # FLOPs per window (multiply-accumulate = 2 ops)
    flops_qkv   = 3 * 2 * N * Cdim * Cdim        # 3 projections: 98304
    flops_qkt   = 2 * N * Cdim * N                # Q @ K^T: 16384
    flops_av    = 2 * N * N * Cdim                 # attn @ V: 16384
    flops_proj  = 2 * N * Cdim * Cdim              # proj: 32768
    flops_fc1   = 2 * N * Cdim * HIDDEN            # fc1: 131072
    flops_fc2   = 2 * N * HIDDEN * Cdim            # fc2: 131072
    # Elementwise (LN, GELU, softmax, residuals) — approximate
    flops_elem  = N * (2*Cdim + 2*Cdim + N*3 + HIDDEN*8 + 2*Cdim)  # ~3000
    flops_per_window = flops_qkv + flops_qkt + flops_av + flops_proj + flops_fc1 + flops_fc2 + flops_elem
    total_flops = num_windows * flops_per_window

    # Memory traffic
    input_bytes = 1 * 720 * 1280 * 32 * 2          # input tensor
    output_bytes = input_bytes                      # output tensor
    # Weights: loaded once from DRAM, then cached in L2 for remaining windows
    wt_qkv   = 3 * Cdim * Cdim * 2                 # 6144 bytes
    wt_proj   = Cdim * Cdim * 2                     # 2048
    wt_fc1    = HIDDEN * Cdim * 2                   # 8192
    wt_fc2    = Cdim * HIDDEN * 2                   # 8192
    wt_bias   = (3*Cdim + Cdim + Cdim + HIDDEN + Cdim) * 2  # biases
    wt_ln     = 4 * Cdim * 2                        # LN w/b x2
    wt_rpe    = N * N * 4                           # rpe_flat (fp32)
    total_weight_bytes = wt_qkv + wt_proj + wt_fc1 + wt_fc2 + wt_bias + wt_ln + wt_rpe

    # Best case: weights cached in L2, traffic = input + output
    mem_cached = input_bytes + output_bytes
    # Worst case: every window re-reads weights from DRAM
    mem_uncached = input_bytes + output_bytes + num_windows * total_weight_bytes

    ai_cached = total_flops / mem_cached
    ai_uncached = total_flops / mem_uncached
    ridge_point = PEAK_FLOPS / (PEAK_BW_GBs * 1e9)

    # Speed-of-light times
    t_compute = total_flops / PEAK_FLOPS * 1000     # ms
    t_mem_cached = mem_cached / (PEAK_BW_GBs * 1e9) * 1000
    t_mem_uncached = mem_uncached / (PEAK_BW_GBs * 1e9) * 1000
    sol_cached = max(t_compute, t_mem_cached)
    sol_uncached = max(t_compute, t_mem_uncached)

    print(f"""
  Workload: ({1}, {720}, {1280}, {32}) fp16, window=4x4, {num_windows} windows
  
  Compute:
    Per window:      {flops_per_window:,} FLOPs (matmul: {flops_per_window - flops_elem:,} + elem: {flops_elem:,})
    Total:           {total_flops/1e9:.2f} GFLOPS
    Peak time:       {t_compute:.3f} ms  (at 160 TFLOPS FP16 DPAS)

  Memory:
    Input + Output:  {mem_cached/1e6:.1f} MB
    Weights (total): {total_weight_bytes/1024:.1f} KB  (small → fits in L2)
    Per-window wt:   {total_weight_bytes:,} bytes × {num_windows} windows = {mem_uncached/1e6:.1f} MB (if uncached)

  Arithmetic Intensity:
    If weights cached:   {ai_cached:.1f} FLOP/byte  → {'COMPUTE-bound' if ai_cached > ridge_point else 'MEMORY-bound'}
    If weights uncached: {ai_uncached:.1f} FLOP/byte → {'COMPUTE-bound' if ai_uncached > ridge_point else 'MEMORY-bound'}
    Ridge point:         {ridge_point:.1f} FLOP/byte

  Speed-of-Light:
    Best case (wt cached):   {sol_cached:.3f} ms  (bottleneck: {'compute' if t_compute > t_mem_cached else 'memory'})
    Worst case (wt uncached): {sol_uncached:.3f} ms  (bottleneck: {'compute' if t_compute > t_mem_uncached else 'memory'})""")

    # Actual achieved
    actual_ms = triton_res["median"]
    eff_cached = sol_cached / actual_ms * 100
    eff_uncached = sol_uncached / actual_ms * 100
    achieved_tflops = total_flops / (actual_ms / 1000) / 1e12
    achieved_bw = mem_cached / (actual_ms / 1000) / 1e9

    print(f"""
  Achieved (triton-fused):
    Latency:         {actual_ms:.2f} ms
    Throughput:      {achieved_tflops:.2f} TFLOPS  ({achieved_tflops/160*100:.1f}% of peak)
    Eff. bandwidth:  {achieved_bw:.1f} GB/s  ({achieved_bw/608*100:.1f}% of peak)
    SOL efficiency:  {eff_cached:.1f}% (wt-cached) / {eff_uncached:.1f}% (wt-uncached)

  NOTE: matmuls are tiny (16×32 × 32×32). DPAS systolic arrays (designed for
  large tiles) are heavily underutilized. Effective peak for these sizes is
  likely 10-30% of theoretical 160 TFLOPS. The kernel is also launching
  57,600 thread groups — occupancy and scheduling overhead matter.""")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    eager_ms = eager_res["median"]
    print(f"  Eager (median):     {eager_ms:.2f} ms")
    for name, data in compile_results.items():
        if data:
            print(f"  {name:25s} {data['bench']['median']:.2f} ms  ({data['speedup']:.2f}x vs eager)")
        else:
            print(f"  {name:25s} FAILED")
    print(f"  {'triton-fused':25s} {triton_res['median']:.2f} ms  ({triton_speedup:.2f}x vs eager)  max_diff={triton_corr['max_abs']:.2e}")

    best_compile = None
    for name, data in compile_results.items():
        if data and (best_compile is None or data["bench"]["median"] < best_compile[1]):
            best_compile = (name, data["bench"]["median"], data["speedup"])

    print(f"\n  Best compile:  {best_compile[0]}  ({best_compile[2]:.2f}x)" if best_compile else "")
    print(f"  Triton fused:  {triton_speedup:.2f}x vs eager, "
          f"{(best_compile[1] / triton_res['median']):.2f}x vs best-compile" if best_compile else "")
    print(f"  SOL estimate:  {sol_cached:.3f}–{sol_uncached:.3f} ms "
          f"(actual {actual_ms:.2f} ms = {eff_cached:.0f}–{eff_uncached:.0f}% of SOL)")


if __name__ == "__main__":
    main()
