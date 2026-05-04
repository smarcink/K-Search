"""
Compare multiple conv_pooling_fp16 implementations on the same input.

Each implementation must be a Python file that exposes:
  * `Model` (or `ModelNew`) — an nn.Module with the ConvBlock forward
  * `get_inputs()` — list of input tensors
  * `get_init_inputs()` — list of constructor args for the module

Usage:
    python compare_conv_pooling_fp16.py                            # default set
    python compare_conv_pooling_fp16.py file_a.py file_b.py ...    # custom set

The script:
  1. Loads each module dynamically.
  2. Instantiates it on CUDA, copies weights from the FIRST module so all
     implementations are numerically comparable.
  3. Runs warm-up + timed iterations with cuda events.
  4. Reports per-implementation latency, throughput, max-abs / max-rel
     error vs the first (reference) implementation, and pass/fail.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_FILES = [
    "conv_pooling_fp16/conv_block_fp16.py",
    "conv_pooling_fp16/conv_block_fp16_opus47_triton_r12.py",
]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(mod, device: str, dtype: torch.dtype = torch.float16) -> nn.Module:
    cls = getattr(mod, "Model", None) or getattr(mod, "ModelNew", None)
    if cls is None:
        raise AttributeError(f"{mod.__name__}: no Model / ModelNew class")
    init_args = mod.get_init_inputs()
    return cls(*init_args).to(dtype=dtype, device=device).eval()


def copy_weights(dst: nn.Module, src: nn.Module, dst_mod=None) -> None:
    """Copy all parameters + buffers from src into dst (matched by name).

    If `dst_mod` defines `state_dict_remap(src_state) -> dict`, that hook is
    applied first so an implementation with a different submodule layout
    can still receive the reference's weights.
    """
    src_state = src.state_dict()
    remap = getattr(dst_mod, "state_dict_remap", None) if dst_mod is not None else None
    if callable(remap):
        src_state = remap(src_state)
    missing, unexpected = dst.load_state_dict(src_state, strict=False)
    if missing:
        print(f"  [warn] missing keys when copying weights: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys when copying weights: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


@torch.inference_mode()
def benchmark(model: nn.Module, inputs: list[torch.Tensor], warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    # Warm-up
    for _ in range(warmup):
        out = model(*inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = model(*inputs)
    end.record()
    torch.cuda.synchronize()
    ms_per_iter = start.elapsed_time(end) / iters
    return ms_per_iter, out


def diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a32 = a.detach().float().flatten()
    b32 = b.detach().float().flatten()
    abs_err = (a32 - b32).abs()
    max_abs = abs_err.max().item()
    mse = (abs_err.pow(2)).mean().item()
    denom = a32.norm() * b32.norm()
    cos_sim = (torch.dot(a32, b32) / denom.clamp_min(1e-12)).item() if denom > 0 else 1.0
    return max_abs, mse, cos_sim


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", default=DEFAULT_FILES, help="Implementation files to compare")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=5e-3, help="Absolute tolerance for pass/fail")
    p.add_argument("--no-copy-weights", action="store_true",
                   help="Do NOT copy weights from the reference; each model uses its own random init")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip the torch.compile variants")
    p.add_argument("--compile-mode", default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                   help="torch.compile mode for compiled variants")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is required.")
    device = "cuda"
    torch.manual_seed(args.seed)

    paths = [Path(f).resolve() for f in args.files]
    for path in paths:
        if not path.exists():
            sys.exit(f"file not found: {path}")

    print(f"Comparing {len(paths)} implementations on {torch.cuda.get_device_name(0)}")
    print(f"warmup={args.warmup}  iters={args.iters}  copy-weights={not args.no_copy_weights}\n")

    # Load + instantiate every implementation.
    mods = [load_module(p) for p in paths]
    models = [build_model(m, device) for m in mods]

    # Inputs come from the first module (they should be functionally identical).
    inputs = [t.to(device) for t in mods[0].get_inputs()]
    print(f"input shapes: {[tuple(t.shape) for t in inputs]}  dtype={inputs[0].dtype}")

    # Make weights consistent so output comparison is meaningful.
    if not args.no_copy_weights:
        ref = models[0]
        for mod, m in zip(mods[1:], models[1:]):
            copy_weights(m, ref, dst_mod=mod)

    # Pre-compute reference output once.
    with torch.inference_mode():
        ref_out = models[0](*inputs)
    torch.cuda.synchronize()

    # Build the (label, model) list: eager versions first, then compiled clones.
    entries: list[tuple[str, nn.Module]] = [(p.name, m) for p, m in zip(paths, models)]
    if not args.no_compile:
        for path, model in zip(paths, models):
            try:
                compiled = torch.compile(model, mode=args.compile_mode, fullgraph=False, dynamic=False)
                entries.append((f"{path.name} [compile:{args.compile_mode}]", compiled))
            except Exception as e:
                print(f"  [warn] could not torch.compile {path.name}: {e}")

    rows: list[tuple[str, float, float, float, float, str]] = []
    for label, model in entries:
        try:
            ms, out = benchmark(model, inputs, args.warmup, args.iters)
            max_abs, mse, cos_sim = diff(out, ref_out)
            ok = max_abs <= args.atol
            status = "PASS" if ok else "FAIL"
            rows.append((label, ms, max_abs, mse, cos_sim, status))
        except Exception as e:
            rows.append((label, float("nan"), float("nan"), float("nan"), float("nan"), f"ERROR: {e}"))

    # Pretty print.
    base_ms = rows[0][1]
    name_w = max(len(r[0]) for r in rows)
    print()
    print(f"{'impl'.ljust(name_w)}  {'ms/iter':>10}  {'speedup':>8}  {'max_abs':>11}  {'mse':>11}  {'cos_sim':>10}  status")
    print("-" * (name_w + 70))
    for name, ms, ma, mse, cs, status in rows:
        speed = f"{base_ms / ms:.2f}x" if ms == ms and ms > 0 else "-"
        print(f"{name.ljust(name_w)}  {ms:>10.3f}  {speed:>8}  {ma:>11.3e}  {mse:>11.3e}  {cs:>10.6f}  {status}")

    print(f"\n(reference for diff/speedup = '{rows[0][0]}')")


if __name__ == "__main__":
    main()
