from __future__ import annotations

import argparse
import json

from .bindings import linalg_caps_summary, linalg_fp16_shape_sweep, linalg_fp16_test, linalg_test, probe_caps, self_test


def _parse_int_csv(text: str) -> tuple[int, ...]:
    values = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return tuple(values)


def _parse_str_csv(text: str) -> tuple[str, ...]:
    values = tuple(part.strip().lower() for part in str(text or "").split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="HLSL probe Python wrapper")
    parser.add_argument("--probe", action="store_true", help="Print D3D12 capability JSON")
    parser.add_argument("--self-test", action="store_true", help="Compile and dispatch the tiny self-test shader")
    parser.add_argument("--linalg-caps", action="store_true", help="Print a compact D3D12 linear algebra capability summary")
    parser.add_argument("--linalg-test", action="store_true", help="Compile and dispatch a tiny dx/linalg.h shader")
    parser.add_argument("--linalg-fp16-test", action="store_true", help="Compile and dispatch a tiny FP16 dx/linalg.h shader")
    parser.add_argument(
        "--linalg-fp16-shape-sweep",
        action="store_true",
        help="Compile and benchmark FP16 MatrixScope::Thread matvec shapes",
    )
    parser.add_argument("--target", default="cs_6_10", help="Shader model target for --linalg-test/--linalg-fp16-test")
    parser.add_argument("--m-values", type=_parse_int_csv, default=(2, 4, 8, 16), help="Comma-separated M values")
    parser.add_argument("--k-values", type=_parse_int_csv, default=(4, 8, 16, 32, 64, 128), help="Comma-separated K values")
    parser.add_argument("--wave-sizes", type=_parse_int_csv, default=(0,), help="Comma-separated WaveSize values; 0 omits the attribute")
    parser.add_argument(
        "--shape-accumulators",
        type=_parse_str_csv,
        default=("fp32",),
        help="Comma-separated accumulator/result modes: fp16 and/or fp32",
    )
    parser.add_argument(
        "--shape-ops",
        type=_parse_str_csv,
        default=("multiply",),
        help="Comma-separated LinAlg operation modes: multiply and/or multiplyadd",
    )
    parser.add_argument("--shape-reps", type=int, default=64, help="Matvec repetitions per shader invocation")
    parser.add_argument("--shape-dispatch-groups", type=int, default=64, help="Dispatch X groups per shape")
    parser.add_argument("--shape-warmup", type=int, default=1, help="Warmup dispatches per shape")
    parser.add_argument("--shape-iters", type=int, default=3, help="Measured dispatches per shape")
    args = parser.parse_args()

    if args.linalg_test:
        print(json.dumps(linalg_test(target=args.target), indent=2))
        return
    if args.linalg_caps:
        print(json.dumps(linalg_caps_summary(), indent=2))
        return
    if args.linalg_fp16_test:
        print(json.dumps(linalg_fp16_test(target=args.target), indent=2))
        return
    if args.linalg_fp16_shape_sweep:
        print(
            json.dumps(
                linalg_fp16_shape_sweep(
                    target=args.target,
                    m_values=args.m_values,
                    k_values=args.k_values,
                    wave_sizes=args.wave_sizes,
                    accumulators=args.shape_accumulators,
                    operations=args.shape_ops,
                    reps=args.shape_reps,
                    dispatch_groups=args.shape_dispatch_groups,
                    warmup=args.shape_warmup,
                    iterations=args.shape_iters,
                ),
                indent=2,
            )
        )
        return
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    print(json.dumps(probe_caps(), indent=2))


if __name__ == "__main__":
    main()
