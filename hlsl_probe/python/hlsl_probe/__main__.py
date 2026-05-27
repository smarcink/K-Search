from __future__ import annotations

import argparse
import json

from .bindings import linalg_caps_summary, linalg_fp16_test, linalg_test, probe_caps, self_test


def main() -> None:
    parser = argparse.ArgumentParser(description="HLSL probe Python wrapper")
    parser.add_argument("--probe", action="store_true", help="Print D3D12 capability JSON")
    parser.add_argument("--self-test", action="store_true", help="Compile and dispatch the tiny self-test shader")
    parser.add_argument("--linalg-caps", action="store_true", help="Print a compact D3D12 linear algebra capability summary")
    parser.add_argument("--linalg-test", action="store_true", help="Compile and dispatch a tiny dx/linalg.h shader")
    parser.add_argument("--linalg-fp16-test", action="store_true", help="Compile and dispatch a tiny FP16 dx/linalg.h shader")
    parser.add_argument("--target", default="cs_6_10", help="Shader model target for --linalg-test/--linalg-fp16-test")
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
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    print(json.dumps(probe_caps(), indent=2))


if __name__ == "__main__":
    main()
