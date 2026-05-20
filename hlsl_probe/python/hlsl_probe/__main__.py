from __future__ import annotations

import argparse
import json

from .bindings import probe_caps, self_test


def main() -> None:
    parser = argparse.ArgumentParser(description="HLSL probe Python wrapper")
    parser.add_argument("--probe", action="store_true", help="Print D3D12 capability JSON")
    parser.add_argument("--self-test", action="store_true", help="Compile and dispatch the tiny self-test shader")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    print(json.dumps(probe_caps(), indent=2))


if __name__ == "__main__":
    main()
