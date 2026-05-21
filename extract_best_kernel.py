"""Extract the fastest kernel from a K-Search artifacts folder.

Scans solutions/ and eval/ dirs (or the solution_db.jsonl) to find the
solution with the best speedup. Prints its details and optionally writes
the kernel source files to an output directory.

Usage:
    python extract_best_kernel.py elementwise_add_fp16/ksearch-sonnet46_cuda
    python extract_best_kernel.py elementwise_add_fp16/ksearch-sonnet46_cuda --output-dir best_kernel/
    python extract_best_kernel.py elementwise_add_fp16/k-search/hlsl_opus47_first_try -o best_hlsl/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_artifacts_root(base: Path) -> Path:
    """Resolve the actual artifacts root (handles the nested task-name dir)."""
    # The structure is: <base>/<task_name>/solutions/<task_name>/*.json
    # Or: <base>/<task_name>/world_model/solution_db.jsonl
    # Find the first dir that has solutions/ or world_model/
    if (base / "solutions").exists() or (base / "world_model").exists():
        return base
    # Try one level down
    for child in base.iterdir():
        if child.is_dir() and ((child / "solutions").exists() or (child / "world_model").exists()):
            return child
    return base


def _normalize_eval_data(eval_data: dict) -> dict:
    """Normalize eval fields across different backends (cuda_kernel vs kernelbench/triton)."""
    # Map speedup_over_eager -> speedup_factor if missing
    if "speedup_factor" not in eval_data and "speedup_over_eager" in eval_data:
        eval_data["speedup_factor"] = eval_data["speedup_over_eager"]
    # Normalize status
    if "status" not in eval_data:
        eval_data["status"] = "unknown"
    return eval_data


def load_solutions_and_evals(artifacts_root: Path) -> list[dict]:
    """Load all solutions with their eval results."""
    results = []

    # Strategy 1: scan eval reports
    eval_dir = artifacts_root / "eval"
    sol_dir = artifacts_root / "solutions"

    if eval_dir.exists():
        for eval_file in eval_dir.rglob("eval_report_*.json"):
            try:
                report = json.loads(eval_file.read_text())
                # Format A (cuda_kernel): {sol_name: eval_data, ...}
                # Format B (kernelbench/triton): {task, problem, ..., solutions: [...]}
                if "solutions" in report and isinstance(report["solutions"], list):
                    # Format B
                    for sol_entry in report["solutions"]:
                        sol_name = sol_entry.get("solution", "")
                        eval_data = _normalize_eval_data(dict(sol_entry))
                        sol_data = _find_solution_json(sol_dir, sol_name)
                        if sol_data:
                            results.append({
                                "name": sol_name,
                                "eval": eval_data,
                                "solution": sol_data,
                                "source": str(eval_file),
                            })
                else:
                    # Format A
                    for sol_name, eval_data in report.items():
                        eval_data = _normalize_eval_data(eval_data)
                        sol_data = _find_solution_json(sol_dir, sol_name)
                        if sol_data:
                            results.append({
                                "name": sol_name,
                                "eval": eval_data,
                                "solution": sol_data,
                                "source": str(eval_file),
                            })
            except Exception as e:
                print(f"  [warn] error reading {eval_file}: {e}", file=sys.stderr)

    # Strategy 2: also check solution_db.jsonl
    db_path = artifacts_root / "world_model" / "solution_db.jsonl"
    if db_path.exists():
        with open(db_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    eval_data = _normalize_eval_data(record.get("eval_result", {}))
                    sol_name = record.get("solution_name", "")
                    # Only add if not already found via eval reports
                    if sol_name and not any(r["name"] == sol_name for r in results):
                        sol_data = _find_solution_json(sol_dir, sol_name)
                        # If no solution file, build from JSONL record directly
                        if not sol_data and "code" in record:
                            sol_data = _build_solution_from_record(record)
                        if sol_data:
                            results.append({
                                "name": sol_name,
                                "eval": eval_data,
                                "solution": sol_data,
                                "source": str(db_path),
                            })
                except Exception:
                    continue

    return results


def _parse_xml_sources(code: str) -> list[dict]:
    """Parse XML-formatted code into a list of {path, content} dicts."""
    import re
    sources = []
    # Match CUDA multi-file blocks, HLSL blocks, and generic <file name="..."> blocks.
    pattern = r'<(header_file|cuda_file|cpp_file|hlsl_file|json_file|file)\s+name="([^"]+)">(.*?)</\1>'
    for match in re.finditer(pattern, code, re.DOTALL):
        sources.append({"path": match.group(2), "content": match.group(3).strip()})
    # Also handle model_new.py (triton/python format) — just raw code
    if not sources:
        sources.append({"path": "model_new.py", "content": code})
    return sources


def _build_solution_from_record(record: dict) -> dict:
    """Build a solution dict from a solution_db.jsonl record."""
    code = record.get("code", "")
    sources = _parse_xml_sources(code)
    return {
        "name": record.get("solution_name", "unknown"),
        "definition": record.get("definition", ""),
        "author": "unknown",
        "description": f"Extracted from solution_db.jsonl",
        "sources": sources,
    }


def _find_solution_json(sol_dir: Path, sol_name: str) -> dict | None:
    """Find a solution JSON file by name."""
    if not sol_dir.exists():
        return None
    for f in sol_dir.rglob(f"{sol_name}.json"):
        try:
            return json.loads(f.read_text())
        except Exception:
            continue
    # Try partial match
    for f in sol_dir.rglob("*.json"):
        if sol_name in f.stem:
            try:
                return json.loads(f.read_text())
            except Exception:
                continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract fastest kernel from K-Search artifacts")
    parser.add_argument("artifacts_dir", help="Path to K-Search artifacts folder")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory to write kernel files, e.g. kernel.hlsl/launch.json or kernel.h/kernel.cu/main.cpp")
    parser.add_argument("--metric", default="speedup_factor",
                        choices=["speedup_factor", "latency_ms"],
                        help="Metric to rank by (default: speedup_factor)")
    args = parser.parse_args()

    base = Path(args.artifacts_dir).resolve()
    if not base.exists():
        sys.exit(f"Artifacts directory not found: {base}")

    artifacts_root = find_artifacts_root(base)
    print(f"Scanning: {artifacts_root}")

    results = load_solutions_and_evals(artifacts_root)
    if not results:
        sys.exit("No solutions with eval results found.")

    # Filter to passed solutions only
    passed = [r for r in results if str(r["eval"].get("status", "")).lower() == "passed"]
    if not passed:
        print(f"No PASSED solutions found ({len(results)} total solutions).")
        print("All results:")
        for r in results:
            print(f"  {r['name']}: status={r['eval'].get('status')} latency={r['eval'].get('latency_ms')}")
        sys.exit(1)

    # Sort by metric
    if args.metric == "speedup_factor":
        passed.sort(key=lambda r: float(r["eval"].get("speedup_factor") or 0), reverse=True)
    else:  # latency_ms (lower is better)
        passed.sort(key=lambda r: float(r["eval"].get("latency_ms") or float("inf")))

    best = passed[0]
    eval_data = best["eval"]
    sol_data = best["solution"]

    print(f"\n{'='*60}")
    print(f"BEST KERNEL: {best['name']}")
    print(f"{'='*60}")
    print(f"  Status:       {eval_data.get('status')}")
    print(f"  Latency:      {eval_data.get('latency_ms', 0):.4f} ms")
    print(f"  Ref latency:  {eval_data.get('reference_latency_ms', 0):.4f} ms")
    print(f"  Speedup:      {eval_data.get('speedup_factor', 0):.3f}x")
    print(f"  Author:       {sol_data.get('author', 'unknown')}")
    print(f"  Description:  {sol_data.get('description', '-')}")

    if len(passed) > 1:
        print(f"\nAll {len(passed)} PASSED solutions (ranked by {args.metric}):")
        for i, r in enumerate(passed):
            e = r["eval"]
            lat = e.get("latency_ms", 0)
            spd = e.get("speedup_factor", 0)
            print(f"  {i+1}. {r['name'][:60]}  lat={lat:.4f}ms  speedup={spd:.3f}x")

    # Write kernel files
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sources = sol_data.get("sources", [])
        for src in sources:
            path = src.get("path", "")
            content = src.get("content", "")
            if path and content:
                (out_dir / path).write_text(content)
                print(f"\n  Wrote: {out_dir / path}")

        # Also save the full solution JSON
        (out_dir / "solution.json").write_text(json.dumps(sol_data, indent=2))
        print(f"  Wrote: {out_dir / 'solution.json'}")
    else:
        # Print main source content
        sources = sol_data.get("sources", [])
        # Prefer the backend's main source, otherwise print first/only source.
        display_src = None
        for src in sources:
            if src.get("path") == "kernel.hlsl":
                display_src = src
                break
        for src in sources:
            if display_src is None and src.get("path") == "kernel.cu":
                display_src = src
                break
        if display_src is None and sources:
            display_src = sources[0]
        if display_src:
            print(f"\n{'='*60}")
            print(f"{display_src['path']}:")
            print(f"{'='*60}")
            print(display_src["content"])


if __name__ == "__main__":
    main()
