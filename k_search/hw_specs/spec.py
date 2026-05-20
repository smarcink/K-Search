"""HWSpec dataclass and rendering logic.

All field names are vendor-neutral; vendor-specific terminology is carried
in ``*_label`` fields and used by ``render_for_prompt()`` so the LLM sees
the exact terms from its training data (SM/EU/CU, Warp/Subgroup/Wavefront, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MatrixAccelOp:
    """Throughput of a matrix accelerator for a specific data type."""

    dtype: str  # e.g. "FP16", "BF16", "TF32", "FP8", "INT8"
    throughput_tflops: float
    shape: str | None = None  # e.g. "m16n8k16", "DPAS 8x8"
    accum_dtype: str | None = None  # e.g. "FP32", "INT32"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "MatrixAccelOp":
        return MatrixAccelOp(
            dtype=str(d.get("dtype", "")),
            throughput_tflops=float(d.get("throughput_tflops", 0.0)),
            shape=d.get("shape"),
            accum_dtype=d.get("accum_dtype"),
        )


@dataclass
class HWSpec:
    """Structured GPU hardware specification.

    Field names are vendor-neutral.  The ``*_label`` fields carry the
    vendor-specific term used when rendering for the LLM.
    """

    # --- Identity ---
    name: str  # e.g. "H100 SXM", "Intel Arc B580"
    vendor: str  # "nvidia" | "intel" | "amd"
    architecture: str  # e.g. "Hopper", "Xe2/Battlemage", "RDNA 4"
    compute_capability: str = ""  # e.g. "sm_90", "xe2", "gfx1201"
    aliases: list[str] = field(default_factory=list)

    # --- Compute ---
    compute_units: int = 0  # SMs / EUs / CUs
    compute_unit_label: str = "Compute Unit"  # "SM" / "Execution Unit (EU)" / "CU"
    max_threads_per_unit: int = 0
    simd_width: int = 0  # 32 (warp) / 16 (subgroup) / 64 (wavefront)
    simd_unit_label: str = "SIMD Lane Group"  # "Warp" / "Subgroup" / "Wavefront"
    clock_mhz: int = 0
    peak_fp32_tflops: float = 0.0
    peak_fp16_tflops: float = 0.0

    # --- Matrix Accelerator ---
    matrix_accel_label: str = ""  # "Tensor Core" / "XMX (Xe Matrix Extension)" / "Matrix Core"
    matrix_accel_ops: list[MatrixAccelOp] = field(default_factory=list)

    # --- Memory ---
    memory_size_gb: float = 0.0
    memory_type: str = ""  # "HBM3", "GDDR6", "HBM2e"
    memory_bandwidth_gb_s: float = 0.0

    # --- Cache / SRAM ---
    l2_cache_mb: float = 0.0
    shared_mem_kb_per_unit: float = 0.0
    shared_mem_label: str = "Shared Memory"  # "Shared Memory" / "SLM" / "LDS"
    max_shared_mem_per_block_kb: float = 0.0

    # --- Registers ---
    registers_per_thread: int = 0
    registers_per_unit: int = 0

    # --- Behavioral notes (vendor-native perf tips) ---
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def from_file(path: str | Path) -> "HWSpec":
        """Load an HWSpec from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"HW spec file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return HWSpec.from_dict(data)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "HWSpec":
        """Construct an HWSpec from a JSON-like dict."""
        accel_ops_raw = d.get("matrix_accel_ops") or []
        accel_ops = [MatrixAccelOp.from_dict(op) for op in accel_ops_raw]
        return HWSpec(
            name=str(d.get("name", "")),
            vendor=str(d.get("vendor", "")),
            architecture=str(d.get("architecture", "")),
            compute_capability=str(d.get("compute_capability", "")),
            aliases=list(d.get("aliases") or []),
            compute_units=int(d.get("compute_units", 0)),
            compute_unit_label=str(d.get("compute_unit_label", "Compute Unit")),
            max_threads_per_unit=int(d.get("max_threads_per_unit", 0)),
            simd_width=int(d.get("simd_width", 0)),
            simd_unit_label=str(d.get("simd_unit_label", "SIMD Lane Group")),
            clock_mhz=int(d.get("clock_mhz", 0)),
            peak_fp32_tflops=float(d.get("peak_fp32_tflops", 0.0)),
            peak_fp16_tflops=float(d.get("peak_fp16_tflops", 0.0)),
            matrix_accel_label=str(d.get("matrix_accel_label", "")),
            matrix_accel_ops=accel_ops,
            memory_size_gb=float(d.get("memory_size_gb", 0.0)),
            memory_type=str(d.get("memory_type", "")),
            memory_bandwidth_gb_s=float(d.get("memory_bandwidth_gb_s", 0.0)),
            l2_cache_mb=float(d.get("l2_cache_mb", 0.0)),
            shared_mem_kb_per_unit=float(d.get("shared_mem_kb_per_unit", 0.0)),
            shared_mem_label=str(d.get("shared_mem_label", "Shared Memory")),
            max_shared_mem_per_block_kb=float(d.get("max_shared_mem_per_block_kb", 0.0)),
            registers_per_thread=int(d.get("registers_per_thread", 0)),
            registers_per_unit=int(d.get("registers_per_unit", 0)),
            notes=list(d.get("notes") or []),
        )

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def render_for_prompt(self) -> str:
        """Render a compact, token-efficient text block using vendor-native labels.

        The output is designed to be injected into LLM prompts as-is.
        """
        # Short label for the compute unit (strip parenthetical for inline use)
        cu_short = self.compute_unit_label.split("(")[0].strip()

        lines: list[str] = []
        lines.append("## Target Hardware Specification")

        # Identity line
        id_parts = [f"GPU: {self.name}"]
        arch_parts: list[str] = []
        if self.architecture:
            arch_parts.append(self.architecture)
        if self.compute_capability:
            arch_parts.append(self.compute_capability)
        if arch_parts:
            id_parts[0] += f" ({', '.join(arch_parts)})"
        lines.append(f"- {id_parts[0]}")

        # Compute line
        compute_parts: list[str] = []
        if self.compute_units:
            compute_parts.append(f"{self.compute_unit_label}s: {self.compute_units}")
        if self.simd_width:
            compute_parts.append(f"{self.simd_unit_label} size: {self.simd_width}")
        if self.max_threads_per_unit:
            compute_parts.append(f"Max threads/{cu_short}: {self.max_threads_per_unit}")
        if compute_parts:
            lines.append(f"- {' | '.join(compute_parts)}")

        # Clock + scalar peak line
        scalar_parts: list[str] = []
        if self.clock_mhz:
            scalar_parts.append(f"Clock: {self.clock_mhz} MHz")
        if self.peak_fp16_tflops:
            scalar_parts.append(f"Peak FP16: {self.peak_fp16_tflops:g} TFLOPS")
        if self.peak_fp32_tflops:
            scalar_parts.append(f"Peak FP32: {self.peak_fp32_tflops:g} TFLOPS")
        if scalar_parts:
            lines.append(f"- {' | '.join(scalar_parts)}")

        # Matrix accelerator throughput (per-dtype)
        if self.matrix_accel_label and self.matrix_accel_ops:
            lines.append(f"- {self.matrix_accel_label} throughput:")
            for op in self.matrix_accel_ops:
                unit = "TOPS" if op.dtype.upper().startswith("INT") else "TFLOPS"
                detail_parts: list[str] = []
                if op.shape:
                    detail_parts.append(op.shape)
                if op.accum_dtype:
                    detail_parts.append(f"accum {op.accum_dtype}")
                detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
                lines.append(f"    {op.dtype}: {op.throughput_tflops:g} {unit}{detail}")

        # Memory line
        mem_parts: list[str] = []
        if self.memory_size_gb:
            s = f"{self.memory_size_gb:g} GB"
            if self.memory_type:
                s += f" {self.memory_type}"
            mem_parts.append(s)
        if self.memory_bandwidth_gb_s:
            mem_parts.append(f"{self.memory_bandwidth_gb_s:g} GB/s peak bandwidth")
        if mem_parts:
            lines.append(f"- Memory: {', '.join(mem_parts)}")

        # Cache / SRAM / Registers line
        cache_parts: list[str] = []
        if self.l2_cache_mb:
            cache_parts.append(f"L2: {self.l2_cache_mb:g} MB")
        if self.shared_mem_kb_per_unit:
            s = f"{self.shared_mem_label}/{cu_short}: {self.shared_mem_kb_per_unit:g} KB"
            if self.max_shared_mem_per_block_kb:
                s += f" (max {self.max_shared_mem_per_block_kb:g} KB/block)"
            cache_parts.append(s)
        if cache_parts:
            lines.append(f"- {' | '.join(cache_parts)}")

        reg_parts: list[str] = []
        if self.registers_per_thread:
            reg_parts.append(f"{self.registers_per_thread}/thread")
        if self.registers_per_unit:
            reg_parts.append(f"{self.registers_per_unit}/{cu_short}")
        if reg_parts:
            lines.append(f"- Registers: {', '.join(reg_parts)}")

        # Notes
        if self.notes:
            notes_text = "; ".join(str(n).strip() for n in self.notes if str(n).strip())
            if notes_text:
                lines.append(f"- Notes: {notes_text}")

        return "\n".join(lines)
