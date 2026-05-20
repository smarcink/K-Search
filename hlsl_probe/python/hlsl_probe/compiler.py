from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DXC = _REPO_ROOT / "thirdparty" / "dxc_2026_02_20" / "bin" / "x64" / "dxc.exe"
_HLSL_INCLUDE = _REPO_ROOT / "thirdparty" / "dxc_2026_02_20" / "inc" / "hlsl"


def compile_hlsl_source(
    source: str,
    *,
    entry: str = "main",
    target: str = "cs_6_8",
    enable_16bit_types: bool = True,
    extra_args: list[str] | None = None,
) -> bytes:
    if not _DXC.exists():
        raise FileNotFoundError(f"DXC not found: {_DXC}")
    with tempfile.TemporaryDirectory(prefix="hlsl_probe_compile_") as tmp_dir:
        tmp = Path(tmp_dir)
        hlsl_path = tmp / "kernel.hlsl"
        dxil_path = tmp / "kernel.dxil"
        hlsl_path.write_text(source, encoding="utf-8")

        cmd = [
            str(_DXC),
            "-T", target,
            "-E", entry,
            "-HV", "2021",
            "-I", str(_HLSL_INCLUDE),
            "-Fo", str(dxil_path),
            str(hlsl_path),
        ]
        if enable_16bit_types:
            cmd.insert(-1, "-enable-16bit-types")
        if extra_args:
            cmd[1:1] = list(extra_args)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "DXC failed with exit code "
                f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return dxil_path.read_bytes()


def compile_hlsl_file(path: str | Path, **kwargs) -> bytes:
    return compile_hlsl_source(Path(path).read_text(encoding="utf-8"), **kwargs)
