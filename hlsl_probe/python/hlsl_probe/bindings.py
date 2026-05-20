from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Iterable


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DLL_CANDIDATES = [
    _ROOT / "build" / "bin" / "Debug" / "hlsl_probe_native.dll",
    _ROOT / "build" / "bin" / "Release" / "hlsl_probe_native.dll",
]


class HlslProbeError(RuntimeError):
    pass


class _InputBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("size_bytes", ctypes.c_uint64),
    ]


class _OutputBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("size_bytes", ctypes.c_uint64),
    ]


class _RunConfig(ctypes.Structure):
    _fields_ = [
        ("dispatch_x", ctypes.c_uint32),
        ("dispatch_y", ctypes.c_uint32),
        ("dispatch_z", ctypes.c_uint32),
        ("inputs", ctypes.POINTER(_InputBuffer)),
        ("input_count", ctypes.c_uint32),
        ("outputs", ctypes.POINTER(_OutputBuffer)),
        ("output_count", ctypes.c_uint32),
    ]


def _find_dll() -> Path:
    override = os.getenv("HLSL_PROBE_NATIVE_DLL")
    if override:
        path = Path(override)
        if path.exists():
            return path
        raise HlslProbeError(f"HLSL_PROBE_NATIVE_DLL does not exist: {path}")
    for candidate in _DEFAULT_DLL_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in _DEFAULT_DLL_CANDIDATES)
    raise HlslProbeError(f"hlsl_probe_native.dll was not found. Searched: {searched}")


def _load_library() -> ctypes.CDLL:
    dll_path = _find_dll()
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_path.parent))
    dll = ctypes.CDLL(str(dll_path))
    dll.hlsl_probe_create.argtypes = [ctypes.c_char_p]
    dll.hlsl_probe_create.restype = ctypes.c_void_p
    dll.hlsl_probe_destroy.argtypes = [ctypes.c_void_p]
    dll.hlsl_probe_destroy.restype = None
    dll.hlsl_probe_get_last_error.argtypes = [ctypes.c_void_p]
    dll.hlsl_probe_get_last_error.restype = ctypes.c_char_p
    dll.hlsl_probe_get_global_last_error.argtypes = []
    dll.hlsl_probe_get_global_last_error.restype = ctypes.c_char_p
    dll.hlsl_probe_get_caps_json.argtypes = [ctypes.c_void_p]
    dll.hlsl_probe_get_caps_json.restype = ctypes.c_void_p
    dll.hlsl_probe_run_dxil.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(_RunConfig),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    dll.hlsl_probe_run_dxil.restype = ctypes.c_int
    dll.hlsl_probe_free_string.argtypes = [ctypes.c_void_p]
    dll.hlsl_probe_free_string.restype = None
    return dll


_DLL: ctypes.CDLL | None = None


def _dll() -> ctypes.CDLL:
    global _DLL
    if _DLL is None:
        _DLL = _load_library()
    return _DLL


def _decode_and_free(ptr: int | None) -> str:
    if not ptr:
        return ""
    dll = _dll()
    try:
        return ctypes.string_at(ptr).decode("utf-8")
    finally:
        dll.hlsl_probe_free_string(ptr)


def _last_error(handle: int | None) -> str:
    dll = _dll()
    raw = dll.hlsl_probe_get_last_error(handle) if handle else dll.hlsl_probe_get_global_last_error()
    return raw.decode("utf-8", errors="replace") if raw else "unknown HLSL probe error"


class HlslProbe:
    def __init__(self, agility_sdk_path: str | os.PathLike[str] | None = None) -> None:
        sdk_bytes = str(agility_sdk_path).encode("utf-8") if agility_sdk_path is not None else None
        self._handle = _dll().hlsl_probe_create(sdk_bytes)
        if not self._handle:
            raise HlslProbeError(_last_error(None))

    def close(self) -> None:
        if self._handle:
            _dll().hlsl_probe_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "HlslProbe":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def caps(self) -> dict:
        ptr = _dll().hlsl_probe_get_caps_json(self._handle)
        if not ptr:
            raise HlslProbeError(_last_error(self._handle))
        return json.loads(_decode_and_free(ptr))

    def run_dxil(
        self,
        dxil: bytes | bytearray | memoryview,
        *,
        inputs: Iterable[bytes | bytearray | memoryview] = (),
        output_sizes: Iterable[int],
        dispatch: tuple[int, int, int] = (1, 1, 1),
    ) -> tuple[dict, list[bytes]]:
        dxil_bytes = bytes(dxil)
        dxil_buf = ctypes.create_string_buffer(dxil_bytes)

        input_bytes = [bytes(item) for item in inputs]
        input_storage = [ctypes.create_string_buffer(item) for item in input_bytes]
        input_array_type = _InputBuffer * max(1, len(input_storage))
        input_array = input_array_type()
        for index, storage in enumerate(input_storage):
            input_array[index] = _InputBuffer(ctypes.cast(storage, ctypes.c_void_p), len(input_bytes[index]))

        sizes = [int(size) for size in output_sizes]
        output_storage = [(ctypes.c_ubyte * size)() for size in sizes]
        output_array_type = _OutputBuffer * max(1, len(output_storage))
        output_array = output_array_type()
        for index, storage in enumerate(output_storage):
            output_array[index] = _OutputBuffer(ctypes.cast(storage, ctypes.c_void_p), sizes[index])

        config = _RunConfig(
            dispatch[0],
            dispatch[1],
            dispatch[2],
            input_array if input_storage else None,
            len(input_storage),
            output_array if output_storage else None,
            len(output_storage),
        )
        result_ptr = ctypes.c_void_p()
        ok = _dll().hlsl_probe_run_dxil(
            self._handle,
            ctypes.cast(dxil_buf, ctypes.c_void_p),
            len(dxil_bytes),
            ctypes.byref(config),
            ctypes.byref(result_ptr),
        )
        if not ok:
            raise HlslProbeError(_last_error(self._handle))
        result = json.loads(_decode_and_free(result_ptr.value))
        outputs = [bytes(storage) for storage in output_storage]
        return result, outputs


def probe_caps() -> dict:
    with HlslProbe() as probe:
        return probe.caps()


def run_dxil(*args, **kwargs):
    with HlslProbe() as probe:
        return probe.run_dxil(*args, **kwargs)


def self_test() -> dict:
    from .compiler import compile_hlsl_source

    shader = """
RWByteAddressBuffer Output : register(u0);
[numthreads(1, 1, 1)]
void main() { Output.Store(0, 123u); }
"""
    dxil = compile_hlsl_source(shader)
    result, outputs = run_dxil(dxil, output_sizes=[4])
    value = int.from_bytes(outputs[0], "little")
    result["output_value"] = value
    result["status"] = "passed" if value == 123 else "failed"
    return result


def _classify_run_error(error: str) -> str:
    if "CreateComputePipelineState" in error:
        return "pso"
    if "Dispatch" in error or "ExecuteCommandLists" in error or "Signal" in error:
        return "dispatch"
    if "readback" in error or "Map(output" in error:
        return "readback"
    return "run"


def linalg_test(target: str = "cs_6_10") -> dict:
    from .compiler import compile_hlsl_source

    shader = """
#include <dx/linalg.h>

ByteAddressBuffer MatrixData : register(t0);
RWByteAddressBuffer Output : register(u0);

[numthreads(1, 1, 1)]
void main() {
    dx::linalg::Matrix<dx::linalg::ComponentType::U32, 2, 2, dx::linalg::MatrixUse::A, dx::linalg::MatrixScope::Thread> matrix = dx::linalg::Matrix<dx::linalg::ComponentType::U32, 2, 2, dx::linalg::MatrixUse::A, dx::linalg::MatrixScope::Thread>::Load<dx::linalg::MatrixLayout::RowMajor>(MatrixData, 0, 8);
    vector<uint32_t, 2> input = { 5u, 6u };
    vector<uint32_t, 2> result = dx::linalg::Multiply<uint32_t>(matrix, input);
    Output.Store(0, result.x);
    Output.Store(4, result.y);
}
"""
    try:
        dxil = compile_hlsl_source(shader, target=target)
    except Exception as exc:
        return {"status": "failed", "stage": "compile", "target": target, "error": str(exc)}

    matrix = b"".join(value.to_bytes(4, "little") for value in [1, 2, 3, 4])
    try:
        result, outputs = run_dxil(dxil, inputs=[matrix], output_sizes=[8])
    except HlslProbeError as exc:
        error = str(exc)
        return {"status": "failed", "stage": _classify_run_error(error), "target": target, "error": error}

    values = [int.from_bytes(outputs[0][0:4], "little"), int.from_bytes(outputs[0][4:8], "little")]
    expected = [17, 39]
    result["target"] = target
    result["dxil_size"] = len(dxil)
    result["expected_values"] = expected
    result["output_values"] = values
    result["stage"] = "dispatch" if values == expected else "verify"
    result["status"] = "passed" if values == expected else "failed"
    return result
