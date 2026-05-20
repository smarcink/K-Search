from __future__ import annotations

import ctypes
import json
import os
import struct
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Iterable, Sequence


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DLL_CANDIDATES = [
    _ROOT / "build" / "bin" / "Debug" / "hlsl_probe_native.dll",
    _ROOT / "build" / "bin" / "Release" / "hlsl_probe_native.dll",
]

HLSL_PROBE_BUFFER_VIEW_RAW = 0
HLSL_PROBE_BUFFER_VIEW_TYPED = 1

HLSL_PROBE_BUFFER_FORMAT_RAW_U32 = 0
HLSL_PROBE_BUFFER_FORMAT_F16 = 1
HLSL_PROBE_BUFFER_FORMAT_F32 = 2
HLSL_PROBE_BUFFER_FORMAT_I8 = 3
HLSL_PROBE_BUFFER_FORMAT_U8 = 4
HLSL_PROBE_BUFFER_FORMAT_I32 = 5
HLSL_PROBE_BUFFER_FORMAT_U32 = 6

_VIEW_ALIASES = {
    "raw": HLSL_PROBE_BUFFER_VIEW_RAW,
    "typed": HLSL_PROBE_BUFFER_VIEW_TYPED,
}

_FORMAT_ALIASES = {
    "raw_u32": HLSL_PROBE_BUFFER_FORMAT_RAW_U32,
    "float16": HLSL_PROBE_BUFFER_FORMAT_F16,
    "fp16": HLSL_PROBE_BUFFER_FORMAT_F16,
    "f16": HLSL_PROBE_BUFFER_FORMAT_F16,
    "half": HLSL_PROBE_BUFFER_FORMAT_F16,
    "float32": HLSL_PROBE_BUFFER_FORMAT_F32,
    "fp32": HLSL_PROBE_BUFFER_FORMAT_F32,
    "f32": HLSL_PROBE_BUFFER_FORMAT_F32,
    "float": HLSL_PROBE_BUFFER_FORMAT_F32,
    "int8": HLSL_PROBE_BUFFER_FORMAT_I8,
    "i8": HLSL_PROBE_BUFFER_FORMAT_I8,
    "uint8": HLSL_PROBE_BUFFER_FORMAT_U8,
    "u8": HLSL_PROBE_BUFFER_FORMAT_U8,
    "int32": HLSL_PROBE_BUFFER_FORMAT_I32,
    "i32": HLSL_PROBE_BUFFER_FORMAT_I32,
    "uint32": HLSL_PROBE_BUFFER_FORMAT_U32,
    "u32": HLSL_PROBE_BUFFER_FORMAT_U32,
}

_FORMAT_NAMES = {
    HLSL_PROBE_BUFFER_FORMAT_RAW_U32: "raw_u32",
    HLSL_PROBE_BUFFER_FORMAT_F16: "float16",
    HLSL_PROBE_BUFFER_FORMAT_F32: "float32",
    HLSL_PROBE_BUFFER_FORMAT_I8: "int8",
    HLSL_PROBE_BUFFER_FORMAT_U8: "uint8",
    HLSL_PROBE_BUFFER_FORMAT_I32: "int32",
    HLSL_PROBE_BUFFER_FORMAT_U32: "uint32",
}

_VIEW_NAMES = {
    HLSL_PROBE_BUFFER_VIEW_RAW: "raw",
    HLSL_PROBE_BUFFER_VIEW_TYPED: "typed",
}

_BYTES_PER_ELEMENT = {
    HLSL_PROBE_BUFFER_FORMAT_RAW_U32: 4,
    HLSL_PROBE_BUFFER_FORMAT_F16: 2,
    HLSL_PROBE_BUFFER_FORMAT_F32: 4,
    HLSL_PROBE_BUFFER_FORMAT_I8: 1,
    HLSL_PROBE_BUFFER_FORMAT_U8: 1,
    HLSL_PROBE_BUFFER_FORMAT_I32: 4,
    HLSL_PROBE_BUFFER_FORMAT_U32: 4,
}


class HlslProbeError(RuntimeError):
    pass


class _BufferDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("view_kind", ctypes.c_uint32),
        ("format", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("element_count", ctypes.c_uint64),
        ("size_bytes", ctypes.c_uint64),
        ("reserved1", ctypes.c_uint64),
        ("reserved2", ctypes.c_uint64),
    ]


class _InputBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("desc", _BufferDesc),
    ]


class _OutputBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("desc", _BufferDesc),
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


@dataclass(frozen=True)
class BufferSpec:
    dtype: str
    element_count: int | None = None
    shape: Sequence[int] | None = None
    view: str = "typed"
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        dtype = _normalize_dtype(self.dtype)
        view = _normalize_view(self.view)
        shape = tuple(int(dim) for dim in self.shape) if self.shape is not None else None
        if shape is not None and any(dim < 0 for dim in shape):
            raise ValueError(f"shape cannot contain negative dimensions: {shape}")

        element_count = self.element_count
        if element_count is None and shape is not None:
            element_count = prod(shape)
        if element_count is None:
            raise ValueError("BufferSpec requires element_count or shape")
        element_count = int(element_count)
        if element_count < 0:
            raise ValueError(f"element_count must be nonnegative, got {element_count}")

        format_id = _FORMAT_ALIASES[dtype]
        if view == "raw" and format_id != HLSL_PROBE_BUFFER_FORMAT_RAW_U32:
            raise ValueError(f"raw view requires raw_u32 dtype, got {dtype}")
        if view == "typed" and format_id == HLSL_PROBE_BUFFER_FORMAT_RAW_U32:
            raise ValueError("raw_u32 dtype requires raw view")

        expected_size = element_count * _BYTES_PER_ELEMENT[format_id]
        size_bytes = expected_size if self.size_bytes is None else int(self.size_bytes)
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be nonnegative, got {size_bytes}")
        if size_bytes != expected_size:
            raise ValueError(
                f"{dtype} {view} buffer expected {expected_size} bytes from "
                f"element_count={element_count}, got size_bytes={size_bytes}"
            )

        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "view", view)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "element_count", element_count)
        object.__setattr__(self, "size_bytes", size_bytes)

    @classmethod
    def raw_u32(cls, size_bytes: int) -> "BufferSpec":
        size_bytes = int(size_bytes)
        if size_bytes % 4 != 0:
            raise ValueError(f"raw_u32 buffers require 4-byte size alignment, got {size_bytes}")
        return cls("raw_u32", element_count=size_bytes // 4, view="raw", size_bytes=size_bytes)

    @classmethod
    def from_tensor(cls, tensor, *, view: str = "typed") -> "BufferSpec":
        return cls(
            _dtype_from_tensor(tensor),
            element_count=int(tensor.numel()),
            shape=tuple(int(dim) for dim in tensor.shape),
            view=view,
            size_bytes=int(tensor.numel()) * int(tensor.element_size()),
        )

    @property
    def format_id(self) -> int:
        return _FORMAT_ALIASES[self.dtype]

    @property
    def view_id(self) -> int:
        return _VIEW_ALIASES[self.view]

    def to_ctype(self) -> _BufferDesc:
        return _BufferDesc(
            ctypes.sizeof(_BufferDesc),
            self.view_id,
            self.format_id,
            0,
            int(self.element_count),
            int(self.size_bytes),
            0,
            0,
        )


@dataclass(frozen=True)
class BufferArg:
    data: bytes | bytearray | memoryview
    spec: BufferSpec

    def __post_init__(self) -> None:
        data = bytes(self.data)
        if len(data) != self.spec.size_bytes:
            raise ValueError(
                f"BufferArg data size {len(data)} does not match spec size_bytes {self.spec.size_bytes}"
            )
        object.__setattr__(self, "data", data)

    @classmethod
    def raw_u32(cls, data: bytes | bytearray | memoryview) -> "BufferArg":
        data_bytes = bytes(data)
        return cls(data_bytes, BufferSpec.raw_u32(len(data_bytes)))

    @classmethod
    def from_tensor(cls, tensor, *, view: str = "typed") -> "BufferArg":
        contiguous = tensor.detach().cpu().contiguous()
        return cls(contiguous.numpy().tobytes(), BufferSpec.from_tensor(contiguous, view=view))


def _normalize_dtype(dtype: str) -> str:
    value = str(dtype).lower().replace("torch.", "").replace(" ", "_")
    if value not in _FORMAT_ALIASES:
        supported = ", ".join(sorted(_FORMAT_ALIASES))
        raise ValueError(f"unsupported buffer dtype {dtype!r}; supported aliases: {supported}")
    return _FORMAT_NAMES[_FORMAT_ALIASES[value]]


def _normalize_view(view: str) -> str:
    value = str(view).lower()
    if value not in _VIEW_ALIASES:
        supported = ", ".join(sorted(_VIEW_ALIASES))
        raise ValueError(f"unsupported buffer view {view!r}; supported: {supported}")
    return _VIEW_NAMES[_VIEW_ALIASES[value]]


def _dtype_from_tensor(tensor) -> str:
    dtype_text = str(tensor.dtype).replace("torch.", "")
    return _normalize_dtype(dtype_text)


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


_DLL_DIR_HANDLES = []


def _load_library() -> ctypes.CDLL:
    dll_path = _find_dll()
    if hasattr(os, "add_dll_directory"):
        _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_path.parent)))
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


def _normalize_input(item) -> BufferArg:
    if isinstance(item, BufferArg):
        return item
    if hasattr(item, "detach") and hasattr(item, "numel") and hasattr(item, "element_size"):
        return BufferArg.from_tensor(item)
    if isinstance(item, tuple) and len(item) == 2:
        data, spec = item
        if isinstance(spec, BufferSpec):
            return BufferArg(data, spec)
    raise TypeError("run_dxil inputs must be BufferArg objects, tensors, or (data, BufferSpec) pairs")


def _normalize_output(item) -> BufferSpec:
    if isinstance(item, BufferSpec):
        return item
    if hasattr(item, "detach") and hasattr(item, "numel") and hasattr(item, "element_size"):
        return BufferSpec.from_tensor(item)
    raise TypeError("run_dxil outputs must be BufferSpec objects or tensor-like objects")


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
        inputs: Iterable[BufferArg] = (),
        outputs: Iterable[BufferSpec],
        dispatch: tuple[int, int, int] = (1, 1, 1),
    ) -> tuple[dict, list[bytes]]:
        dxil_bytes = bytes(dxil)
        dxil_buf = ctypes.create_string_buffer(dxil_bytes)

        input_args = [_normalize_input(item) for item in inputs]
        input_storage = [ctypes.create_string_buffer(item.data) for item in input_args]
        input_array_type = _InputBuffer * max(1, len(input_storage))
        input_array = input_array_type()
        for index, storage in enumerate(input_storage):
            input_array[index] = _InputBuffer(
                ctypes.cast(storage, ctypes.c_void_p),
                input_args[index].spec.to_ctype(),
            )

        output_specs = [_normalize_output(item) for item in outputs]
        output_storage = [(ctypes.c_ubyte * int(spec.size_bytes))() for spec in output_specs]
        output_array_type = _OutputBuffer * max(1, len(output_storage))
        output_array = output_array_type()
        for index, storage in enumerate(output_storage):
            output_array[index] = _OutputBuffer(
                ctypes.cast(storage, ctypes.c_void_p),
                output_specs[index].to_ctype(),
            )

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
        output_bytes = [bytes(storage) for storage in output_storage]
        return result, output_bytes


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
    result, outputs = run_dxil(dxil, outputs=[BufferSpec.raw_u32(4)])
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
    if "format" in error or "metadata" in error or "BufferDesc" in error:
        return "metadata"
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
        result, outputs = run_dxil(
            dxil,
            inputs=[BufferArg.raw_u32(matrix)],
            outputs=[BufferSpec.raw_u32(8)],
        )
    except HlslProbeError as exc:
        error = str(exc)
        return {"status": "failed", "stage": _classify_run_error(error), "target": target, "error": error}

    values = [int.from_bytes(outputs[0][0:4], "little"), int.from_bytes(outputs[0][4:8], "little")]
    result.update(
        {
            "stage": "dispatch" if values == [17, 39] else "verify",
            "target": target,
            "expected_values": [17, 39],
            "output_values": values,
            "status": "passed" if values == [17, 39] else "failed",
        }
    )
    return result


def linalg_fp16_test(target: str = "cs_6_10") -> dict:
    from .compiler import compile_hlsl_source

    shader = """
#include <dx/linalg.h>

ByteAddressBuffer MatrixData : register(t0);
ByteAddressBuffer VectorData : register(t1);
RWByteAddressBuffer Output : register(u0);

[numthreads(1, 1, 1)]
void main() {
    dx::linalg::Matrix<dx::linalg::ComponentType::F16, 2, 4, dx::linalg::MatrixUse::A, dx::linalg::MatrixScope::Thread> matrix =
        dx::linalg::Matrix<dx::linalg::ComponentType::F16, 2, 4, dx::linalg::MatrixUse::A, dx::linalg::MatrixScope::Thread>::Load<dx::linalg::MatrixLayout::RowMajor>(MatrixData, 0, 4 * sizeof(float16_t));
    vector<float16_t, 4> input = VectorData.Load<vector<float16_t, 4> >(0);
    vector<float16_t, 2> result = dx::linalg::Multiply<float16_t>(matrix, input);
    Output.Store<vector<float16_t, 2> >(0, result);
}
"""
    try:
        dxil = compile_hlsl_source(shader, target=target)
    except Exception as exc:
        return {"status": "failed", "stage": "compile", "target": target, "error": str(exc)}

    matrix = struct.pack("<8e", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    vector = struct.pack("<4e", 0.5, -1.0, 2.0, 0.25)
    expected = [5.5, 12.5]
    try:
        result, outputs = run_dxil(
            dxil,
            inputs=[BufferArg.raw_u32(matrix), BufferArg.raw_u32(vector)],
            outputs=[BufferSpec.raw_u32(4)],
        )
    except HlslProbeError as exc:
        error = str(exc)
        return {"status": "failed", "stage": _classify_run_error(error), "target": target, "error": error}

    values = list(struct.unpack("<2e", outputs[0]))
    result.update(
        {
            "stage": "dispatch" if values == expected else "verify",
            "target": target,
            "expected_values": expected,
            "output_values": values,
            "status": "passed" if values == expected else "failed",
        }
    )
    return result