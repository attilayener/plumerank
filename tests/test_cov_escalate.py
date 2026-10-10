"""Coverage tests for escalate.py lines 184 and 255."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from plumerank.constants import TILE_SIZE
from plumerank.escalate import TileClassifier


def _minimal_onnx(path: pathlib.Path, out_dim: int = 2) -> pathlib.Path:
    """Write a minimal ONNX model for testing (copy of fixture logic)."""

    # Import the encoder functions from test_escalate
    def _varint(n: int) -> bytes:
        out = bytearray()
        while True:
            b = n & 0x7F
            n >>= 7
            out.append(b | 0x80 if n else b)
            if not n:
                return bytes(out)

    def _tag(field: int, wire: int) -> bytes:
        return _varint((field << 3) | wire)

    def _int_field(field: int, value: int) -> bytes:
        return _tag(field, 0) + _varint(value)

    def _len_field(field: int, payload: bytes | str) -> bytes:
        raw = payload.encode() if isinstance(payload, str) else payload
        return _tag(field, 2) + _varint(len(raw)) + raw

    def _value_info(name: str, dims: tuple[int, ...]) -> bytes:
        shape = b"".join(_len_field(1, _int_field(1, d)) for d in dims)
        tensor_type = _int_field(1, 1) + _len_field(2, shape)  # elem_type 1 == FLOAT
        return _len_field(1, name) + _len_field(2, _len_field(1, tensor_type))

    def _initializer(name: str, arr: np.ndarray) -> bytes:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        body = b"".join(_int_field(1, d) for d in arr.shape)
        return body + _int_field(2, 1) + _len_field(8, name) + _len_field(9, arr.tobytes())

    def _node(op: str, ins: tuple[str, ...], outs: tuple[str, ...], name: str) -> bytes:
        return (
            b"".join(_len_field(1, i) for i in ins)
            + b"".join(_len_field(2, o) for o in outs)
            + _len_field(3, name)
            + _len_field(4, op)
        )

    w = np.zeros((3, out_dim), dtype=np.float32)
    w[0, out_dim - 1] = 3.0
    graph = (
        _len_field(1, _node("GlobalAveragePool", ("input",), ("pooled",), "gap"))
        + _len_field(1, _node("Flatten", ("pooled",), ("flat",), "flat"))
        + _len_field(1, _node("Gemm", ("flat", "W", "B"), ("logits",), "fc"))
        + _len_field(2, "plumerank_test_tile_classifier")
        + _len_field(5, _initializer("W", w))
        + _len_field(5, _initializer("B", np.zeros(out_dim, dtype=np.float32)))
        + _len_field(11, _value_info("input", (1, 3, TILE_SIZE, TILE_SIZE)))
        + _len_field(12, _value_info("logits", (1, out_dim)))
    )
    model = (
        _int_field(1, 7)  # ir_version
        + _len_field(2, "plumerank-tests")
        + _len_field(7, graph)
        + _len_field(8, _len_field(1, "") + _int_field(2, 13))  # opset ai.onnx v13
    )
    path.write_bytes(model)
    return path


@pytest.fixture
def onnx_2class(tmp_path: pathlib.Path) -> pathlib.Path:
    """Minimal 2-class ONNX model."""
    return _minimal_onnx(tmp_path / "two.onnx", out_dim=2)


def test_path_property_returns_the_model_path(onnx_2class: pathlib.Path) -> None:
    """Line 184: path property is accessed and returns the pathlib.Path."""
    clf = TileClassifier(onnx_2class)
    assert clf.path == onnx_2class
    assert isinstance(clf.path, pathlib.Path)


def test_path_property_matches_constructor_argument(tmp_path: pathlib.Path) -> None:
    """Line 184: path property returns exactly what was passed to __init__."""
    onnx_file = _minimal_onnx(tmp_path / "model.onnx")
    clf = TileClassifier(onnx_file)
    assert clf.path is clf._path
    assert str(clf.path) == str(onnx_file)


def test_forward_refuses_a_tile_of_the_wrong_size(onnx_2class: pathlib.Path) -> None:
    """Line 255: _forward never resizes; a non-TILE_SIZE tile is refused."""
    clf = TileClassifier(onnx_2class)
    with pytest.raises(ValueError, match="expected"):
        clf._forward(np.zeros((TILE_SIZE - 1, TILE_SIZE, 3), dtype=np.uint8))
