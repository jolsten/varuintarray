import numpy as np
import pytest

from varuintarray.array import VarUIntArray


class TestWordSize64:
    """Tests for the maximum supported word_size=64."""

    def test_create(self):
        arr = VarUIntArray([0, 1, 2**64 - 1], word_size=64)
        assert arr.word_size == 64
        assert arr[2] == 2**64 - 1

    def test_invert(self):
        arr = VarUIntArray([0], word_size=64)
        result = np.invert(arr)
        assert result[0] == 2**64 - 1

    def test_unpackbits_roundtrip(self):
        arr = VarUIntArray([1, 2**64 - 1], word_size=64)
        bits = arr.unpackbits()
        assert bits.shape == (2, 64)
        packed = VarUIntArray.packbits(bits)
        assert packed.tolist() == arr.tolist()


class TestZeroDimensional:
    """Tests for 0-d (scalar) VarUIntArray."""

    def test_create(self):
        arr = VarUIntArray(5, word_size=4)
        assert arr.ndim == 0
        assert arr.word_size == 4
        assert int(arr) == 5

    def test_invert(self):
        arr = VarUIntArray(0x5, word_size=4)
        result = np.invert(arr)
        assert int(result) == 0xA

    def test_serialize_roundtrip(self):
        arr = VarUIntArray(42, word_size=10)
        restored = VarUIntArray.from_json(arr.to_json())
        assert int(restored) == 42
        assert restored.word_size == 10


class TestEmptyArray:
    """Tests for empty VarUIntArray."""

    def test_create(self):
        arr = VarUIntArray([], word_size=10)
        assert arr.shape == (0,)
        assert arr.word_size == 10

    def test_invert(self):
        arr = VarUIntArray([], word_size=10)
        result = np.invert(arr)
        assert result.shape == (0,)

    def test_serialize_roundtrip(self):
        arr = VarUIntArray([], word_size=10)
        restored = VarUIntArray.from_dict(arr.to_dict())
        assert restored.shape == (0,)
        assert restored.word_size == 10


class TestMismatchedWordSize:
    """Tests for operations between VarUIntArrays with different word_size."""

    def test_add_mismatched_raises(self):
        a = VarUIntArray([1, 2], word_size=4)
        b = VarUIntArray([1, 2], word_size=8)
        with pytest.raises(ValueError, match="different word sizes"):
            np.add(a, b)

    def test_xor_mismatched_raises(self):
        a = VarUIntArray([1, 2], word_size=10)
        b = VarUIntArray([1, 2], word_size=12)
        with pytest.raises(ValueError, match="different word sizes"):
            np.bitwise_xor(a, b)

    def test_eq_mismatched_raises(self):
        a = VarUIntArray([5], word_size=3)
        b = VarUIntArray([5], word_size=8)
        with pytest.raises(ValueError, match="different word sizes"):
            a == b

    def test_eq_same_word_size_ok(self):
        a = VarUIntArray([1, 2, 3], word_size=10)
        b = VarUIntArray([1, 2, 3], word_size=10)
        np.testing.assert_array_equal(a == b, [True, True, True])


class TestSubtractUnderflow:
    """Tests for subtraction underflow detection."""

    def test_underflow_raises(self):
        arr = VarUIntArray([0], word_size=4)
        with pytest.raises(OverflowError, match="underflow"):
            np.subtract(arr, 1)

    def test_no_underflow_ok(self):
        arr = VarUIntArray([5], word_size=4)
        result = np.subtract(arr, 3)
        assert result[0] == 2
