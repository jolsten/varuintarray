"""Tests for boolean indexing and comparison ufunc behavior.

Regression tests for the bug where comparison ufuncs (==, !=, <, etc.)
returned VarUIntArray instead of plain ndarray, causing ~mask to produce
uint8 values (e.g. 255) instead of boolean values, which broke boolean
indexing and numpy.testing.assert_array_equal.
"""

import numpy as np
import pytest

from varuintarray.array import VarUIntArray


class TestComparisonReturnsNdarray:
    """Comparison ufuncs must return plain ndarray with bool dtype."""

    @pytest.mark.parametrize("word_size", [1, 4, 8, 10, 16, 32])
    def test_equal(self, word_size):
        arr = VarUIntArray([1, 2, 3], word_size=word_size)
        result = arr == arr
        assert type(result) is np.ndarray
        assert result.dtype == np.bool_

    @pytest.mark.parametrize("word_size", [1, 4, 8, 10, 16, 32])
    def test_not_equal(self, word_size):
        arr = VarUIntArray([1, 2, 3], word_size=word_size)
        result = arr != arr
        assert type(result) is np.ndarray
        assert result.dtype == np.bool_

    def test_less(self):
        a = VarUIntArray([1, 2, 3], word_size=8)
        b = VarUIntArray([2, 2, 2], word_size=8)
        result = a < b
        assert type(result) is np.ndarray
        np.testing.assert_array_equal(result, [True, False, False])

    def test_greater(self):
        a = VarUIntArray([1, 2, 3], word_size=8)
        b = VarUIntArray([2, 2, 2], word_size=8)
        result = a > b
        assert type(result) is np.ndarray
        np.testing.assert_array_equal(result, [False, False, True])

    def test_less_equal(self):
        a = VarUIntArray([1, 2, 3], word_size=8)
        b = VarUIntArray([2, 2, 2], word_size=8)
        result = a <= b
        assert type(result) is np.ndarray
        np.testing.assert_array_equal(result, [True, True, False])

    def test_greater_equal(self):
        a = VarUIntArray([1, 2, 3], word_size=8)
        b = VarUIntArray([2, 2, 2], word_size=8)
        result = a >= b
        assert type(result) is np.ndarray
        np.testing.assert_array_equal(result, [False, True, True])


class TestBooleanMaskIndexing:
    """Boolean masks from comparisons must work for indexing."""

    def test_equal_mask_indexing(self):
        arr = VarUIntArray([10, 20, 30], word_size=8)
        ref = VarUIntArray([10, 99, 30], word_size=8)
        mask = arr == ref  # [True, False, True]
        result = arr[mask]
        np.testing.assert_array_equal(result, [10, 30])

    def test_inverted_mask_indexing(self):
        arr = VarUIntArray([10, 20, 30], word_size=8)
        ref = VarUIntArray([10, 99, 30], word_size=8)
        mask = arr == ref  # [True, False, True]
        result = arr[~mask]
        np.testing.assert_array_equal(result, [20])

    def test_inverted_mask_stays_boolean(self):
        arr = VarUIntArray([1, 2, 3], word_size=8)
        mask = arr == arr  # all True
        inverted = ~mask
        assert inverted.dtype == np.bool_
        np.testing.assert_array_equal(inverted, [False, False, False])

    def test_no_255_index_error(self):
        """Regression: ~all_true_mask must not produce [255, 255, ...] uint8."""
        data = VarUIntArray(list(range(55)), word_size=8)
        mask = data == data  # all True
        inv = ~mask  # must be [False, False, ...], NOT [255, 255, ...]
        assert inv.dtype == np.bool_
        result = data[inv]
        assert len(result) == 0


class TestAssertArrayEqual:
    """numpy.testing.assert_array_equal must work with VarUIntArray."""

    @pytest.mark.parametrize("word_size", [1, 4, 8, 10, 16, 32])
    def test_equal_arrays(self, word_size):
        arr = VarUIntArray([1, 2, 3], word_size=word_size)
        np.testing.assert_array_equal(arr, arr)

    def test_equal_arrays_copy(self):
        a = VarUIntArray([1, 2, 3], word_size=8)
        b = VarUIntArray([1, 2, 3], word_size=8)
        np.testing.assert_array_equal(a, b)

    def test_unequal_arrays_raise(self):
        a = VarUIntArray([1, 2, 3], word_size=8)
        b = VarUIntArray([1, 99, 3], word_size=8)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(a, b)

    def test_large_array(self):
        data = list(range(200))
        a = VarUIntArray(data, word_size=8)
        b = VarUIntArray(data, word_size=8)
        np.testing.assert_array_equal(a, b)


class TestInvalidInputRejected:
    """VarUIntArray must reject non-unsigned-integer input arrays."""

    def test_bool_list_rejected(self):
        with pytest.raises(TypeError, match="requires unsigned integer data"):
            VarUIntArray([True, False, True], word_size=1)

    def test_bool_ndarray_rejected(self):
        with pytest.raises(TypeError, match="requires unsigned integer data"):
            VarUIntArray(np.array([True, False]), word_size=8)

    def test_complex_rejected(self):
        with pytest.raises(TypeError, match="requires unsigned integer data"):
            VarUIntArray(np.array([1 + 2j]), word_size=8)

    def test_string_rejected(self):
        with pytest.raises(TypeError, match="requires unsigned integer data"):
            VarUIntArray(np.array(["a", "b"]), word_size=8)

    def test_signed_int_accepted(self):
        # Plain Python int lists are inferred as signed int by numpy;
        # this is the normal usage pattern and must work.
        arr = VarUIntArray([1, 2, 3], word_size=8)
        np.testing.assert_array_equal(arr, [1, 2, 3])

    def test_unsigned_int_accepted(self):
        arr = VarUIntArray(np.array([1, 2, 3], dtype=np.uint8), word_size=8)
        np.testing.assert_array_equal(arr, [1, 2, 3])
