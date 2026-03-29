import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tests.strategies import varuintarrays
from varuintarray.array import (
    VarUIntArray,
    unpackbits,
)

MAX_WORDSIZE = 64


@given(
    st.integers(min_value=2, max_value=MAX_WORDSIZE),
)
def test_unpackbits_0d_shape(word_size: int):
    array = VarUIntArray(0, word_size=word_size)
    unpacked = unpackbits(array)
    assert unpacked.shape == (word_size,)


@given(
    st.integers(min_value=1, max_value=1024),
    st.integers(min_value=2, max_value=MAX_WORDSIZE),
)
def test_unpackbits_1d_shape(num_cols: int, word_size: int):
    # Restrict cases to where num_cols doesn't make range(num_cols) overflow the array dtype
    assume(num_cols < 2**word_size)

    array = VarUIntArray(range(num_cols), word_size=word_size)
    unpacked = unpackbits(array)

    assert array.ndim == unpacked.ndim - 1
    assert unpacked.shape == (num_cols, word_size)


@given(
    st.integers(min_value=1, max_value=1024),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=2, max_value=MAX_WORDSIZE),
)
def test_unpackbits_2d_shape(num_cols: int, num_rows: int, word_size: int):
    # Restrict cases to where num_cols doesn't make range(num_cols) overflow the array dtype
    assume(num_cols < 2**word_size)

    array = VarUIntArray([range(num_cols)] * num_rows, word_size=word_size)
    unpacked = unpackbits(array)

    assert array.ndim == unpacked.ndim - 1
    assert unpacked.shape == (num_rows, num_cols, word_size)


@pytest.mark.parametrize(
    "word_size, data, result",
    [
        # 0-d (scalar) test cases
        (4, 0b0000, [0, 0, 0, 0]),
        (8, 0b00000000, [0, 0, 0, 0, 0, 0, 0, 0]),
        (8, 0b01010101, [0, 1, 0, 1, 0, 1, 0, 1]),
        (8, 0b10101010, [1, 0, 1, 0, 1, 0, 1, 0]),
        # 1-d array test cases
        (
            2,
            [0b00, 0b01],
            [[0, 0], [0, 1]],
        ),
        # 2-d array test cases
        (
            2,
            [[0, 1], [2, 3]],
            [[[0, 0], [0, 1]], [[1, 0], [1, 1]]],
        ),
        (4, [[0], [15]], [[[0, 0, 0, 0]], [[1, 1, 1, 1]]]),
        (
            10,
            [[0], [1023]],
            [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],
        ),
    ],
)
class TestUnpackBits:
    def test_unpackbits(
        self, word_size: int, data: list[list[int]], result: list[list[int]]
    ):
        array = VarUIntArray(data, word_size=word_size)
        unpacked = unpackbits(array).tolist()
        assert unpacked == result

    def test_roundtrip(
        self, word_size: int, data: list[list[int]], result: list[list[int]]
    ):
        array = VarUIntArray(data, word_size=word_size)
        unpacked = np.unpackbits(array)
        packed = VarUIntArray.packbits(unpacked)
        assert isinstance(packed, VarUIntArray)
        assert array.word_size == packed.word_size
        assert array.tolist() == packed.tolist()


@pytest.mark.parametrize(
    "word_size, data, expected",
    [
        # 3-bit: 5 (101) -> 101 (5), 3 (011) -> 110 (6)
        (3, [5, 3], [5, 6]),
        # 8-bit: 1 (00000001) -> 10000000 (128)
        (8, [1], [128]),
        # 4-bit: 0 stays 0, 15 (1111) stays 15
        (4, [0, 15], [0, 15]),
        # 4-bit: 1 (0001) -> 1000 (8)
        (4, [1], [8]),
        # 1-bit: only one bit, reversal is identity
        (1, [0, 1], [0, 1]),
        # 2-d array
        (3, [[5, 3], [1, 7]], [[5, 6], [4, 7]]),
    ],
)
class TestReverseBits:
    def test_reversebits(
        self, word_size: int, data: list, expected: list
    ):
        arr = VarUIntArray(data, word_size=word_size)
        result = arr.reversebits()
        assert isinstance(result, VarUIntArray)
        assert result.word_size == word_size
        assert result.tolist() == expected

    def test_reversebits_double_reversal(
        self, word_size: int, data: list, expected: list
    ):
        """Reversing bits twice should return the original values."""
        arr = VarUIntArray(data, word_size=word_size)
        assert arr.reversebits().reversebits().tolist() == arr.tolist()


@given(varuintarrays())
def test_reversebits_involution(arr: VarUIntArray):
    """Property: reversing bits twice is always the identity."""
    # Mask to valid word_size range since the strategy may generate
    # values that use bits beyond word_size.
    mask = np.array(2**arr.word_size - 1, dtype=arr.dtype)
    arr = VarUIntArray(np.bitwise_and(arr, mask), word_size=arr.word_size)
    result = arr.reversebits().reversebits()
    np.testing.assert_array_equal(result, arr)
