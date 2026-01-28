import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

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
