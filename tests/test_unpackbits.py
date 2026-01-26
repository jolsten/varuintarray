import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from varuintarray.array import (
    VarUIntArray,
)


@given(
    st.integers(min_value=1, max_value=1024),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=2, max_value=24),
)
def test_unpackbits_shape(num_cols: int, num_rows: int, word_size: int):
    # Restrict cases to where num_cols doesn't make range(num_cols) overflow the array dtype
    assume(num_cols < 2**word_size)

    array = VarUIntArray([range(num_cols)] * num_rows, word_size=word_size)
    unpacked = array.unpack()
    assert unpacked.shape[0] == num_rows
    assert unpacked.shape[1] == num_cols * word_size


@pytest.mark.parametrize(
    "word_size, data, result",
    [
        [
            2,
            [0, 1],
            [0, 0, 0, 1],
        ],
        [
            2,
            [[0, 1], [2, 3]],
            [[0, 0, 0, 1], [1, 0, 1, 1]],
        ],
        [4, [[0], [15]], [[0, 0, 0, 0], [1, 1, 1, 1]]],
        [
            10,
            [[0], [1023]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        ],
    ],
)
def test_unpackbits(word_size: int, data: list[list[int]], result: list[list[int]]):
    assert VarUIntArray(data, word_size=word_size).unpack().tolist() == result
