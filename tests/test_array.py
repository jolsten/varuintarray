import pytest

from varuintarray.array import (
    VarUIntArray,
    word_size_to_dtype,
    word_size_to_machine_size,
)


@pytest.mark.parametrize(
    "word_size, mach_size",
    [
        (4, 8),
        (8, 8),
        (9, 16),
        (24, 32),
    ],
)
def test_word_size_to_machine_size(word_size: int, mach_size: int):
    assert word_size_to_machine_size(word_size) == mach_size


@pytest.mark.parametrize(
    "word_size, dtype",
    [
        (4, "u1"),
        (8, "u1"),
        (10, "u2"),
    ],
)
def test_word_size_to_dtype(word_size: int, dtype: str):
    result_dtype = word_size_to_dtype(word_size)
    assert dtype == result_dtype


@pytest.mark.parametrize("word_size", [1, 8, 10, 12, 16])
def test_array_word_size(word_size: int):
    data = range(2**word_size)
    array = VarUIntArray(data, word_size=word_size)
    assert array.word_size == word_size


def test_overflow():
    # Values that should overflow based on word_size will not overflow if they
    # are being stored in a uint that is large enough to hold them. Rather
    # than check values at class instantiation... garbage in, garbage out?
    data = range(1024)

    with pytest.raises(OverflowError):
        VarUIntArray(data, word_size=8)
