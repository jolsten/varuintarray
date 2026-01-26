import pytest

from varuintarray.array import word_size_to_dtype, word_size_to_machine_size


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
