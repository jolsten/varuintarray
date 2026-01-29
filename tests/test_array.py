import pytest

from varuintarray.array import VarUIntArray


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
        array = VarUIntArray(data, word_size=8)
