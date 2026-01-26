import pytest
from hypothesis import given

from tests import strategies as cst
from varuintarray.array import (
    VarUIntArray,
    serialize_varuintarray,
    validate_varuintarray,
)


@pytest.mark.parametrize("word_size", [1, 8, 10, 12, 16])
def test_array_word_size(word_size: int):
    data = range(2**word_size)
    array = VarUIntArray(data, word_size=word_size)
    assert array.word_size == word_size


@given(cst.varuintarrays())
def test_serialize(array: VarUIntArray):
    data = serialize_varuintarray(array)
    assert isinstance(data, dict)
    assert data["word_size"] == array.word_size
    assert data["values"] == array.tolist()


@given(cst.varuintarrays())
def test_roundtrip(array: VarUIntArray):
    data = serialize_varuintarray(array)
    result = validate_varuintarray(data)
    assert isinstance(array, VarUIntArray)
    assert array.word_size == result.word_size
    assert array.tolist() == result.tolist()
