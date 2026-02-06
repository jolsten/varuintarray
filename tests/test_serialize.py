from hypothesis import given

from tests import strategies as cst
from varuintarray.array import (
    VarUIntArray,
    serialize,
    validate,
)


@given(cst.varuintarrays())
def test_serialize(array: VarUIntArray):
    data = serialize(array)
    assert isinstance(data, dict)
    assert data["word_size"] == array.word_size
    assert data["values"] == array.tolist()


@given(cst.varuintarrays())
def test_roundtrip(array: VarUIntArray):
    data = serialize(array)
    result = validate(data)
    assert isinstance(array, VarUIntArray)
    assert array.word_size == result.word_size
    assert array.tolist() == result.tolist()
