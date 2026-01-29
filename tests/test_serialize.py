from hypothesis import given

from tests import strategies as cst
from varuintarray.array import (
    VarUIntArray,
    serialize_varuintarray,
    validate_varuintarray,
)


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
