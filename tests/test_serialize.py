from hypothesis import given

from tests import strategies as cst
from varuintarray.array import VarUIntArray


@given(cst.varuintarrays())
def test_serialize(array: VarUIntArray):
    data = array.to_dict()
    assert isinstance(data, dict)
    assert data["word_size"] == array.word_size
    assert data["values"] == array.tolist()


@given(cst.varuintarrays())
def test_roundtrip(array: VarUIntArray):
    data = array.to_dict()
    result = VarUIntArray.from_dict(data)
    assert isinstance(array, VarUIntArray)
    assert array.word_size == result.word_size
    assert array.tolist() == result.tolist()
