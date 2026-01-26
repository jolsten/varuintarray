import numpy as np
import pytest

from varuintarray.array import VarUIntArray


@pytest.mark.parametrize(
    "word_size, value, inverted",
    [
        (4, 0x0, 0xF),
        (4, 0x5, 0xA),
        (4, 0xA, 0x5),
        (4, 0xF, 0x0),
        (8, 0x00, 0xFF),
        (8, 0x55, 0xAA),
        (8, 0xAA, 0x55),
        (8, 0xFF, 0x00),
        (12, 0x000, 0xFFF),
        (12, 0x555, 0xAAA),
        (12, 0xAAA, 0x555),
        (12, 0xFFF, 0x000),
        (16, 0x05AF, 0xFA50),
    ],
)
class TestInvert:
    def test_invert_function(self, word_size: int, value: int, inverted: int):
        array = VarUIntArray(np.full((100, 50), value, dtype=int), word_size=word_size)
        result = np.invert(array)
        assert [inverted == r for r in result.flatten()]

    def test_invert_method(self, word_size: int, value: int, inverted: int):
        array = VarUIntArray(np.full((100, 50), value, dtype=int), word_size=word_size)
        result = array.invert()
        assert [inverted == r for r in result.flatten()]
