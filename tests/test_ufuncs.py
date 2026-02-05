from typing import Callable

import numpy as np
import pytest

from varuintarray.array import VarUIntArray, _word_size_to_dtype


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


@pytest.mark.parametrize("word_size", [1, 4, 8, 10, 12, 16, 24, 32])
@pytest.mark.parametrize(
    "ufunc, args",
    [
        # Arithmetic
        (np.add, [1]),
        (np.subtract, [1]),
        (np.multiply, [2]),
        (np.divide, [2]),
        (np.floor_divide, [2]),
        (np.power, [2]),
        (np.remainder, [2]),
        (np.mod, [2]),
        (np.fmod, [2]),
        (np.negative, []),
        (np.positive, []),
        (np.abs, []),
        (np.absolute, []),
        # Bitwise operations
        (np.invert, []),
        (np.bitwise_invert, []),
        (np.bitwise_xor, [0xFF]),
        (np.bitwise_and, [0xAA]),
        (np.bitwise_left_shift, [2]),
        (np.bitwise_right_shift, [2]),
    ],
)
def test_ufunc_result_type(word_size: int, ufunc: Callable, args):
    dtype = _word_size_to_dtype(word_size)
    data = np.random.default_rng().integers(0, 2**word_size, size=10, dtype=dtype)
    array = VarUIntArray(data, word_size=word_size)
    result = ufunc(array, *args)
    assert isinstance(result, VarUIntArray)
