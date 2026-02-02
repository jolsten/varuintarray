import numpy as np
import pytest

from varuintarray.array import VarUIntArray
from varuintarray.takeskip import (
    Command,
    Invert,
    Ones,
    Reverse,
    Skip,
    Take,
    Zeros,
    _parse_command,
    takeskip,
)


@pytest.mark.parametrize(
    "s, expected",
    [
        ("s8", [Skip(8)]),
        ("i8", [Invert(8)]),
        ("t4", [Take(4)]),
        ("r8", [Reverse(8)]),
        ("n8", [Ones(8)]),
        ("z8", [Zeros(8)]),
        ("s1 t8 s1", [Skip(1), Take(8), Skip(1)]),
    ],
)
def test_parser(s: str, expected: list[Command]):
    instructions = _parse_command(s)
    assert instructions == expected


def test_takeskip_word_s1t8s1():
    data = np.array(range(256), dtype=">u2") << 1
    mask = np.tile([0, 512], reps=len(data) // 2)
    data = np.bitwise_xor(data, mask)

    array = VarUIntArray(data, word_size=10)
    out = takeskip("s1t8s1", array, mode="word")
    assert out.word_size == 8
    assert out.tolist() == list(range(256))

    NUM_ROWS = 10
    array = VarUIntArray([data] * NUM_ROWS, word_size=10)
    out = takeskip("s1t8s1", array, mode="word")
    assert out.word_size == 8
    assert out.tolist() == [list(range(256))] * NUM_ROWS


example_256 = VarUIntArray(range(256), word_size=8)


def test_s4t4():
    out = takeskip("s4t4", example_256, mode="word")
    assert out.word_size == 4
    assert out.tolist() == (np.arange(256, dtype="u1") % 16).tolist()


def test_t8():
    out = takeskip("t8", example_256, mode="word")
    assert out.word_size == 8
    assert out.tolist() == example_256.tolist()


def test_i8():
    out = takeskip("i8", example_256, mode="word")
    assert out.word_size == 8
    assert out.tolist() == list(reversed(range(256)))


@pytest.mark.parametrize(
    "word_size, value, reverse",
    [
        (8, 0, 0),
        (3, 0b001, 0b100),
        (8, 0x55, 0xAA),
    ],
)
def test_r8(word_size: int, value: int, reverse: int):
    LEN = 10
    array = VarUIntArray([value] * LEN, word_size=word_size)
    out = takeskip(f"r{word_size}", array, mode="word")
    assert out.word_size == word_size
    assert out.tolist() == [reverse] * LEN
