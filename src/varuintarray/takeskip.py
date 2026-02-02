import re
from abc import abstractmethod
from typing import Literal, Optional

import numpy as np

from varuintarray.array import VarUIntArray, packbits, unpackbits


class Command:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Command):
            raise TypeError

        if self.value != other.value:
            return False

        if self.__class__ != other.__class__:
            return False

        return True

    @property
    def input_size(self) -> int:
        return self.value

    @property
    @abstractmethod
    def result_size(self) -> int: ...

    @abstractmethod
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]: ...


INSTRUCTION_REGISTRY: dict[str, type[Command]] = {}


def register(token: str):
    def decorator(cls: type[Command]):
        INSTRUCTION_REGISTRY[token] = cls
        return cls

    return decorator


@register("t")
class Take(Command):
    @property
    def result_size(self) -> int:
        return self.value

    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return array


@register("s")
class Skip(Command):
    @property
    def result_size(self) -> int:
        return 0

    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return None


@register("i")
class Invert(Take):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return np.bitwise_xor(array, np.uint8(1))


@register("r")
class Reverse(Take):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return array[..., ::-1]


class Pad(Command):
    @property
    def input_size(self) -> int:
        return 0

    @property
    def result_size(self) -> int:
        return self.value


@register("n")
class Ones(Pad):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return np.ones((*array.shape[0:-1], self.value), dtype="u1")


@register("z")
class Zeros(Pad):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return np.zeros((*array.shape[0:-1], self.value), dtype="u1")


INSTRUCTION_TOKEN_RE = re.compile(r"\s*([a-z])(\d+)", re.IGNORECASE)


def _parse_command(s: str) -> list[Command]:
    pos = 0
    commands = []
    while True:
        # Search for next token, starting after last token
        m = INSTRUCTION_TOKEN_RE.match(s, pos)

        if not m:
            # If remaining text is only whitespace, we're done
            if s[pos:].strip() == "":
                return commands

            # Otherwise, report precise failure
            raise ValueError(
                f"Invalid token starting at position {pos}: {s[pos : pos + 10]!r}"
            )

        # Set new end position
        pos = m.end()

        # Convert text token into components, construct Command object
        type_, value = m.groups()
        type_ = type_.lower()
        value = int(value)

        if type_ not in INSTRUCTION_REGISTRY:
            msg = f"Invalid token starting at position {pos}: {s[pos : pos + 10]!r}"
            raise ValueError(msg)

        cls = INSTRUCTION_REGISTRY[type_]
        command = cls(value)
        commands.append(command)


def takeskip(
    command: str,
    array: VarUIntArray,
    *,
    mode: Literal["word", "row"],
    word_size: Optional[int] = None,
) -> VarUIntArray:
    """Perform a take-skip style operation.

    Take-Skip operations are a syntax of selecting (and potentially manipulating) bits from
    a sequence of bits.

    The command string is made up of individual command elements. Each element is
    a string containing a letter and a number. The letter represents the operation, and the
    number is the number of bits for the operation.

    Valid operations include:
        t: take    - take bits (no manipulation)
        s: skip    - skip bits
        r: reverse - reverse the order of bits
        i: invert  - invert
        o: ones    - pad with 1
        z: zeros   - pad with 0

    For example:
        * s4t4 - skip 4 bits, take 4 bits
        * t4r4 - take 4 bits, reverse 4 bits

    Note:
        * Commands are case insensitive and ignore whitespace.
        * In "word" mode, the command string determines the resulting array.word_size
        * In "row" mode, the command string must result in a total length that is a multiple
        of the output array.word_size. Pad bits can be used to ensure the result length is valid
        if necessary.

    Args:
        command: The string expressing the operation to perform.
        array: The target of the operation `VarUIntArray`.
        mode: Whether to execute the operation on each "word" or "row".
        word_size: For "row" mode, word_size can specify a new output word size.

    Returns:
        The resulting array.

    Raises:
        ValueError: If there is an error in the take-skip command syntax.
    """
    if word_size is None:
        word_size = array.word_size

    commands = _parse_command(command)
    unpacked = unpackbits(array)
    result_size = sum(i.result_size for i in commands)

    if result_size == 0:
        msg = "Command would result in output with word_size 0."
        raise ValueError(msg)

    if mode == "word":
        # Reshape into a 2-d array where dimension 2 has length == array.word_size
        unpacked = unpacked.reshape(-1, array.word_size)
    elif mode == "row":
        if array.shape[-1] * word_size != result_size:
            msg = (
                f"Command does not result in a size with length equal to an integer "
                f"multiple of the input array (word_size={word_size}), try adding "
                "pad bits or changing the desired output word_size."
            )
            raise ValueError(msg)
        # Reshape into a 2-d array where dimension 2 has length == array.word_size * num elements per row
        unpacked = unpacked.reshape(-1, array.shape[-1] * array.word_size)
    else:
        msg = f"""valid mode choices include: "word" and "row", not {mode!r}"""
        raise ValueError(msg)

    result = np.zeros([*unpacked.shape[:-1], result_size], dtype="u1")

    in_ptr = 0
    out_ptr = 0
    for cmd in commands:
        manipulated_bits = cmd(unpacked[:, in_ptr : in_ptr + cmd.input_size])
        if manipulated_bits is not None:
            result[:, out_ptr : out_ptr + cmd.result_size] = manipulated_bits

        in_ptr += cmd.input_size
        out_ptr += cmd.result_size

    if mode == "word":
        result = result.reshape(*array.shape, result_size)

    elif mode == "row":
        result = result.reshape(*array.shape[0:-1], -1, word_size)

    return packbits(result)
