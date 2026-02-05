import pathlib
from abc import abstractmethod
from typing import Any, Literal, Optional, Union

import numpy as np
from lark import Lark, Transformer, v_args

from varuintarray.array import VarUIntArray, packbits, unpackbits

grammar = (pathlib.Path(__file__).parent / "takeskip.lark").read_text()
command_parser = Lark(grammar, parser="earley")


class Command:
    def __init__(self, value: Any) -> None:
        self.value = value

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
        """The consumed number of input bits."""
        return self.value

    @property
    @abstractmethod
    def result_size(self) -> int:
        """The output number of bits."""
        ...

    @abstractmethod
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]: ...


class Take(Command):
    def __init__(self, value: int) -> None:
        self.value = int(value)

    @property
    def result_size(self) -> int:
        return self.value

    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return array


class Skip(Command):
    def __init__(self, value: int) -> None:
        self.value = int(value)

    @property
    def result_size(self) -> int:
        return 0

    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return None


class Invert(Take):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return np.bitwise_xor(array, np.uint8(1))


class Reverse(Take):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return array[..., ::-1]


class Backup(Command):
    def __init__(self, value: Any) -> None:
        self.value = int(value)

    @property
    def input_size(self) -> int:
        return -self.value

    @property
    def result_size(self) -> int:
        return 0


class Pad(Command):
    @property
    def input_size(self) -> int:
        return 0

    @property
    def result_size(self) -> int:
        return self.value


class Ones(Pad):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return np.ones((*array.shape[0:-1], self.value), dtype="u1")


class Zeros(Pad):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return np.zeros((*array.shape[0:-1], self.value), dtype="u1")


class Data(Pad):
    def __init__(self, value: str) -> None:
        self.value = str(value)

    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError


class Permute(Command):
    def __init__(self, args: list[Union[int, tuple[int, int]]]) -> None:
        results = []
        for item in args:
            if isinstance(item, np.ndarray):
                results.append(item)
            elif isinstance(item, int):
                idx = np.array([item], dtype="int64")
                results.append(idx)
            else:
                raise TypeError
        self.value = np.concatenate(results)

    @property
    def input_size(self) -> int:
        return max(self.value) + 1

    @property
    def result_size(self) -> int:
        return len(self.value)

    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return array[..., self.value]


def one_based_range_to_indices(start, end):
    """
    Convert one-based range to zero-based indices.

    Args:
        start: One-based start position (integer)
        end: One-based end position (integer)

    Returns:
        List of zero-based indices

    Examples:
        1-4 -> [0, 1, 2, 3]
        4-1 -> [3, 2, 1, 0]
        5-5 -> [4]
    """
    # Convert to zero-based
    start_idx = start - 1
    end_idx = end - 1

    # Determine step direction
    step = 1 if start <= end else -1
    return np.arange(start_idx, end_idx + step, step)


class CommandParser(Transformer):
    @v_args(inline=True)
    def integer(self, s: str):
        return int(s)

    def flatten(self, args):
        result = []
        for item in args:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

    def repeat(self, args):
        print(repr(args))
        *commands, n = args
        return commands * n

    @v_args(inline=True)
    def take(self, n: int) -> Take:
        return Take(n)

    @v_args(inline=True)
    def skip(self, n: int) -> Skip:
        return Skip(n)

    @v_args(inline=True)
    def invert(self, n: int) -> Invert:
        return Invert(n)

    @v_args(inline=True)
    def reverse(self, n: int) -> Reverse:
        return Reverse(n)

    @v_args(inline=True)
    def zero_pad(self, n: int) -> Zeros:
        return Zeros(n)

    @v_args(inline=True)
    def one_pad(self, n: int) -> Ones:
        return Ones(n)

    @v_args(inline=True)
    def data_pad(self, s: str) -> Data:
        return Data(s)

    @v_args(inline=True)
    def range(self, a, b) -> tuple[int, int]:
        return one_based_range_to_indices(a, b)

    @v_args(inline=True)
    def csv(self, first, *rest) -> list[int]:
        result = [first]
        for x in rest:
            if isinstance(x, int):
                # convert to zero-based
                idx = np.array([x - 1], dtype="int64")
                result.append(idx)
            else:
                result.append(x)
        return result

    @v_args(inline=True)
    def permute(self, args) -> Permute:
        return Permute(args)


def parse_command(s: str) -> list[Command]:
    tree = command_parser.parse(s)
    return CommandParser().transform(tree)


def takeskip(
    command: str,
    array: VarUIntArray,
    *,
    mode: Literal["word", "row"],
    remnant: Literal["remove", "keep", "pad"] = "remove",
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

    if remnant not in [None, "remove", "keep", "pad"]:
        msg = "invalid remnant argument"
        raise ValueError(msg)

    commands = parse_command(command)
    unpacked = unpackbits(array)
    result_size = sum(i.result_size for i in commands)

    if result_size == 0:
        msg = "Command would result in output with word_size 0."
        raise ValueError(msg)

    if mode == "word":
        # Input length in bits is the word size
        input_size = array.word_size
    elif mode == "row":
        # Input length in bits is the word size * number of elements in final dimension of input array
        input_size = array.word_size * array.shape[-1]
    else:
        msg = f"""valid mode choices include: "word" and "row", not {mode!r}"""
        raise ValueError(msg)

    # Reshape into one input length per row
    unpacked = unpacked.reshape(-1, input_size)

    # Determine the command's resulting output length in bits
    if remnant == "remove":
        output_size = result_size
    else:
        # If "keep" or "pad", ensure the output size is the input size, plus any additional bits
        # added by backing up in the stream
        delta_size = sum([c.value for c in commands if isinstance(c, Backup)])
        output_size = input_size + delta_size
    result = np.zeros([*unpacked.shape[:-1], output_size], dtype="u1")

    in_ptr = 0
    out_ptr = 0
    for cmd in commands:
        manipulated_bits = cmd(unpacked[:, in_ptr : in_ptr + cmd.input_size])
        if manipulated_bits is not None:
            result[:, out_ptr : out_ptr + cmd.result_size] = manipulated_bits

        in_ptr += cmd.input_size
        out_ptr += cmd.result_size

    if remnant == "keep":
        result[:, out_ptr:] = unpacked[:, out_ptr:]

    if mode == "word":
        result = result.reshape(*array.shape, result_size)
    elif mode == "row":
        result = result.reshape(*array.shape[0:-1], -1, result_size)

    return packbits(result)
