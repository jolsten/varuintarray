from typing import Any

import numpy as np
import numpy.typing as npt


def word_size_to_machine_size(size: int) -> int:
    """Find the smallest machine size that contains the input size.

    Determine the machine word size (8, 16, 32, 64, 128) appropriate for an
    arbitrary input word size.

    Arguments:
        size : word size in bits
    """
    if size <= 0:
        msg = f"bit size {size!r} cannot be negative"
        raise ValueError(msg)
    elif size <= 8:  # noqa: PLR2004
        return 8
    elif size <= 16:  # noqa: PLR2004
        return 16
    elif size <= 32:  # noqa: PLR2004
        return 32
    elif size <= 64:  # noqa: PLR2004
        return 64
    msg = f"bit size {size!r} must be <= 64"
    raise ValueError(msg)


def word_size_to_dtype(size: int) -> str:
    """Get `np.dtype` string for a given word size.

    Arguments:
        size : word size in bits
    """
    return f"u{word_size_to_machine_size(size) // 8}"


class VarUIntArray(np.ndarray):
    """Variable-length Unsigned Integer Array.

    This subclass of `np.ndarray` is intended to make it easier to use numpy ufuncs
    on an ndarray while respecting bits per word values that do not fit neatly into
    standard machine sizes (8, 16, 32, etc.). In other words, this ensures some
    ufuncs handle pad bits correctly.
    """

    input_array: npt.ArrayLike
    word_size: int

    def __new__(cls, input_array: npt.ArrayLike, word_size: int):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        dtype = ">" + word_size_to_dtype(word_size)

        obj = np.asarray(input_array, dtype=dtype).view(cls)

        # add the new attribute to the created instance
        obj.word_size = word_size

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.word_size = getattr(obj, "word_size", 0)

    def __repr__(self) -> str:
        base = super().__repr__()

        if hasattr(self, "word_size"):
            return base[:-1] + f", word_size={self.word_size})"
        return base

    def __array_wrap__(self, obj, context=None, return_scalar=False):  # noqa: FBT002
        if obj is self:  # for in-place operations
            result = obj
        else:
            result = obj.view(type(self))

        result = super().__array_wrap__(obj, context, return_scalar)

        if context is not None:
            func, args, out_i = context
            # input_args = args[:func.nin]

            if func is np.invert:
                # Ensure the result of np.invert doesn't return pad bits as ones
                result = np.bitwise_and(result.view(np.ndarray), 2**self.word_size - 1)
                result = self.__class__(result, word_size=self.word_size)
            # elif func is np.ascontiguousarray:
            #     # TODO: Is this necessary?
            #     result = np.ascontiguousarray(result)
            #     result = self.__class__(result, word_size=self.word_size)

        return result

    def invert(self) -> np.ndarray:
        """Invert the bits in the array, respecting word_size."""
        return np.invert(self)

    def unpack(self) -> np.ndarray:
        """Unpack the bits in the array, respecting word_size.

        Omits the pad bits used in the ndarray.
        """
        # Convert to a regular ndarray with 8-bit words
        # this preps it for np.unpackbits
        result = self.view(np.ndarray).view("u1")

        # Unpack bits in each row
        result = np.unpackbits(result, axis=1)

        # Slice the "real" as indicated by word_size
        # e.g. for a 10-bit word packed into a uint16, return the lower 10 bits
        result = result.reshape(-1, self.itemsize * 8)[:, -self.word_size :]

        # Return shape to proper number of rows
        result = result.reshape(self.shape[0], self.shape[1] * self.word_size)

        return result

    @classmethod
    def pack(cls, data: np.ndarray, word_size: int) -> "VarUIntArray":
        """Pack an array into words of a specific size."""
        data = np.asarray(data)

        rows = data.shape[0]

        # Reshape to one word per row
        result = data.reshape(-1, word_size)

        # Determine appropriate number of pad bits
        mach_size = word_size_to_machine_size(word_size)
        pad_size = mach_size - word_size

        # Insert pad bits on left side
        result = np.pad(result, ((0, 0), (pad_size, 0)), mode="constant")

        # Convert padded binary into bytes
        result = np.packbits(result)

        # View packed bytes as appropriate uint dtype
        dtype = ">" + word_size_to_dtype(word_size)
        result = result.view(dtype)

        # Reshape back to original number of rows
        result = result.reshape(rows, -1)

        return VarUIntArray(result, word_size=word_size)


def validate_varuintarray(data) -> VarUIntArray:
    if isinstance(data, VarUIntArray):
        return data

    if isinstance(data, dict):
        array = data["values"]
        word_size = data["word_size"]
        return VarUIntArray(array, word_size=word_size)

    msg = f"Cannot convert {data!r} to VarUIntArray"
    raise TypeError(msg)


def serialize_varuintarray(data: VarUIntArray) -> dict[str, Any]:
    return {
        "word_size": data.word_size,
        "values": data.tolist(),
    }
