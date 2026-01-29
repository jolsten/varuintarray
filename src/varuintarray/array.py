from typing import Any, Callable, Iterable, Mapping

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

        print("abc")

        if context is not None:
            func, args, out_i = context
            # input_args = args[:func.nin]

            if func is np.invert:
                # Ensure the result of np.invert doesn't return pad bits as ones
                result = np.bitwise_and(result.view(np.ndarray), 2**self.word_size - 1)
                result = self.__class__(result, word_size=self.word_size)

        return result

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if func is np.unpackbits:
            array, *_ = args
            if "axis" in kwargs:
                msg = "axis keyword argument not valid when using np.unpackbits() on a VarUIntArray"
                raise ValueError(msg)
            return unpackbits(array)
        return super().__array_function__(func, types, args, kwargs)

    def invert(self) -> np.ndarray:
        """Invert the bits in the array, respecting word_size."""
        return np.invert(self)

    def unpackbits(self) -> np.ndarray:
        return unpackbits(self)

    @classmethod
    def packbits(cls, data: np.ndarray) -> "VarUIntArray":
        return packbits(data)


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


def unpackbits(array: VarUIntArray) -> np.ndarray:
    """Unpack the bits in the array, respecting word_size.

    Omits the pad bits used in the ndarray.
    """
    shape = array.shape
    result = array.view(np.ndarray)

    # Add a dimension such that each word is indexed in the innermost dimension
    result = result.reshape(*shape, 1)

    # Convert to a regular ndarray with 8-bit words
    # this preps it for np.unpackbits
    result = result.view("u1")

    # Unpack bits in each word
    result = np.unpackbits(result, axis=result.ndim - 1)

    # Slice the array keeping everything except the pad bits in the deepest index level
    # e.g. for a 10-bit word packed into a uint16, return the lower 10 bits
    # The implementation here is an N-dimensional generalization of, e.g.
    # 1-d array [:, -array.word_size:]
    # 2-d array [:, :, -array.word_size:]
    slices = (slice(None),) * (result.ndim - 1) + (slice(-array.word_size, None),)
    result = result[slices]

    return result


def packbits(array: np.ndarray) -> VarUIntArray:
    """Pack an array into a `VarUIntArray`.

    The deepest dimension must index each bit within each individual word.

    Inserts pad bits before packing into the appropriate dtype.
    """
    shape = array.shape
    ndim = array.ndim
    word_size = shape[-1]

    # Determine appropriate number of pad bits
    pad = word_size_to_machine_size(word_size) - word_size

    # Dynamically create pad tuples
    # 1. Pad 0 before, 0 after for each dimension except the last
    # 2. Pad N before, 0 after for the last dimension to get to necessary machine word size
    pad = ((0, 0),) * (ndim - 1) + ((pad, 0),)
    result = np.pad(array, pad, mode="constant")

    # Pack padded bits back into words
    result = np.packbits(result, axis=-1)

    # Convert to appropriate uint dtype
    dtype = ">" + word_size_to_dtype(word_size)
    result = result.view(dtype)

    # Drop innermost dimension
    result = result.squeeze(axis=-1)

    return VarUIntArray(result, word_size=word_size)
