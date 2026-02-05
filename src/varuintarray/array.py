from typing import Any, Callable, Iterable, Mapping

import numpy as np
import numpy.typing as npt


def __word_size_to_machine_size(size: int) -> int:
    """Find the smallest machine size that contains the input size.

    Determine the machine word size (8, 16, 32, 64, 128) appropriate for an
    arbitrary input word size.

    Args:
        size: Word size in bits. Must be a positive integer <= 64.

    Returns:
        The smallest standard machine word size (8, 16, 32, or 64) that can
        contain the input size.

    Raises:
        ValueError: If size is <= 0 or > 64.

    Examples:
        >>> __word_size_to_machine_size(10)
        16
        >>> __word_size_to_machine_size(5)
        8
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


def __word_size_to_dtype(size: int) -> str:
    """Get numpy dtype string for a given word size.

    Converts a word size in bits to the corresponding numpy unsigned integer
    dtype string (e.g., 'u1', 'u2', 'u4', 'u8').

    Args:
        size: Word size in bits.

    Returns:
        A numpy dtype string in the format 'uN' where N is the number of bytes
        needed to store the word size.

    Examples:
        >>> __word_size_to_dtype(10)
        'u2'
        >>> __word_size_to_dtype(5)
        'u1'
    """
    return f"u{__word_size_to_machine_size(size) // 8}"


class VarUIntArray(np.ndarray):
    """Variable-length Unsigned Integer Array.

    This subclass of `np.ndarray` is intended to make it easier to use numpy ufuncs
    on an ndarray while respecting bits per word values that do not fit neatly into
    standard machine sizes (8, 16, 32, etc.). In other words, this ensures some
    ufuncs handle pad bits correctly.

    Attributes:
        input_array: The input array data.
        word_size: Number of significant bits per word (excludes padding bits).

    Examples:
        >>> arr = VarUIntArray([1, 2, 3], word_size=10)
        >>> arr.word_size
        10
    """

    input_array: npt.ArrayLike
    word_size: int

    def __new__(cls, input_array: npt.ArrayLike, word_size: int):
        """Create a new VarUIntArray instance.

        Args:
            input_array: Array-like data to be converted.
            word_size: Number of significant bits per word.

        Returns:
            A new VarUIntArray instance.
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        dtype = ">" + __word_size_to_dtype(word_size)

        obj = np.asarray(input_array, dtype=dtype).view(cls)

        # add the new attribute to the created instance
        obj.word_size = word_size

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        """Finalize the array after creation.

        This method is called whenever the system allocates a new array from obj.
        Used to ensure word_size attribute is properly propagated.

        Args:
            obj: The object from which the array is being finalized.
        """
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.word_size = getattr(obj, "word_size", 0)

    def __repr__(self) -> str:
        """Return a string representation of the VarUIntArray.

        Returns:
            A string representation that includes the word_size attribute.
        """
        base = super().__repr__()

        if hasattr(self, "word_size"):
            return base[:-1] + f", word_size={self.word_size})"
        return base

    def __array_wrap__(self, obj, context=None, return_scalar=False):  # noqa: FBT002
        """Wrap the result of a ufunc operation.

        Special handling for bitwise operations to ensure pad bits are
        handled correctly.

        Args:
            obj: The result object from the ufunc.
            context: The ufunc context (function, arguments, output index).
            return_scalar: Whether to return a scalar.

        Returns:
            The wrapped result, with proper handling of pad bits for
            bitwise operations.
        """
        if obj is self:  # for in-place operations
            result = obj
        else:
            result = obj.view(type(self))

        result = super().__array_wrap__(obj, context, return_scalar)

        if context is not None:
            func, args, out_i = context
            # input_args = args[:func.nin]

            if func in (np.invert, np.bitwise_invert):
                # Ensure the result of np.invert doesn't return pad bits as ones
                result = np.bitwise_and(result.view(np.ndarray), 2**self.word_size - 1)
                return self.__class__(result, word_size=self.word_size)

        return result

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Handle numpy array function protocol.

        Provides custom implementation for np.unpackbits.

        Args:
            func: The numpy function being called.
            types: The types involved in the function call.
            args: Positional arguments to the function.
            kwargs: Keyword arguments to the function.

        Returns:
            The result of the function call.

        Raises:
            ValueError: If np.unpackbits is called with an axis argument.
        """
        if func is np.unpackbits:
            array, *_ = args
            if "axis" in kwargs:
                msg = "axis keyword argument not valid when using np.unpackbits() on a VarUIntArray"
                raise TypeError(msg)
            return unpackbits(array)
        return super().__array_function__(func, types, args, kwargs)

    def invert(self) -> np.ndarray:
        """Invert the bits in the array, respecting word_size.

        Returns:
            A new array with all bits inverted, excluding pad bits.
        """
        return np.invert(self)

    def unpackbits(self) -> np.ndarray:
        """Unpack the bits in each word, respecting word_size.

        Returns:
            An array with unpacked bits, excluding pad bits. The result has
            one additional dimension compared to the input.
        """
        return unpackbits(self)

    @classmethod
    def packbits(cls, data: np.ndarray) -> "VarUIntArray":
        """Pack an np.ndarray into a VarUIntArray.

        Args:
            data: Array to pack. The last dimension contains bits for each word.

        Returns:
            A VarUIntArray with packed bits.
        """
        return packbits(data)


def validate_varuintarray(data) -> VarUIntArray:
    """Validate and convert data to a VarUIntArray.

    Args:
        data: Either a VarUIntArray instance or a dictionary containing
            'values' and 'word_size' keys.

    Returns:
        A validated VarUIntArray instance.

    Raises:
        TypeError: If data cannot be converted to a VarUIntArray.

    Examples:
        >>> arr = VarUIntArray([1, 2, 3], word_size=10)
        >>> validate_varuintarray(arr) is arr
        True
        >>> validate_varuintarray({"values": [1, 2, 3], "word_size": 10})
        VarUIntArray([1, 2, 3], dtype='>u2', word_size=10)
    """
    if isinstance(data, VarUIntArray):
        return data

    if isinstance(data, dict):
        array = data["values"]
        word_size = data["word_size"]
        return VarUIntArray(array, word_size=word_size)

    msg = f"Cannot convert {data!r} to VarUIntArray"
    raise TypeError(msg)


def serialize_varuintarray(data: VarUIntArray) -> dict[str, Any]:
    """Serialize a VarUIntArray to a dictionary.

    Converts a VarUIntArray to a JSON-serializable dictionary format.

    Args:
        data: The VarUIntArray to serialize.

    Returns:
        A dictionary containing 'word_size' and 'values' keys.

    Examples:
        >>> arr = VarUIntArray([1, 2, 3], word_size=10)
        >>> serialize_varuintarray(arr)
        {'word_size': 10, 'values': [1, 2, 3]}
    """
    return {
        "word_size": data.word_size,
        "values": data.tolist(),
    }


def unpackbits(array: VarUIntArray) -> np.ndarray:
    """Unpack the bits in the array, respecting word_size.

    Works like `np.unpackbits`, but uses word_size to exclude padding bits.

    The resulting `np.ndarray` will have one additional dimension. The new
    (innermost) dimension will have size equal to array.word_size.

    Args:
        array: The VarUIntArray to unpack.

    Returns:
        An ndarray with unpacked bits. The result has one additional dimension
        compared to the input, where the innermost dimension contains the
        unpacked bits (size = array.word_size).

    Notes:
        This replicates the functionality of `np.unpackbits` but uses the
        word_size attribute to omit the pad bits used in the underlying
        numpy dtype.

        Additionally, using the innermost dimension removes ambiguity about
        the VarUIntArray word_size.

    Examples:
        >>> arr = VarUIntArray([5], word_size=3)
        >>> unpackbits(arr)
        array([[1, 0, 1]], dtype=uint8)
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
    """Pack an np.ndarray into a VarUIntArray.

    This works for N-dimensional arrays, but the last (innermost) dimension
    must contain the bits within each word. The word_size attribute is
    determined by the size of the last dimension (i.e. array.shape[-1]).

    The resulting `VarUIntArray` will have ndim one less than the input array.

    Args:
        array: The array containing bits to pack. Must have dtype np.uint8
            and contain only zeros and ones. The last dimension contains
            the bits for each word.

    Returns:
        A VarUIntArray with packed bits. The result has one fewer dimension
        than the input, with word_size set to the size of the input's last
        dimension.

    Examples:
        >>> bits = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        >>> packbits(bits)
        VarUIntArray([5, 3], dtype='>u1', word_size=3)
    """
    shape = array.shape
    ndim = array.ndim
    word_size = shape[-1]

    # Determine appropriate number of pad bits
    pad = __word_size_to_machine_size(word_size) - word_size

    # Dynamically create pad tuples
    # 1. Pad 0 before, 0 after for each dimension except the last
    # 2. Pad N before, 0 after for the last dimension to get to necessary machine word size
    pad = ((0, 0),) * (ndim - 1) + ((pad, 0),)
    result = np.pad(array, pad, mode="constant")

    # Pack padded bits back into words
    result = np.packbits(result, axis=-1)

    # Convert to appropriate uint dtype
    dtype = ">" + __word_size_to_dtype(word_size)
    result = result.view(dtype)

    # Drop innermost dimension
    result = result.squeeze(axis=-1)

    return VarUIntArray(result, word_size=word_size)
