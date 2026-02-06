"""Variable-length unsigned integer arrays for NumPy.

This module provides VarUIntArray, a NumPy ndarray subclass for working with
unsigned integers of arbitrary bit widths (1-64 bits) that don't necessarily
align with standard machine word sizes (8, 16, 32, 64 bits).

The primary purpose is to correctly handle padding bits during bitwise operations
and array manipulations. When you store a 10-bit value in a 16-bit container,
the 6 padding bits need special handling during operations like bitwise inversion.
VarUIntArray manages this automatically.

Key Features:
    - Support for any word size from 1 to 64 bits
    - Automatic selection of appropriate underlying NumPy dtype
    - Correct handling of padding bits in bitwise operations
    - Pack and unpack operations between bit arrays and integer arrays
    - Full integration with NumPy's universal functions (ufuncs)

Functions:
    packbits: Pack bit arrays into VarUIntArray
    unpackbits: Unpack VarUIntArray into bit arrays
    validate_varuintarray: Convert dictionary to VarUIntArray
    serialize_varuintarray: Serialize VarUIntArray to dictionary

Notes:
    - All VarUIntArray instances use big-endian byte order
    - Maximum supported word size is 64 bits
    - Only unsigned integers are supported
    - Padding bits are always set to zero after operations

Examples:
    Basic usage with 10-bit words::

        >>> arr = VarUIntArray([1, 2, 1023], word_size=10)
        >>> arr
        VarUIntArray([  1,   2, 1023], dtype='>u2', word_size=10)
        >>> arr.invert()
        VarUIntArray([1022, 1021,    0], dtype='>u2', word_size=10)

    Packing and unpacking bits::

        >>> bits = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        >>> packed = packbits(bits)
        >>> packed
        VarUIntArray([5, 3], dtype='>u1', word_size=3)
        >>> unpackbits(packed)
        array([[1, 0, 1],
               [0, 1, 1]], dtype=uint8)
"""

from varuintarray.array import VarUIntArray, packbits, unpackbits

__all__ = ["VarUIntArray", "unpackbits", "packbits"]
