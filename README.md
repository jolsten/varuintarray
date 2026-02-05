# VarUIntArray

A NumPy subclass for working with variable-length unsigned integers that don't fit standard machine word sizes.

## Overview

`VarUIntArray` extends `numpy.ndarray` to handle arbitrary bit-width unsigned integers (e.g., 3-bit, 10-bit, 12-bit) while correctly managing padding bits when using NumPy's universal functions (ufuncs). This is particularly useful when working with:

- Custom binary formats with non-standard word sizes
- Packed bit arrays where words don't align to 8, 16, 32, or 64 bits
- Data structures that require precise bit-width control

## Key Features

- **Arbitrary Word Sizes**: Support for any word size from 1 to 64 bits
- **Automatic Padding Management**: Correctly handles padding bits in bitwise operations
- **NumPy Integration**: Works seamlessly with NumPy ufuncs and array operations
- **Pack/Unpack Operations**: Convert between bit arrays and packed integer arrays

## Installation

This module can be installed from PyPi:

```bash
pip install varuintarray
```

## Quick Start

```python
import numpy as np
from varuintarray import VarUIntArray

# Create a VarUIntArray with 10-bit words
arr = VarUIntArray([1, 2, 1023], word_size=10)
print(arr)
# VarUIntArray([  1,   2, 1023], dtype='>u2', word_size=10)

# Bitwise operations respect word_size
inverted = arr.invert()
print(inverted)
# VarUIntArray([1022, 1021,    0], dtype='>u2', word_size=10)

# Unpack to individual bits
bits = arr.unpackbits()
print(bits.shape)
# (3, 10)  # 3 words, 10 bits each

# Pack bits back into words
packed = VarUIntArray.packbits(bits)
print(packed)
# VarUIntArray([  1,   2, 1023], dtype='>u2', word_size=10)
```

## Core Concepts

### Word Size vs Machine Size

Standard computers work with word sizes of 8, 16, 32, or 64 bits. When you need a 10-bit word, it must be stored in a 16-bit container, leaving 6 padding bits unused. `VarUIntArray` automatically:

1. Selects the appropriate machine word size (8, 16, 32, or 64 bits)
2. Tracks the actual word size you care about
3. Ensures padding bits are handled correctly in operations

### Padding Bit Handling

The most important feature is correct handling of padding bits during bitwise operations. For example:

```python
# 3-bit word stored in 8-bit container
arr = VarUIntArray([5], word_size=3)  # Binary: 101

# Standard NumPy invert would give 11111010 (250)
# VarUIntArray.invert() gives 010 (2) - correct for 3-bit word
inverted = arr.invert()
print(inverted[0])  # 2, not 250
```

## API Reference

### VarUIntArray Class

#### Constructor

```python
VarUIntArray(input_array, word_size)
```

**Parameters:**
- `input_array`: Array-like data to convert
- `word_size`: Number of significant bits per word (1-64)

#### Methods

- `invert()`: Bitwise invert respecting word_size
- `unpackbits()`: Unpack to individual bits (adds one dimension)
- `packbits(data)`: Class method to pack bit array into VarUIntArray

#### Attributes

- `word_size`: Number of significant bits per word

### Functions

#### `unpackbits(array)`

Unpack a VarUIntArray into individual bits, excluding padding.

```python
arr = VarUIntArray([5, 3], word_size=3)
bits = unpackbits(arr)
# array([[1, 0, 1],
#        [0, 1, 1]], dtype=uint8)
```

**Parameters:**
- `array`: VarUIntArray to unpack

**Returns:** ndarray with shape `(*original_shape, word_size)`

#### `packbits(array)`

Pack a bit array into a VarUIntArray.

```python
bits = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
arr = packbits(bits)
# VarUIntArray([5, 3], dtype='>u1', word_size=3)
```

**Parameters:**
- `array`: ndarray of uint8 containing 0s and 1s, where the last dimension contains bits for each word

**Returns:** VarUIntArray with one fewer dimension

#### `validate_varuintarray(data)`

Convert various formats to VarUIntArray.

```python
# From VarUIntArray (pass-through)
arr = VarUIntArray([1, 2, 3], word_size=10)
validate_varuintarray(arr)  # Returns arr

# From dictionary
validate_varuintarray({'values': [1, 2, 3], 'word_size': 10})
```

#### `serialize_varuintarray(data)`

Serialize VarUIntArray to JSON-compatible dictionary.

```python
arr = VarUIntArray([1, 2, 3], word_size=10)
serialize_varuintarray(arr)
# {'word_size': 10, 'values': [1, 2, 3]}
```

## Use Cases

### Custom Binary Protocols

Working with network protocols or file formats that use non-standard bit widths:

```python
# 12-bit color values (common in some image formats)
colors = VarUIntArray([4095, 2048, 0], word_size=12)
```

### Bit Manipulation

Performing bitwise operations on packed data:

```python
data = VarUIntArray([0b1010, 0b0101], word_size=4)
mask = VarUIntArray([0b1100, 0b0011], word_size=4)
result = data & mask  # Bitwise AND
```

## Implementation Details

### Memory Layout

VarUIntArray uses big-endian byte order (`'>'` dtype prefix) for consistency. Data is stored in the smallest standard NumPy unsigned integer type that can hold the specified word_size.

### Limitations

- Maximum word size: 64 bits
- Only unsigned integers are supported
- The `axis` parameter is not supported for `np.unpackbits` on VarUIntArray

## Examples

### Complete Workflow

```python
import numpy as np
from varuintarray import VarUIntArray

# Create some 5-bit values
data = VarUIntArray([31, 16, 0, 15], word_size=5)

# Unpack to bits
bits = data.unpackbits()
print(bits)
# [[1 1 1 1 1]
#  [1 0 0 0 0]
#  [0 0 0 0 0]
#  [0 1 1 1 1]]

# Flip specific bits
bits[:, 0] = 1 - bits[:, 0]  # Flip first bit

# Pack back
result = VarUIntArray.packbits(bits)
print(result)
# VarUIntArray([ 0, 16, 16, 31], dtype='>u1', word_size=5)

# Bitwise operations
inverted = result.invert()
print(inverted)
# VarUIntArray([31, 15, 15,  0], dtype='>u1', word_size=5)
```

### Serialization

```python
from varuintarray import VarUIntArray, serialize_varuintarray, validate_varuintarray
import json

# Create and serialize
arr = VarUIntArray([100, 200, 300], word_size=12)
data = serialize_varuintarray(arr)
json_str = json.dumps(data)

# Deserialize
loaded_data = json.loads(json_str)
arr_restored = validate_varuintarray(loaded_data)
```

## License

BSD 3-Clause License

Copyright (c) 2026, Jonathan Olsten

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.