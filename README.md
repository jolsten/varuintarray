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

### Create a VarUIntArray with 10-bit words

```python
>>> arr = VarUIntArray([1, 2, 1023], word_size=10)
>>> arr
VarUIntArray([   1,    2, 1023], dtype='>u2', word_size=10)
```

### Bitwise operations respect word_size
```python
>>> inverted = arr.invert()
>>> inverted
VarUIntArray([1022, 1021,    0], dtype='>u2', word_size=10)
```

### Unpack to individual bits
```python
>>> bits = arr.unpackbits()
>>> bits.shape
(3, 10)
```

### Pack bits back into words
```python
>>> packed = VarUIntArray.packbits(bits)
>>> packed
VarUIntArray([  1,   2, 1023], dtype='>u2', word_size=10)
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
>>> arr = VarUIntArray([5], word_size=3)  # Binary: 101

# Standard NumPy invert would give 11111010 (250)
# VarUIntArray.invert() gives 010 (2) - correct for 3-bit word
>>> inverted = arr.invert()
>>> int(inverted[0])
2
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
- `to_dict()`: Serialize to a dictionary
- `from_dict(data)`: Static method to deserialize from a dictionary
- `to_json()`: Serialize to a JSON string
- `from_json(string)`: Class method to deserialize from a JSON string

#### Attributes

- `word_size`: Number of significant bits per word

### Functions

#### `unpackbits(array)`

Unpack a VarUIntArray into individual bits, excluding padding.

```python
>>> arr = VarUIntArray([5, 3], word_size=3)
>>> unpackbits(arr)
array([[1, 0, 1],
       [0, 1, 1]], dtype=uint8)
```

**Parameters:**
- `array`: VarUIntArray to unpack

**Returns:** ndarray with shape `(*original_shape, word_size)`

#### `packbits(array)`

Pack a bit array into a VarUIntArray.

```python
>>> bits = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
>>> packbits(bits)
VarUIntArray([5, 3], dtype=uint8, word_size=3)
```

**Parameters:**
- `array`: ndarray of uint8 containing 0s and 1s, where the last dimension contains bits for each word

**Returns:** VarUIntArray with one fewer dimension

#### `VarUIntArray.to_dict()`

Serialize VarUIntArray to JSON-compatible dictionary.

```python
>>> arr = VarUIntArray([1, 2, 3], word_size=10)
>>> arr.to_dict()
{'word_size': 10, 'values': [1, 2, 3]}
```

#### `VarUIntArray.from_dict(data)`

Convert various formats to VarUIntArray.

```python
# From dictionary
>>> VarUIntArray.from_dict({'values': [1, 2, 3], 'word_size': 10})
VarUIntArray([1, 2, 3], dtype='>u2', word_size=10)
```

#### `VarUIntArray.to_json()`

Serialize VarUIntArray to a JSON string.

```python
>>> arr = VarUIntArray([1, 2, 3], word_size=10)
>>> arr.to_json()
'{"word_size": 10, "values": [1, 2, 3]}'
```

#### `VarUIntArray.from_json(string)`

Deserialize a VarUIntArray from a JSON string.

```python
>>> json_str = '{"word_size": 10, "values": [1, 2, 3]}'
>>> VarUIntArray.from_json(json_str)
VarUIntArray([1, 2, 3], dtype='>u2', word_size=10)
```

## Use Cases

### Custom Binary Protocols

Working with network protocols or file formats that use non-standard bit widths:

```python
# 12-bit color values (common in some image formats)
>>> colors = VarUIntArray([4095, 2048, 0], word_size=12)
```

### Bit Manipulation

Performing bitwise operations on packed data:

```python
>>> data = VarUIntArray([0b1010, 0b0101], word_size=4)
>>> mask = VarUIntArray([0b1100, 0b0011], word_size=4)
>>> result = data & mask  # Bitwise AND
```

## Implementation Details

### Memory Layout

- VarUIntArray uses big-endian byte order (`'>'` dtype prefix) for consistency.
- Data is stored in the smallest standard NumPy unsigned integer type that can hold the specified word_size.

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
# VarUIntArray([15, 0, 16, 31], dtype='>u1', word_size=5)

# Bitwise operations
inverted = result.invert()
print(inverted)
# VarUIntArray([16, 31, 15,  0], dtype='>u1', word_size=5)
```

### Serialization

```python
>>> from varuintarray import VarUIntArray
>>> import json
# Serialize dict
>>> arr = VarUIntArray([100, 200, 300], word_size=12)
>>> serialized = arr.to_dict()
>>> serialized
{'word_size': 12, 'values': [100, 200, 300]}
# Deserialize dict
>>> VarUIntArray.from_dict(serialized)
VarUIntArray([100, 200, 300], dtype='>u2', word_size=12)
# Serialize JSON
>>> serialized = arr.to_json()
>>> serialized
'{"word_size": 12, "values": [100, 200, 300]}'
# Deserialize JSON
>>> VarUIntArray.from_json(serialized)
VarUIntArray([100, 200, 300], dtype='>u2', word_size=12)
```

## License

`varuintarray` is licensed under the MIT License - see the LICENSE file for details
