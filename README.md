# Usage

`VarUIntArray` is a subclass of `np.ndarray` with a `word_size` attribute. 
Some `numpy` functions are handled to ensure the expected behavior given a specific word_size
(that doesn't happen to be uint8, uint16, uint32, or uint64).

## np.invert

```python
In [1]: import numpy as np

# Suppose we have 4-bit data array expressed with pad bits to 8 bits per word
In [2]: array = np.array([0, 1, 2, 3], dtype="u1")

# Inverting the array inverts all of the bits, 
# including the upper 4 bits that aren't valid
In [4]: np.invert(array)
Out[4]: array([255, 254, 253, 252], dtype=uint8)
```

```python
In [1]: import numpy as np; from varuintarray import VarUIntArray

# Instead, use VarUIntArray to do the same thing
In [2]: array = VarUIntArray([0, 1, 2, 3], word_size=4)

# Inverting the array only inverts the valid bits!
In [3]: np.invert(array)
Out[3]: VarUIntArray([15, 14, 13, 12], dtype=uint8, word_size=4)
```

## np.unpackbits

```python
In [1]: import numpy as np

# Unpacking a standard ndarray, of course, yields 8 * itemsize bits per element
In [2]: array = np.array([[0], [15]], dtype="u1")

In [3]: np.unpackbits(array).reshape(2, -1)
Out[3]: 
array([[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 1, 1]], dtype=uint8)
```

Doing the same with a `VarUIntArray`...

```python
In [1]: import numpy as np; from varuintarray import VarUIntArray

In [2]: array = VarUIntArray([[0], [15]], word_size=4)

In [3]: array.unpack()
Out[3]: 
array([[0, 0, 0, 0],
       [1, 1, 1, 1]], dtype=uint8)
```

A `VarUIntArray` can be constructed from bits as well...

```python
In [1]: from varuintarray import VarUIntArray

In [2]: VarUIntArray.pack([[0, 0, 0, 0], [1, 1, 1, 1]], word_size=4)
Out[2]: 
VarUIntArray([[ 0],
              [15]], dtype=uint8, word_size=4)
```