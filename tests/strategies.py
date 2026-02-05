from typing import Optional

from hypothesis import strategies as st
from hypothesis.extra import numpy as nps

from varuintarray.array import (
    VarUIntArray,
    _word_size_to_dtype,
)

MAX_BITS_PER_WORD = 64


@st.composite
def varuintarrays(draw, word_size: Optional[int] = None) -> VarUIntArray:
    if word_size is None:
        size: int = draw(st.integers(min_value=1, max_value=MAX_BITS_PER_WORD))
    else:
        size = word_size

    dtype = _word_size_to_dtype(size)

    data = draw(
        nps.arrays(
            dtype=dtype,
            shape=nps.array_shapes(min_dims=0, max_dims=2, min_side=1, max_side=1024),
        )
    )

    return VarUIntArray(data, word_size=size)
