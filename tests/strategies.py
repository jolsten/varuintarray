from typing import Optional

from hypothesis import strategies as st

from varuintarray.array import (
    VarUIntArray,
)

MAX_BITS_PER_WORD = 64


@st.composite
def varuintarrays(draw, word_size: Optional[int] = None) -> VarUIntArray:
    if word_size is None:
        size: int = draw(st.integers(min_value=1, max_value=MAX_BITS_PER_WORD))
    else:
        size = word_size
    values = draw(
        st.lists(
            st.integers(min_value=0, max_value=2**size - 1),
            min_size=1,
            max_size=1024,
        )
    )
    return VarUIntArray(values, word_size=size)
