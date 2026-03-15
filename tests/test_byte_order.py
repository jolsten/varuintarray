import sys

import numpy as np
import pytest

from varuintarray.array import (
    VarUIntArray,
    _normalize_byteorder,
    packbits,
    unpackbits,
)


class TestNormalizeByteorder:
    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("big", ">"),
            ("little", "<"),
            ("native", "="),
            (">", ">"),
            ("<", "<"),
            ("=", "="),
        ],
    )
    def test_valid_values(self, input_val, expected):
        assert _normalize_byteorder(input_val) == expected

    def test_invalid_value(self):
        with pytest.raises(ValueError, match="Invalid byte order"):
            _normalize_byteorder("middle")

    @pytest.mark.parametrize("bad_input", ["", "|", "BIG", "Little", "network", "host"])
    def test_various_invalid_values(self, bad_input):
        with pytest.raises(ValueError, match="Invalid byte order"):
            _normalize_byteorder(bad_input)


class TestConstructorByteOrder:
    def test_default_is_native(self):
        arr = VarUIntArray([1, 2, 3], word_size=10)
        assert arr.byteorder == "="
        assert arr.dtype.byteorder in ("=", sys.byteorder[0])

    def test_big_endian(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder="big")
        assert arr.byteorder == ">"
        assert arr.dtype.byteorder == ">"

    def test_little_endian(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder="little")
        assert arr.byteorder == "<"
        # numpy normalizes native-matching byte order to '='
        assert arr.dtype.byteorder in ("<", "=")

    def test_numpy_prefix_syntax(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder=">")
        assert arr.byteorder == ">"

    def test_values_preserved_across_byte_orders(self):
        """Same input values should produce same logical values regardless of byte order."""
        values = [1, 255, 1023]
        big = VarUIntArray(values, word_size=10, byteorder="big")
        little = VarUIntArray(values, word_size=10, byteorder="little")
        native = VarUIntArray(values, word_size=10, byteorder="native")
        np.testing.assert_array_equal(big.tolist(), values)
        np.testing.assert_array_equal(little.tolist(), values)
        np.testing.assert_array_equal(native.tolist(), values)

    def test_uint8_byte_order_irrelevant(self):
        """For word_size <= 8, byte order doesn't affect storage."""
        big = VarUIntArray([1, 2, 3], word_size=5, byteorder="big")
        little = VarUIntArray([1, 2, 3], word_size=5, byteorder="little")
        np.testing.assert_array_equal(big, little)

    def test_uint8_byteorder_attribute_preserved(self):
        """Even for single-byte dtypes, the byteorder attribute should be stored."""
        arr = VarUIntArray([1, 2, 3], word_size=8, byteorder="big")
        assert arr.byteorder == ">"

    @pytest.mark.parametrize("word_size", [10, 16, 24, 32, 48, 64])
    def test_values_preserved_multibyte_word_sizes(self, word_size):
        """Values preserved across byte orders for all multi-byte word sizes."""
        values = [0, 1, 2**word_size - 1, 2**(word_size - 1)]
        for order in ("big", "little", "native"):
            arr = VarUIntArray(values, word_size=word_size, byteorder=order)
            assert arr.tolist() == values

    def test_invalid_byteorder_in_constructor(self):
        with pytest.raises(ValueError, match="Invalid byte order"):
            VarUIntArray([1, 2, 3], word_size=10, byteorder="bad")

    def test_constructor_with_pre_typed_big_endian_input(self):
        """Passing a big-endian numpy array with little-endian byteorder."""
        data = np.array([1, 2, 3], dtype=">u2")
        arr = VarUIntArray(data, word_size=10, byteorder="little")
        assert arr.byteorder == "<"
        assert arr.tolist() == [1, 2, 3]

    def test_constructor_with_pre_typed_little_endian_input(self):
        """Passing a little-endian numpy array with big-endian byteorder."""
        data = np.array([1, 2, 3], dtype="<u2")
        arr = VarUIntArray(data, word_size=10, byteorder="big")
        assert arr.byteorder == ">"
        assert arr.tolist() == [1, 2, 3]


class TestRawMemoryLayout:
    """Verify that byte order actually affects the raw byte layout in memory."""

    def test_big_endian_bytes(self):
        """Big-endian: most significant byte first."""
        arr = VarUIntArray([0x0102], word_size=16, byteorder="big")
        raw = arr.view(np.uint8)
        assert raw[0] == 0x01
        assert raw[1] == 0x02

    def test_little_endian_bytes(self):
        """Little-endian: least significant byte first."""
        arr = VarUIntArray([0x0102], word_size=16, byteorder="little")
        raw = arr.view(np.uint8)
        assert raw[0] == 0x02
        assert raw[1] == 0x01

    def test_big_vs_little_raw_bytes_differ(self):
        """For multi-byte values, raw bytes must differ between byte orders."""
        big = VarUIntArray([0x0102], word_size=16, byteorder="big")
        little = VarUIntArray([0x0102], word_size=16, byteorder="little")
        big_bytes = big.view(np.uint8).tolist()
        little_bytes = little.view(np.uint8).tolist()
        assert big_bytes != little_bytes
        assert big_bytes == list(reversed(little_bytes))

    def test_32bit_memory_layout(self):
        """Verify 32-bit value byte layout."""
        arr_big = VarUIntArray([0x01020304], word_size=32, byteorder="big")
        raw_big = arr_big.view(np.uint8).tolist()
        assert raw_big == [0x01, 0x02, 0x03, 0x04]

        arr_little = VarUIntArray([0x01020304], word_size=32, byteorder="little")
        raw_little = arr_little.view(np.uint8).tolist()
        assert raw_little == [0x04, 0x03, 0x02, 0x01]


class TestByteOrderPropagation:
    def test_array_finalize(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder="big")
        sliced = arr[:2]
        assert sliced.byteorder == ">"

    def test_ufunc_preserves_byteorder(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder="big")
        result = np.invert(arr)
        assert result.byteorder == ">"

    def test_concatenate_preserves_byteorder(self):
        a = VarUIntArray([1, 2], word_size=10, byteorder="big")
        b = VarUIntArray([3, 4], word_size=10, byteorder="big")
        result = np.concatenate([a, b])
        assert isinstance(result, VarUIntArray)
        assert result.byteorder == ">"

    def test_append_preserves_byteorder(self):
        a = VarUIntArray([1, 2], word_size=10, byteorder="little")
        b = VarUIntArray([3, 4], word_size=10, byteorder="little")
        result = np.append(a, b)
        assert isinstance(result, VarUIntArray)
        assert result.byteorder == "<"

    def test_copy_preserves_byteorder(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder="big")
        result = np.copy(arr)
        assert result.byteorder == ">"

    def test_extend_preserves_byteorder(self):
        arr = VarUIntArray([1, 2], word_size=10, byteorder="big")
        result = arr.extend([3, 4])
        assert result.byteorder == ">"

    def test_append_method_preserves_byteorder(self):
        arr = VarUIntArray([1, 2], word_size=10, byteorder="little")
        result = arr.append(3)
        assert result.byteorder == "<"

    def test_fancy_indexing_preserves_byteorder(self):
        arr = VarUIntArray([10, 20, 30, 40], word_size=10, byteorder="big")
        result = arr[[0, 2]]
        assert isinstance(result, VarUIntArray)
        assert result.byteorder == ">"
        assert result.tolist() == [10, 30]

    def test_boolean_indexing_preserves_byteorder(self):
        arr = VarUIntArray([10, 20, 30], word_size=10, byteorder="big")
        mask = np.array([True, False, True])
        result = arr[mask]
        assert isinstance(result, VarUIntArray)
        assert result.byteorder == ">"
        assert result.tolist() == [10, 30]

    def test_reshape_preserves_byteorder(self):
        arr = VarUIntArray([1, 2, 3, 4], word_size=10, byteorder="big")
        result = arr.reshape(2, 2)
        assert isinstance(result, VarUIntArray)
        assert result.byteorder == ">"

    def test_copy_preserves_little_endian(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder="little")
        result = np.copy(arr)
        assert result.byteorder == "<"

    def test_ufunc_preserves_little_endian(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder="little")
        result = np.invert(arr)
        assert result.byteorder == "<"


class TestUfuncByteOrderCorrectness:
    """Verify ufunc results are numerically correct across byte orders."""

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_invert_correctness(self, byteorder):
        arr = VarUIntArray([0, 1, 1023], word_size=10, byteorder=byteorder)
        result = np.invert(arr)
        assert result.tolist() == [1023, 1022, 0]

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_add_correctness(self, byteorder):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder=byteorder)
        result = np.add(arr, 10)
        assert result.tolist() == [11, 12, 13]

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_subtract_correctness(self, byteorder):
        arr = VarUIntArray([10, 20, 30], word_size=10, byteorder=byteorder)
        result = np.subtract(arr, 5)
        assert result.tolist() == [5, 15, 25]

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_bitwise_xor_correctness(self, byteorder):
        arr = VarUIntArray([0b1010101010], word_size=10, byteorder=byteorder)
        result = np.bitwise_xor(arr, 0b1111111111)
        assert result.tolist() == [0b0101010101]

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_left_shift_correctness(self, byteorder):
        arr = VarUIntArray([1], word_size=10, byteorder=byteorder)
        result = np.left_shift(arr, 9)
        assert result.tolist() == [512]

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_left_shift_overflow_masked(self, byteorder):
        """Left-shifting beyond word_size should be masked to zero."""
        arr = VarUIntArray([1], word_size=10, byteorder=byteorder)
        result = np.left_shift(arr, 10)
        assert result.tolist() == [0]

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_multiply_overflow_masked(self, byteorder):
        """Multiplication overflow should be masked to word_size bits."""
        arr = VarUIntArray([512], word_size=10, byteorder=byteorder)
        result = np.multiply(arr, 4)
        # 512 * 4 = 2048 = 0b100000000000, masked to 10 bits = 0
        assert result.tolist() == [0]


class TestMixedByteOrderOperations:
    """Test operations between arrays with different byte orders."""

    def test_concatenate_mixed_byteorder_values_correct(self):
        """Concatenating arrays with different byte orders should preserve values."""
        a = VarUIntArray([1, 2], word_size=10, byteorder="big")
        b = VarUIntArray([3, 4], word_size=10, byteorder="little")
        result = np.concatenate([a, b])
        assert isinstance(result, VarUIntArray)
        assert result.tolist() == [1, 2, 3, 4]

    def test_append_mixed_byteorder_values_correct(self):
        """Appending arrays with different byte orders should preserve values."""
        a = VarUIntArray([1, 2], word_size=10, byteorder="big")
        b = VarUIntArray([3, 4], word_size=10, byteorder="little")
        result = np.append(a, b)
        assert isinstance(result, VarUIntArray)
        assert result.tolist() == [1, 2, 3, 4]

    def test_concatenate_mixed_byteorder_uses_first(self):
        """Result byteorder should come from the first (self) array."""
        a = VarUIntArray([1, 2], word_size=10, byteorder="big")
        b = VarUIntArray([3, 4], word_size=10, byteorder="little")
        result = np.concatenate([a, b])
        assert result.byteorder == ">"

    def test_binary_ufunc_mixed_byteorder_values_correct(self):
        """Binary ufuncs between different byte orders should produce correct values."""
        a = VarUIntArray([100, 200], word_size=10, byteorder="big")
        b = VarUIntArray([10, 20], word_size=10, byteorder="little")
        result = np.add(a, b)
        assert result.tolist() == [110, 220]

    def test_bitwise_and_mixed_byteorder(self):
        a = VarUIntArray([0b1010101010], word_size=10, byteorder="big")
        b = VarUIntArray([0b1111100000], word_size=10, byteorder="little")
        result = np.bitwise_and(a, b)
        assert result.tolist() == [0b1010100000]

    def test_comparison_mixed_byteorder(self):
        a = VarUIntArray([1, 2, 3], word_size=10, byteorder="big")
        b = VarUIntArray([3, 2, 1], word_size=10, byteorder="little")
        result = a == b
        assert type(result) is np.ndarray
        assert result.tolist() == [False, True, False]


class TestPackbitsbyteorder:
    def test_default_native(self):
        bits = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        result = packbits(bits)
        assert result.byteorder == "="

    def test_big_endian(self):
        bits = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)
        result = packbits(bits, byteorder="big")
        assert result.byteorder == ">"
        assert result[0] == 1

    def test_little_endian(self):
        bits = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)
        result = packbits(bits, byteorder="little")
        assert result.byteorder == "<"
        assert result[0] == 1

    def test_roundtrip_big_endian(self):
        bits = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8)
        packed = packbits(bits, byteorder="big")
        unpacked = unpackbits(packed)
        np.testing.assert_array_equal(bits, unpacked)

    def test_roundtrip_little_endian(self):
        bits = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8)
        packed = packbits(bits, byteorder="little")
        unpacked = unpackbits(packed)
        np.testing.assert_array_equal(bits, unpacked)

    def test_classmethod_forwards_byteorder(self):
        bits = np.array([[1, 0, 1]], dtype=np.uint8)
        result = VarUIntArray.packbits(bits, byteorder="big")
        assert result.byteorder == ">"

    def test_packbits_big_vs_little_same_values(self):
        """Packing the same bits with different byte orders produces same logical values."""
        bits = np.array(
            [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
            dtype=np.uint8,
        )
        big = packbits(bits, byteorder="big")
        little = packbits(bits, byteorder="little")
        assert big.tolist() == little.tolist()

    def test_roundtrip_2d_big_endian(self):
        """Pack/unpack roundtrip with 2D bit arrays and big-endian byte order."""
        bits = np.array(
            [
                [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            ],
            dtype=np.uint8,
        )
        packed = packbits(bits, byteorder="big")
        assert packed.shape == (2, 2)
        unpacked = unpackbits(packed)
        np.testing.assert_array_equal(bits, unpacked)

    def test_roundtrip_2d_little_endian(self):
        """Pack/unpack roundtrip with 2D bit arrays and little-endian byte order."""
        bits = np.array(
            [
                [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            ],
            dtype=np.uint8,
        )
        packed = packbits(bits, byteorder="little")
        assert packed.shape == (2, 2)
        unpacked = unpackbits(packed)
        np.testing.assert_array_equal(bits, unpacked)

    @pytest.mark.parametrize("word_size", [10, 16, 24, 32])
    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_packbits_roundtrip_various_sizes(self, word_size, byteorder):
        """Roundtrip pack/unpack works for various word sizes and byte orders."""
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=(5, word_size), dtype=np.uint8)
        packed = packbits(bits, byteorder=byteorder)
        unpacked = unpackbits(packed)
        np.testing.assert_array_equal(bits, unpacked)


class TestUnpackbitsByteOrder:
    """Verify unpackbits produces correct bit values regardless of byte order."""

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_unpackbits_value_1_10bit(self, byteorder):
        arr = VarUIntArray([1], word_size=10, byteorder=byteorder)
        bits = unpackbits(arr)
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        assert bits.tolist() == [expected]

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_unpackbits_max_10bit(self, byteorder):
        arr = VarUIntArray([1023], word_size=10, byteorder=byteorder)
        bits = unpackbits(arr)
        expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert bits.tolist() == [expected]

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_unpackbits_alternating_10bit(self, byteorder):
        arr = VarUIntArray([0b1010101010], word_size=10, byteorder=byteorder)
        bits = unpackbits(arr)
        expected = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        assert bits.tolist() == [expected]

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_unpackbits_24bit(self, byteorder):
        """24-bit values stored in 32-bit container."""
        arr = VarUIntArray([0xABCDEF], word_size=24, byteorder=byteorder)
        bits = unpackbits(arr)
        # 0xABCDEF = 1010 1011 1100 1101 1110 1111
        expected = [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        assert bits.tolist() == [expected]


class TestSerializationByteOrder:
    def test_to_dict_includes_byteorder(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder="big")
        d = arr.to_dict()
        assert d["byteorder"] == "big"

    def test_to_dict_native(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder="native")
        d = arr.to_dict()
        assert d["byteorder"] == "native"

    def test_to_dict_little(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder="little")
        d = arr.to_dict()
        assert d["byteorder"] == "little"

    def test_from_dict_with_byteorder(self):
        d = {"word_size": 10, "byteorder": "big", "values": [1, 2, 3]}
        arr = VarUIntArray.from_dict(d)
        assert arr.byteorder == ">"

    def test_from_dict_without_byteorder_defaults_native(self):
        """Backwards compatibility: old serialized data without byteorder."""
        d = {"word_size": 10, "values": [1, 2, 3]}
        arr = VarUIntArray.from_dict(d)
        assert arr.byteorder == "="

    def test_json_roundtrip_preserves_byteorder(self):
        arr = VarUIntArray([1, 2, 3], word_size=10, byteorder="little")
        restored = VarUIntArray.from_json(arr.to_json())
        assert restored.byteorder == "<"
        np.testing.assert_array_equal(arr, restored)

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_json_roundtrip_all_orders(self, byteorder):
        """JSON roundtrip preserves values and byteorder for all byte orders."""
        values = [0, 1, 255, 1023]
        arr = VarUIntArray(values, word_size=10, byteorder=byteorder)
        restored = VarUIntArray.from_json(arr.to_json())
        prefix = _normalize_byteorder(byteorder)
        assert restored.byteorder == prefix
        assert restored.tolist() == values

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_dict_roundtrip_all_orders(self, byteorder):
        """Dict roundtrip preserves values and byteorder for all byte orders."""
        values = [0, 1, 255, 1023]
        arr = VarUIntArray(values, word_size=10, byteorder=byteorder)
        restored = VarUIntArray.from_dict(arr.to_dict())
        prefix = _normalize_byteorder(byteorder)
        assert restored.byteorder == prefix
        assert restored.tolist() == values

    def test_from_dict_with_prefix_syntax(self):
        """from_dict should accept numpy prefix syntax for byteorder."""
        d = {"word_size": 10, "byteorder": ">", "values": [1, 2, 3]}
        arr = VarUIntArray.from_dict(d)
        assert arr.byteorder == ">"
        assert arr.tolist() == [1, 2, 3]


class TestByteOrderEdgeCases:
    """Edge cases specific to byte order handling."""

    def test_scalar_byteorder(self):
        """0-d scalar VarUIntArray should preserve byteorder."""
        arr = VarUIntArray(42, word_size=10, byteorder="big")
        assert arr.byteorder == ">"
        assert int(arr) == 42

    def test_empty_array_byteorder(self):
        """Empty VarUIntArray should preserve byteorder."""
        arr = VarUIntArray([], word_size=10, byteorder="big")
        assert arr.byteorder == ">"

    def test_2d_array_byteorder(self):
        arr = VarUIntArray([[1, 2], [3, 4]], word_size=10, byteorder="big")
        assert arr.byteorder == ">"
        assert arr.tolist() == [[1, 2], [3, 4]]

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_invert_roundtrip(self, byteorder):
        """Double invert should return to original values."""
        arr = VarUIntArray([0, 1, 511, 1023], word_size=10, byteorder=byteorder)
        result = np.invert(np.invert(arr))
        assert result.tolist() == arr.tolist()

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_serialize_roundtrip_empty(self, byteorder):
        """Serialization roundtrip works for empty arrays with all byte orders."""
        arr = VarUIntArray([], word_size=10, byteorder=byteorder)
        restored = VarUIntArray.from_dict(arr.to_dict())
        prefix = _normalize_byteorder(byteorder)
        assert restored.byteorder == prefix
        assert restored.tolist() == []

    @pytest.mark.parametrize("byteorder", ["big", "little", "native"])
    def test_serialize_roundtrip_scalar(self, byteorder):
        """Serialization roundtrip works for scalar arrays with all byte orders."""
        arr = VarUIntArray(42, word_size=10, byteorder=byteorder)
        restored = VarUIntArray.from_dict(arr.to_dict())
        prefix = _normalize_byteorder(byteorder)
        assert restored.byteorder == prefix
        assert int(restored) == 42

    def test_subtract_underflow_all_byteorders(self):
        """Underflow detection works regardless of byte order."""
        for order in ("big", "little", "native"):
            arr = VarUIntArray([0], word_size=10, byteorder=order)
            with pytest.raises(OverflowError, match="underflow"):
                np.subtract(arr, 1)

    def test_eq_across_byteorders(self):
        """Equality comparison between big and little endian arrays with same values."""
        a = VarUIntArray([1, 2, 3], word_size=10, byteorder="big")
        b = VarUIntArray([1, 2, 3], word_size=10, byteorder="little")
        result = a == b
        assert type(result) is np.ndarray
        assert result.tolist() == [True, True, True]
