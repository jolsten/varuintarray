# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.0.6] - 2026-03-09

### Changed

- **Breaking:** Operations between VarUIntArrays with different `word_size` values now raise `ValueError`. Previously, the result silently used the first operand's word_size.
- **Breaking:** `np.subtract` now raises `OverflowError` on unsigned underflow (e.g. subtracting a larger value from a smaller one). Previously it silently wrapped.
- **Breaking:** Comparing VarUIntArrays with different `word_size` via `==` now raises `ValueError`.
- `validate()` and `serialize()` are now private (`_validate`, `_serialize`). Use `VarUIntArray.to_dict()` / `VarUIntArray.from_dict()` instead.
- Removed unused `lark` dependency.
- Fixed project description typo ("unsinged" → "unsigned").

### Added

- `py.typed` marker for PEP 561 type checking support.
- CI test against `numpy==2.0.*` lower bound.

### Fixed

- `TestInvert` assertions now actually validate results (previously always passed due to asserting on a non-empty list).

## [1.0.5] - 2026-03-05

### Changed

- `__array_wrap__` now uses positional-only arguments with defaults (`context=None, return_scalar=False, /`) to match the NumPy 2.x `ndarray.__array_wrap__` interface.

## [1.0.2] - 2026-02-27

### Fixed

- Comparison ufuncs (`==`, `!=`, `<`, `>`, `<=`, `>=`) now return plain `np.ndarray` with `dtype=bool` instead of `VarUIntArray`. This fixes boolean mask indexing and `numpy.testing.assert_array_equal` failures where `~mask` produced uint8 values (e.g. 255) instead of boolean values.

### Added

- Constructor now rejects non-numeric input types (boolean, complex, string) with a `TypeError`, preventing silent coercion to unsigned integers.

## [1.0.1] - 2026-02-24

### Fixed

- `__array_wrap__` now accepts `context` and `return_scalar` as positional-only arguments, fixing a deprecation warning in NumPy 2.x.
