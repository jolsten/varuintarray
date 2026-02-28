# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.0.2] - 2026-02-27

### Fixed

- Comparison ufuncs (`==`, `!=`, `<`, `>`, `<=`, `>=`) now return plain `np.ndarray` with `dtype=bool` instead of `VarUIntArray`. This fixes boolean mask indexing and `numpy.testing.assert_array_equal` failures where `~mask` produced uint8 values (e.g. 255) instead of boolean values.

### Added

- Constructor now rejects non-numeric input types (boolean, complex, string) with a `TypeError`, preventing silent coercion to unsigned integers.

## [1.0.1] - 2026-02-24

### Fixed

- `__array_wrap__` now accepts `context` and `return_scalar` as positional-only arguments, fixing a deprecation warning in NumPy 2.x.
