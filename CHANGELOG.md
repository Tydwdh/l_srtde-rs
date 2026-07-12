# Changelog

## 0.2.0 - 2026-07-12

- Unify the native Rust and C ABI solver paths.
- Process every trial vector that was evaluated in a soft-budget batch.
- Reject non-finite objective values and bounds with overflowing widths.
- Mark the raw-pointer C ABI entry point as `unsafe` for Rust callers.
- Reuse FFI evaluation buffers and remove the per-generation population clone.
- Add regression tests, release benchmarks, and updated API documentation.

## 0.1.3

- Add the Chinese documentation entry.
