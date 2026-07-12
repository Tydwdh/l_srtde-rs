# l_srtde

[English](README.md) | [简体中文](README.zh-CN.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)

A Rust implementation of the L-SRTDE (Large Scale Random Topology Differential
Evolution) algorithm for large-scale numerical optimization.

This crate focuses on large-scale global optimization problems and uses `rayon`
to parallelize population evaluation.

## Reference

This crate implements the algorithm proposed by V. Stanovov and E. Semenkin.
If you use the algorithm or this code in research, cite the original paper:

> V. Stanovov and E. Semenkin, "Success Rate-based Adaptive Differential
> Evolution L-SRTDE for CEC 2024 Competition," 2024 IEEE Congress on
> Evolutionary Computation (CEC), Yokohama, Japan, 2024, pp. 1-8,
> doi: [10.1109/CEC60901.2024.10611907](https://doi.org/10.1109/CEC60901.2024.10611907).

## Features

- Parallel population evaluation with `rayon`
- Success-rate based adaptation of the scaling factor `F`
- Linear population size reduction (LPSR)
- Random-topology strategy for large search spaces
- Pure Rust implementation
- C ABI dynamic library for non-Rust callers

## Installation

```toml
[dependencies]
l_srtde = "0.2.0"
```

## Quick Start

```rust
use l_srtde::{Lsrtde, Problem};

struct SphereProblem {
    dim: usize,
}

impl Problem for SphereProblem {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn get_bounds(&self, _index: usize) -> (f64, f64) {
        (-100.0, 100.0)
    }

    fn evaluate(&self, genome: &[f64]) -> f64 {
        genome.iter().map(|x| x * x).sum()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = SphereProblem { dim: 100 };

    let solver = Lsrtde::new(&problem)
        .with_max_evaluations(500_000)
        .with_seed(42);

    let solution = solver.run()?;

    println!("Best Fitness: {:.6e}", solution.fitness);
    println!("Best Genome: {:.2?}", solution.genome);
    Ok(())
}
```

## Advanced Configuration

```rust
let solver = Lsrtde::new(&problem)
    .with_max_evaluations(1_000_000)
    .with_pop_size_multiplier(18)
    .with_memory_size(5)
    .with_seed(12345);
```

You can also use a callback to monitor progress or stop the search early:

```rust
let mut generation = 0;
let solution = solver.run_with_callback(move |solution, evaluations| {
    generation += 1;

    if generation % 10 == 0 {
        println!("Eval: {}, Current Best: {}", evaluations, solution.fitness);
    }

    true
})?;
```

## C ABI / Dynamic Library

The crate can also be built as a dynamic library for C, C++, Python `ctypes`,
and other languages that can load a C ABI library.

Build the release library:

```bash
cargo build --release
```

The dynamic library is written to `target/release`:

- Windows: `l_srtde.dll`
- Linux: `libl_srtde.so`
- macOS: `libl_srtde.dylib`

Use `include/l_srtde.h` from C:

```c
#include "l_srtde.h"

static int32_t sphere_batch(
    const double *points,
    size_t point_count,
    size_t dim,
    double *fitness_out,
    void *user_data
) {
    (void)user_data;
    for (size_t i = 0; i < point_count; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            double x = points[i * dim + j];
            sum += x * x;
        }
        fitness_out[i] = sum;
    }
    return LSRTDE_OK;
}
```

The callback receives a row-major batch of `point_count * dim` doubles and must
write `point_count` finite fitness values. Return `0` on success and a non-zero
value to stop the optimization with `LSRTDE_CALLBACK_ERROR`. The input and
output pointers are valid only for the duration of the callback and must not be
retained after it returns.

Example link commands:

```bash
# Linux/macOS
cc main.c -Iinclude -Ltarget/release -ll_srtde -o main

# Windows MSVC
cl main.c /I include target\release\l_srtde.dll.lib
```

Minimal Python `ctypes` loading:

```python
import ctypes

lib = ctypes.CDLL("./l_srtde.dll")  # use ./libl_srtde.so or ./libl_srtde.dylib on Unix
lib.lsrtde_minimize.restype = ctypes.c_int32

CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
)

@CALLBACK
def sphere(points, point_count, dim, fitness_out, user_data):
    for i in range(point_count):
        total = 0.0
        for j in range(dim):
            x = points[i * dim + j]
            total += x * x
        fitness_out[i] = total
    return 0
```

When a C ABI config field is set to zero, `max_evaluations`, `memory_size`, and
`pop_size_multiplier` use the Rust defaults. Set `use_seed` to `0` for a random
seed or non-zero to use `seed`.

## Validation And Budget Semantics

`run()` and `run_with_callback()` now return `Result<_, LsrtdeError>`. The
solver rejects invalid configurations before any parallel evaluation starts.

The current validation rules are:

- `dimension() > 0`
- `memory_size > 0`
- `dimension * pop_size_multiplier` must not overflow
- initial population size must be at least `3`
- every `(lower, upper)` bound pair must be finite, satisfy `lower < upper`, and
  have a finite width (`upper - lower`)
- objective evaluations must return finite fitness values; Rust returns
  `LsrtdeError::NonFiniteFitness` and the C ABI returns
  `LSRTDE_NONFINITE_FITNESS`

`with_max_evaluations()` is a soft budget, not a hard cap:

- the initial population is always evaluated in full
- each generation evaluates a full batch of trial vectors in parallel
- total objective evaluations can exceed the configured budget
- the overrun is bounded by at most one current-generation population size

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE).
