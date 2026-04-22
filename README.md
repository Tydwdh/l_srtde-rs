# l_srtde

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

## Installation

```toml
[dependencies]
l_srtde = "0.1.1"
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

## Validation And Budget Semantics

`run()` and `run_with_callback()` now return `Result<_, LsrtdeError>`. The
solver rejects invalid configurations before any parallel evaluation starts.

The current validation rules are:

- `dimension() > 0`
- `memory_size > 0`
- `dimension * pop_size_multiplier` must not overflow
- initial population size must be at least `3`
- every `(lower, upper)` bound pair must be finite and satisfy `lower < upper`

`with_max_evaluations()` is a soft budget, not a hard cap:

- the initial population is always evaluated in full
- each generation evaluates a full batch of trial vectors in parallel
- total objective evaluations can exceed the configured budget
- the overrun is bounded by at most one current-generation population size

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE).
