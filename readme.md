# l_srtde

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)

**A high-performance Rust implementation of the L-SRTDE (Large Scale Random Topology Differential Evolution) algorithm.**

This crate provides a robust and efficient solver for **Large Scale Global Optimization (LSGO)** problems. It is designed with parallelism in mind, utilizing `rayon` to accelerate population evaluation, making it suitable for high-dimensional numerical optimization tasks.

## 📄 Reference & Attribution

This project is a **Rust implementation** of the algorithm originally proposed by V. Stanovov and E. Semenkin. **I am not the author of the original algorithm.**

If you use this algorithm or this code in your research, please cite the original paper:

> **V. Stanovov and E. Semenkin**, "Success Rate-based Adaptive Differential Evolution L-SRTDE for CEC 2024 Competition," *2024 IEEE Congress on Evolutionary Computation (CEC)*, Yokohama, Japan, 2024, pp. 1-8, doi: [10.1109/CEC60901.2024.10611907](https://doi.org/10.1109/CEC60901.2024.10611907).

*Keywords: Sensitivity, Accuracy, Evolutionary computation, Benchmark testing, Optimization, Differential evolution, Numerical optimization, Parameter adaptation.*

---

## ✨ Features

* **Parallel Execution**: Leverages `rayon` for automatic parallel evaluation of the population, significantly speeding up large-scale problems.
* **Success-Rate Based Adaptation**: Implements the core novelty of L-SRTDE, adapting the Scaling Factor ($F$) based on the ratio of improved solutions.
* **LPSR (Linear Population Size Reduction)**: Dynamically reduces population size to balance exploration and exploitation.
* **Random Topology**: Uses a random topology strategy to handle high-dimensional search spaces efficiently.
* **Pure Rust**: Memory-safe, thread-safe, and zero-cost abstractions.

## 📦 Installation

Since this crate is hosted on GitHub, you can add it to your `Cargo.toml` as a Git dependency:

```toml
[dependencies]
l_srtde = { git = "https://github.com/Tydwdh/l_srtde-rs", branch = "main" }
```



🚀 Quick Start
Here is a simple example solving the Sphere function ($f(x) = \sum x_i^2$).
```Rust
use l_srtde::{Lsrtde, Problem, Solution};

// 1. Define the problem structure
struct SphereProblem {
    dim: usize,
}

// 2. Implement the Problem trait
impl Problem for SphereProblem {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn get_bounds(&self, _index: usize) -> (f32, f32) {
        (-100.0, 100.0)
    }

    fn evaluate(&self, genome: &[f32]) -> f32 {
        // Goal: Minimize the sum of squares
        genome.iter().map(|x| x * x).sum()
    }
}

fn main() {
    // Initialize the problem with 100 dimensions
    let problem = SphereProblem { dim: 100 };

    // Configure the solver
    let solver = Lsrtde::new(&problem)
        .with_max_evaluations(50_000) // Set max function evaluations
        .with_seed(42);               // Set a fixed seed for reproducibility

    println!("Running L-SRTDE on {}D Sphere problem...", problem.dim);

    // Run optimization
    let solution = solver.run();

    println!("Optimization Finished!");
    println!("Best Fitness: {:.6e}", solution.fitness);
}
```
⚙️ Advanced Configuration
You can customize the solver using the builder pattern:

```Rust
let solver = Lsrtde::new(&problem)
    .with_max_evaluations(1_000_000)
    .with_pop_size_multiplier(18)    // Default is 18
    .with_memory_size(5)             // Default is 5
    .with_seed(12345);
```
You can also use a callback to monitor progress or implement custom stopping criteria:

```Rust
solver.run_with_callback(|solution, evaluations| {
    if evaluations % 1000 == 0 {
        println!("Eval: {}, Current Best: {}", evaluations, solution.fitness);
    }
    true // Return false to stop early
});
```
⚖️ License
This project is licensed under either of

Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)

MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)
