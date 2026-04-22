//! # L-SRTDE
//!
//! A Rust implementation of Large Scale Random Topology Differential Evolution
//! for large-scale numerical optimization problems.
//!
//! The solver evaluates the population in parallel with `rayon` and adapts the
//! scaling factor `F` using the success rate of the current generation.
//!
//! ## Soft Evaluation Budget
//!
//! `with_max_evaluations()` configures a soft budget, not a strict hard limit.
//! The solver always evaluates the full initial population first, and each
//! generation evaluates a full batch of trial vectors in parallel before the
//! budget check is applied to accepted children. As a result, the total number
//! of calls to [`Problem::evaluate`] may exceed the configured value.
//!
//! ## Quick Start
//!
//! ```rust
//! use l_srtde::{Lsrtde, Problem};
//!
//! struct SphereProblem {
//!     dim: usize,
//! }
//!
//! impl Problem for SphereProblem {
//!     fn dimension(&self) -> usize {
//!         self.dim
//!     }
//!
//!     fn get_bounds(&self, _index: usize) -> (f64, f64) {
//!         (-100.0, 100.0)
//!     }
//!
//!     fn evaluate(&self, genome: &[f64]) -> f64 {
//!         genome.iter().map(|x| x * x).sum()
//!     }
//! }
//!
//! let problem = SphereProblem { dim: 100 };
//! let solver = Lsrtde::new(&problem)
//!     .with_max_evaluations(500_000)
//!     .with_seed(42);
//!
//! let solution = solver.run().unwrap();
//! assert!(solution.fitness < 1e-5);
//! ```

use core::fmt;
use std::error::Error;

use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// Defines an optimization problem that can be solved by [`Lsrtde`].
///
/// The solver uses `rayon` internally, so implementations must be `Sync`.
///
/// # Examples
///
/// ```rust
/// use l_srtde::Problem;
///
/// struct SphereProblem {
///     dim: usize,
/// }
///
/// impl Problem for SphereProblem {
///     fn dimension(&self) -> usize {
///         self.dim
///     }
///
///     fn get_bounds(&self, _index: usize) -> (f64, f64) {
///         (-100.0, 100.0)
///     }
///
///     fn evaluate(&self, genome: &[f64]) -> f64 {
///         genome.iter().map(|x| x * x).sum()
///     }
/// }
/// ```
pub trait Problem: Sync {
    /// Returns the dimensionality of the decision vector.
    fn dimension(&self) -> usize;

    /// Returns the lower and upper bound for one dimension.
    fn get_bounds(&self, index: usize) -> (f64, f64);

    /// Evaluates one candidate solution.
    ///
    /// Lower values are considered better.
    fn evaluate(&self, genome: &[f64]) -> f64;
}

/// The best solution found by the solver.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Best candidate vector.
    pub genome: Vec<f64>,
    /// Objective value for the candidate vector.
    pub fitness: f64,
}

/// Errors returned when the solver configuration is invalid.
#[derive(Debug, Clone, PartialEq)]
pub enum LsrtdeError {
    ZeroDimension,
    ZeroMemorySize,
    PopulationSizeOverflow,
    PopulationTooSmall {
        population_size: usize,
        minimum: usize,
    },
    InvalidBounds {
        index: usize,
        lower: f64,
        upper: f64,
    },
}

impl fmt::Display for LsrtdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "problem dimension must be greater than zero"),
            Self::ZeroMemorySize => write!(f, "memory_size must be greater than zero"),
            Self::PopulationSizeOverflow => write!(
                f,
                "initial population size overflowed usize when computing dimension * multiplier"
            ),
            Self::PopulationTooSmall {
                population_size,
                minimum,
            } => write!(
                f,
                "initial population size {population_size} is too small; expected at least {minimum}"
            ),
            Self::InvalidBounds {
                index,
                lower,
                upper,
            } => write!(
                f,
                "invalid bounds at dimension {index}: expected finite lower < upper, got ({lower}, {upper})"
            ),
        }
    }
}

impl Error for LsrtdeError {}

#[derive(Clone, Debug)]
struct Individual {
    genome: Vec<f64>,
    fitness: f64,
}

/// L-SRTDE solver.
pub struct Lsrtde<'a, P: Problem> {
    problem: &'a P,
    /// Soft evaluation budget. The actual number of evaluations can exceed this
    /// value because the initial population and each generation are evaluated in
    /// whole batches.
    max_evaluations: usize,
    /// Size of the adaptive CR memory.
    memory_size: usize,
    /// Initial population size multiplier. The initial population size is
    /// `dimension * pop_size_multiplier`.
    pop_size_multiplier: usize,
    /// Optional random seed for reproducibility.
    seed: Option<u64>,
}

#[derive(Debug)]
struct ValidatedConfig {
    n_vars: usize,
    pop_size_init: usize,
    bounds: Vec<(f64, f64)>,
}

impl<'a, P: Problem> Lsrtde<'a, P> {
    /// Creates a solver with default parameters.
    pub fn new(problem: &'a P) -> Self {
        Self {
            problem,
            max_evaluations: 100_000,
            memory_size: 5,
            pop_size_multiplier: 18,
            seed: None,
        }
    }

    /// Sets the soft evaluation budget.
    ///
    /// This is not a strict hard limit. The solver always evaluates the full
    /// initial population first, and each generation evaluates a full batch of
    /// trial vectors in parallel. The total number of calls to
    /// [`Problem::evaluate`] therefore:
    ///
    /// - is at least the initial population size
    /// - can exceed `n` by up to one current-generation population size
    pub fn with_max_evaluations(mut self, n: usize) -> Self {
        self.max_evaluations = n;
        self
    }

    /// Sets the size of the adaptive CR memory.
    pub fn with_memory_size(mut self, size: usize) -> Self {
        self.memory_size = size;
        self
    }

    /// Sets the initial population size multiplier.
    pub fn with_pop_size_multiplier(mut self, multiplier: usize) -> Self {
        self.pop_size_multiplier = multiplier;
        self
    }

    /// Sets the RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Runs the solver until the soft budget is exhausted or the callback stops
    /// the search.
    #[inline]
    pub fn run(&self) -> Result<Solution, LsrtdeError> {
        self.run_with_callback(|_, _| true)
    }

    /// Runs the solver and invokes a callback once per generation.
    ///
    /// The callback receives the current best solution and the accepted
    /// evaluation counter. Returning `false` stops the search early.
    ///
    /// The soft-budget semantics are the same as [`Lsrtde::with_max_evaluations`]:
    /// the total number of objective evaluations can exceed the configured
    /// value because trial vectors are generated and evaluated in full batches.
    pub fn run_with_callback<F>(&self, mut callback: F) -> Result<Solution, LsrtdeError>
    where
        F: FnMut(&Solution, usize) -> bool,
    {
        let config = self.validate_config()?;
        let n_vars = config.n_vars;
        let max_feval = self.max_evaluations;
        let bounds = config.bounds;

        let master_seed = self.seed.unwrap_or_else(|| rand::rng().random());
        let mut master_rng = StdRng::seed_from_u64(master_seed);

        let pop_size_init = config.pop_size_init;
        let mut n_inds_front = pop_size_init;
        let n_inds_front_max = pop_size_init;
        let n_inds_min = 4.min(pop_size_init);

        let mut popul: Vec<Individual> = Vec::with_capacity(pop_size_init * 2);

        for _ in 0..pop_size_init {
            let genome: Vec<f64> = bounds
                .iter()
                .map(|&(lower, upper)| master_rng.random_range(lower..upper))
                .collect();
            popul.push(Individual {
                genome,
                fitness: f64::INFINITY,
            });
        }

        let mut memory_cr = vec![1.0f64; self.memory_size];
        let mut memory_iter = 0;
        let mut success_rate = 0.5_f64;

        popul.par_iter_mut().for_each(|ind| {
            ind.fitness = self.problem.evaluate(&ind.genome);
        });

        let mut nf_eval = pop_size_init;

        let mut global_best_ind = popul[0].clone();
        for ind in &popul {
            if ind.fitness < global_best_ind.fitness {
                global_best_ind = ind.clone();
            }
        }

        while nf_eval < max_feval {
            let current_sol = Solution {
                genome: global_best_ind.genome.clone(),
                fitness: global_best_ind.fitness,
            };
            if !callback(&current_sol, nf_eval) {
                break;
            }

            popul.sort_by(|a, b| a.fitness.total_cmp(&b.fitness));

            let progress = nf_eval as f64 / max_feval as f64;
            let next_size_f = ((n_inds_min as f64 - n_inds_front_max as f64) * progress)
                + n_inds_front_max as f64;
            let next_size = next_size_f as usize;

            if popul.len() > n_inds_front {
                popul.truncate(n_inds_front);
            }
            n_inds_front = next_size.max(n_inds_min).min(popul.len());

            if popul[0].fitness < global_best_ind.fitness {
                global_best_ind = popul[0].clone();
            }

            let popul_front = popul.clone();
            let mean_f = 0.4_f64 + (success_rate * 5.0_f64).tanh() * 0.25_f64;
            let sigma_f = 0.02_f64;
            let sigma_cr = 0.05_f64;

            let dist_rank = if n_inds_front > 1 {
                let weights: Vec<f64> = (0..n_inds_front)
                    .map(|i| (-(i as f64) / n_inds_front as f64 * 3.0_f64).exp())
                    .collect();
                WeightedIndex::new(&weights).ok()
            } else {
                None
            };

            let p_size_val =
                (n_inds_front as f64 * 0.7_f64 * (-success_rate * 7.0_f64).exp()) as usize;
            let p_size_val = p_size_val.max(2).min(n_inds_front);

            let seeds: Vec<u64> = (0..n_inds_front).map(|_| master_rng.random()).collect();

            let results: Vec<_> = (0..n_inds_front)
                .into_par_iter()
                .zip(seeds.into_par_iter())
                .map(|(i, seed)| {
                    let mut local_rng = StdRng::seed_from_u64(seed);

                    let target_idx = i;
                    let mem_idx = local_rng.random_range(0..self.memory_size);

                    let mut prand_idx;
                    loop {
                        prand_idx = local_rng.random_range(0..p_size_val);
                        if prand_idx != target_idx {
                            break;
                        }
                    }

                    let mut rand1_idx;
                    loop {
                        if let Some(ref dist) = dist_rank {
                            rand1_idx = dist.sample(&mut local_rng);
                        } else {
                            rand1_idx = local_rng.random_range(0..n_inds_front);
                        }
                        if rand1_idx != prand_idx {
                            break;
                        }
                    }

                    let mut rand2_idx;
                    loop {
                        rand2_idx = local_rng.random_range(0..n_inds_front);
                        if rand2_idx != prand_idx && rand2_idx != rand1_idx {
                            break;
                        }
                    }

                    let mut f_val;
                    loop {
                        let z: f64 = local_rng.sample(StandardNormal);
                        f_val = mean_f + sigma_f * z;
                        if f_val >= 0.0 {
                            f_val = f_val.min(1.0);
                            break;
                        }
                    }

                    let z_cr: f64 = local_rng.sample(StandardNormal);
                    let mut cr_val = memory_cr[mem_idx] + sigma_cr * z_cr;
                    cr_val = cr_val.clamp(0.0, 1.0);

                    let x_target = &popul_front[target_idx].genome;
                    let x_pbest = &popul_front[prand_idx].genome;
                    let x_r1 = &popul_front[rand1_idx].genome;
                    let x_r2 = &popul_front[rand2_idx].genome;

                    let mut trial_genome = x_target.clone();
                    let j_rand = local_rng.random_range(0..n_vars);

                    for j in 0..n_vars {
                        if local_rng.random_bool(cr_val) || j == j_rand {
                            let val = x_target[j]
                                + f_val * (x_pbest[j] - x_target[j])
                                + f_val * (x_r1[j] - x_r2[j]);

                            let (min_j, max_j) = bounds[j];
                            if val < min_j || val > max_j {
                                trial_genome[j] = local_rng.random_range(min_j..max_j);
                            } else {
                                trial_genome[j] = val;
                            }
                        }
                    }

                    let trial_fit = self.problem.evaluate(&trial_genome);

                    (
                        target_idx,
                        Individual {
                            genome: trial_genome,
                            fitness: trial_fit,
                        },
                        cr_val,
                    )
                })
                .collect();

            let mut success_cr_list = Vec::new();
            let mut fit_delta_list = Vec::new();
            let mut new_children = Vec::new();

            for (target_idx, trial_ind, cr_val) in results {
                nf_eval += 1;

                if trial_ind.fitness <= popul_front[target_idx].fitness {
                    if trial_ind.fitness < global_best_ind.fitness {
                        global_best_ind = trial_ind.clone();
                    }
                    if trial_ind.fitness < popul_front[target_idx].fitness {
                        success_cr_list.push(cr_val);
                        fit_delta_list
                            .push((popul_front[target_idx].fitness - trial_ind.fitness).abs());
                    }
                    new_children.push(trial_ind);
                }

                if nf_eval >= max_feval {
                    break;
                }
            }

            let success_count = new_children.len();
            success_rate = success_count as f64 / n_inds_front as f64;

            popul.extend(new_children);

            if success_count > 0 {
                let sum_w: f64 = fit_delta_list.iter().sum();
                if sum_w > 1e-10 {
                    let mut mean_wl_cr = 0.0;
                    let mut sum_w_sq = 0.0;

                    for i in 0..success_cr_list.len() {
                        let w = fit_delta_list[i] / sum_w;
                        mean_wl_cr += w * success_cr_list[i] * success_cr_list[i];
                        sum_w_sq += w * success_cr_list[i];
                    }

                    let new_cr = if sum_w_sq > 0.0 {
                        mean_wl_cr / sum_w_sq
                    } else {
                        0.5
                    };

                    memory_cr[memory_iter] = 0.5 * new_cr + 0.5 * memory_cr[memory_iter];
                    memory_iter = (memory_iter + 1) % self.memory_size;
                }
            }
        }

        Ok(Solution {
            genome: global_best_ind.genome,
            fitness: global_best_ind.fitness,
        })
    }

    fn validate_config(&self) -> Result<ValidatedConfig, LsrtdeError> {
        let n_vars = self.problem.dimension();
        if n_vars == 0 {
            return Err(LsrtdeError::ZeroDimension);
        }

        if self.memory_size == 0 {
            return Err(LsrtdeError::ZeroMemorySize);
        }

        let pop_size_init = self
            .pop_size_multiplier
            .checked_mul(n_vars)
            .ok_or(LsrtdeError::PopulationSizeOverflow)?;

        if pop_size_init < 3 {
            return Err(LsrtdeError::PopulationTooSmall {
                population_size: pop_size_init,
                minimum: 3,
            });
        }

        let mut bounds = Vec::with_capacity(n_vars);
        for index in 0..n_vars {
            let (lower, upper) = self.problem.get_bounds(index);
            if !lower.is_finite() || !upper.is_finite() || lower >= upper {
                return Err(LsrtdeError::InvalidBounds {
                    index,
                    lower,
                    upper,
                });
            }
            bounds.push((lower, upper));
        }

        Ok(ValidatedConfig {
            n_vars,
            pop_size_init,
            bounds,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{Lsrtde, LsrtdeError, Problem};

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

    struct CountingProblem {
        dim: usize,
        evals: AtomicUsize,
    }

    impl CountingProblem {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                evals: AtomicUsize::new(0),
            }
        }

        fn evaluations(&self) -> usize {
            self.evals.load(Ordering::Relaxed)
        }
    }

    impl Problem for CountingProblem {
        fn dimension(&self) -> usize {
            self.dim
        }

        fn get_bounds(&self, _index: usize) -> (f64, f64) {
            (-1.0, 1.0)
        }

        fn evaluate(&self, genome: &[f64]) -> f64 {
            self.evals.fetch_add(1, Ordering::Relaxed);
            genome.iter().map(|x| x * x).sum()
        }
    }

    struct InvalidBoundsProblem {
        dim: usize,
        bounds: (f64, f64),
    }

    impl Problem for InvalidBoundsProblem {
        fn dimension(&self) -> usize {
            self.dim
        }

        fn get_bounds(&self, _index: usize) -> (f64, f64) {
            self.bounds
        }

        fn evaluate(&self, genome: &[f64]) -> f64 {
            genome.iter().sum()
        }
    }

    #[test]
    fn rejects_zero_dimension() {
        let problem = SphereProblem { dim: 0 };

        let err = Lsrtde::new(&problem).run().unwrap_err();

        assert_eq!(err, LsrtdeError::ZeroDimension);
    }

    #[test]
    fn rejects_zero_memory_size() {
        let problem = SphereProblem { dim: 10 };

        let err = Lsrtde::new(&problem).with_memory_size(0).run().unwrap_err();

        assert_eq!(err, LsrtdeError::ZeroMemorySize);
    }

    #[test]
    fn rejects_population_size_one() {
        let problem = SphereProblem { dim: 1 };

        let err = Lsrtde::new(&problem)
            .with_pop_size_multiplier(1)
            .run()
            .unwrap_err();

        assert_eq!(
            err,
            LsrtdeError::PopulationTooSmall {
                population_size: 1,
                minimum: 3,
            }
        );
    }

    #[test]
    fn rejects_population_size_two() {
        let problem = SphereProblem { dim: 1 };

        let err = Lsrtde::new(&problem)
            .with_pop_size_multiplier(2)
            .run()
            .unwrap_err();

        assert_eq!(
            err,
            LsrtdeError::PopulationTooSmall {
                population_size: 2,
                minimum: 3,
            }
        );
    }

    #[test]
    fn accepts_population_size_three() {
        let problem = SphereProblem { dim: 1 };

        let solution = Lsrtde::new(&problem)
            .with_pop_size_multiplier(3)
            .with_max_evaluations(10)
            .run()
            .unwrap();

        assert!(solution.fitness.is_finite());
        assert_eq!(solution.genome.len(), 1);
    }

    #[test]
    fn rejects_bounds_with_lower_ge_upper() {
        let problem = InvalidBoundsProblem {
            dim: 3,
            bounds: (1.0, 1.0),
        };

        let err = Lsrtde::new(&problem).run().unwrap_err();

        assert_eq!(
            err,
            LsrtdeError::InvalidBounds {
                index: 0,
                lower: 1.0,
                upper: 1.0,
            }
        );
    }

    #[test]
    fn rejects_non_finite_bounds() {
        let problem = InvalidBoundsProblem {
            dim: 3,
            bounds: (f64::NEG_INFINITY, 1.0),
        };

        let err = Lsrtde::new(&problem).run().unwrap_err();

        assert_eq!(
            err,
            LsrtdeError::InvalidBounds {
                index: 0,
                lower: f64::NEG_INFINITY,
                upper: 1.0,
            }
        );
    }

    #[test]
    fn max_evaluations_is_a_soft_budget() {
        let problem = CountingProblem::new(10);

        let solution = Lsrtde::new(&problem).with_max_evaluations(5).run().unwrap();

        assert!(solution.fitness.is_finite());
        assert_eq!(problem.evaluations(), 180);
    }
}
