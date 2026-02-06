//! # L-SRTDE: Large Scale Random Topology Differential Evolution
//!
//! L-SRTDE 是一种针对大规模数值优化问题设计的高性能差分进化算法。
//!
//! 该算法的主要创新在于解决了标准差分进化算法 (DE) 中对 **缩放因子 (Scaling Factor, F)** 高度敏感的问题。
//! 不同于传统的基于历史成功记录 (Success History-based, 如 SHADE) 的方法，L-SRTDE 直接利用当前代的
//! **改进解比例 (Success Rate)** 来动态调整 F。
//!
//! ## 核心机制 (Core Mechanism)
//!
//! ### 1. 基于成功率的参数自适应 (Success Rate Adaptation)
//!
//! 算法通过计算当前代中产生更优解的个体比例（即成功率 `SR`），来实时调整缩放因子 `F`。
//! 这种机制使得算法能够根据当前的搜索状态（探索 vs 开发）快速响应。
//!
//! 计算公式如下：
//!
//! ```text
//! F = 0.4 + 0.25 * tanh(5.0 * SR)
//! ```
//!
//! * 当 **SR 较高**时（容易找到更好解），`F` 增大，增强全局探索能力。
//! * 当 **SR 较低**时（陷入局部或难以改进），`F` 减小，转向局部精细开发。
//!
//! ### 2. 混合策略
//! * **F 参数**：基于即时成功率 (Success Rate)。
//! * **CR 参数**：基于历史记忆库 (Memory-based)，类似于 SHADE 算法。
//! * **拓扑结构**：采用随机拓扑结构以适应大规模维度。
//!
//! ## 快速开始 (Quick Start)
//!
//! ```rust
//! use l_srtde::{Lsrtde, Problem, Solution};
//!
//! // 1. 定义你的问题
//! struct SphereProblem {
//!     dim: usize,
//! }
//!
//! impl Problem for SphereProblem {
//!     fn dimension(&self) -> usize { self.dim }
//!     
//!     fn get_bounds(&self, _index: usize) -> (f64, f64) {
//!         (-100.0, 100.0)
//!     }
//!     
//!     fn evaluate(&self, genome: &[f64]) -> f64 {
//!         // 目标是最小化平方和
//!         genome.iter().map(|x| x * x).sum()
//!     }
//! }
//!
//!
//!// 2. 初始化问题实例
//!let problem = SphereProblem { dim: 100 };
//!
//!// 3. 配置并运行求解器
//!let solver = Lsrtde::new(&problem)
//!     .with_max_evaluations(500_000)
//!     .with_seed(42); // 固定种子以便复现
//!
//!let solution = solver.run();
//!
//!println!("Best Fitness: {}", solution.fitness);
//!// 此时 fitness 应该非常接近 0.0
//!assert!(solution.fitness < 1e-5);
//!
//! ```
//! ## 参考文献
//!
//! * *V. Stanovov and E. Semenkin, "Success Rate-based Adaptive Differential Evolution L-SRTDE for CEC 2024 Competition," 2024 IEEE Congress on Evolutionary Computation (CEC), Yokohama, Japan, 2024, pp. 1-8, doi: 10.1109/CEC60901.2024.10611907. keywords: {Sensitivity;Accuracy;Evolutionary computation;Benchmark testing;Optimization;differential evolution;numerical optimization;parameter adaptation}*

use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// 定义优化问题的 Trait。
///
/// 任何想要使用 `Lsrtde`求解器的问题都需要实现此 Trait。
/// 由于求解器内部使用了 `rayon` 进行并行计算，因此实现必须是 `Sync` 的。
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
///     fn dimension(&self) -> usize { self.dim }
///     fn get_bounds(&self, _index: usize) -> (f64, f64) { (-100.0, 100.0) }
///     fn evaluate(&self, genome: &[f64]) -> f64 {
///         genome.iter().map(|x| x * x).sum()
///     }
/// }
/// ```
pub trait Problem: Sync {
    /// 返回问题的维度（决策变量的数量）。
    fn dimension(&self) -> usize;

    /// 获取指定维度的上下界。
    ///
    /// # Arguments
    /// * `index` - 维度的索引 (0 到 dimension - 1)。
    ///
    /// # Returns
    /// 一个元组 `(min, max)`，表示该维度的取值范围。
    fn get_bounds(&self, index: usize) -> (f64, f64);

    /// 计算给定解（基因组）的适应度值。
    ///
    /// 通常是一个最小化问题，值越小越好。
    fn evaluate(&self, genome: &[f64]) -> f64;
}

/// 表示优化算法的一个解。
#[derive(Debug, Clone)]
pub struct Solution {
    /// 最优解的变量向量。
    pub genome: Vec<f64>,
    /// 该解对应的适应度值 (Fitness)。
    pub fitness: f64,
}

/// 内部使用的个体结构体，包含基因和适应度。
#[derive(Clone, Debug)]
struct Individual {
    genome: Vec<f64>,
    fitness: f64,
}

/// LSRTDE (Large Scale Random Topology Differential Evolution) 优化算法求解器。
///
/// 该求解器实现了自适应差分进化算法，包含以下特性：
/// * **并行计算**：利用 `rayon` 并行评估种群。
/// * **LPSR**：线性种群规模缩减 (Linear Population Size Reduction)。
/// * **自适应参数**：基于历史成功记录自适应调整缩放因子 (F) 和交叉概率 (CR)。
///
/// # Type Parameters
/// * `P` - 实现了 `Problem` Trait 的具体问题类型。
pub struct Lsrtde<'a, P: Problem> {
    problem: &'a P,
    /// 最大允许的评估次数 (Function Evaluations)。
    max_evaluations: usize,
    /// 历史记忆库的大小，用于参数自适应。
    memory_size: usize,
    /// 初始种群规模倍数 (Population = dimension * multiplier)。
    pop_size_multiplier: usize,
    /// 随机数种子。
    seed: Option<u64>,
}

impl<'a, P: Problem> Lsrtde<'a, P> {
    /// 创建一个新的求解器实例。
    ///
    /// 使用默认参数初始化：
    /// * `max_evaluations`: 100,000
    /// * `memory_size`: 5
    /// * `pop_size_multiplier`: 18
    ///
    /// # Arguments
    /// * `problem` - 实现了 `Problem` trait 的对象引用。
    pub fn new(problem: &'a P) -> Self {
        Self {
            problem,
            max_evaluations: 100_000,
            memory_size: 5,
            pop_size_multiplier: 18,
            seed: None,
        }
    }

    /// 设置最大评估次数。
    ///
    /// 这是算法的终止条件之一。
    pub fn with_max_evaluations(mut self, n: usize) -> Self {
        self.max_evaluations = n;
        self
    }

    /// 设置历史记忆库的大小。
    ///
    /// 影响参数自适应的学习速率。
    pub fn with_memory_size(mut self, size: usize) -> Self {
        self.memory_size = size;
        self
    }

    /// 设置初始种群规模倍数。
    ///
    /// 初始种群大小将计算为 `multiplier * problem.dimension()`。
    pub fn with_pop_size_multiplier(mut self, multiplier: usize) -> Self {
        self.pop_size_multiplier = multiplier;
        self
    }

    /// 设置随机数种子。
    ///
    /// 设置种子可以确保结果的可复现性。如果未设置，将使用系统熵生成随机种子。
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// 运行优化算法直到达到最大评估次数。
    ///
    /// 此方法是 `run_with_callback` 的简化版本，不使用回调函数。
    ///
    /// # Returns
    /// 找到的全局最优解 `Solution`。
    #[inline]
    pub fn run(&self) -> Solution {
        self.run_with_callback(|_, _| true)
    }

    /// 运行优化算法，并支持每代回调。
    ///
    /// # Arguments
    /// * `callback` - 一个闭包 `FnMut(&Solution, usize) -> bool`。
    ///     * 第一个参数是当前的全局最优解。
    ///     * 第二个参数是当前的评估次数 (evaluations count)。
    ///     * 返回 `true` 继续运行，返回 `false` 提前终止算法。
    ///
    /// # Returns
    /// 终止时找到的全局最优解。
    pub fn run_with_callback<F>(&self, mut callback: F) -> Solution
    where
        F: FnMut(&Solution, usize) -> bool,
    {
        let n_vars = self.problem.dimension();
        let max_feval = self.max_evaluations;

        // 1. 初始化 RNG
        // 如果用户提供了种子，则使用它；否则从系统获取随机数
        let master_seed = self.seed.unwrap_or_else(|| rand::rng().random());
        let mut master_rng = StdRng::seed_from_u64(master_seed);

        // 2. 参数初始化
        let pop_size_init = self.pop_size_multiplier * n_vars;
        let mut n_inds_front = pop_size_init;
        let n_inds_front_max = pop_size_init;
        // 最小种群规模不低于 4
        let n_inds_min = 4.min(pop_size_init);

        let mut popul: Vec<Individual> = Vec::with_capacity(pop_size_init * 2);

        // 3. 初始种群生成
        for _ in 0..pop_size_init {
            let genome: Vec<f64> = (0..n_vars)
                .map(|i| {
                    let (l, r) = self.problem.get_bounds(i);
                    master_rng.random_range(l..r)
                })
                .collect();
            popul.push(Individual {
                genome,
                fitness: f64::INFINITY,
            });
        }

        // C++ 中初始为 1.0 (Initialize 函数末尾)
        let mut memory_cr = vec![1.0f64; self.memory_size];
        let mut memory_iter = 0;
        let mut success_rate = 0.5_f64;

        // 4. 评估初始种群 (并行)
        popul.par_iter_mut().for_each(|ind| {
            ind.fitness = self.problem.evaluate(&ind.genome);
        });

        let mut nf_eval = pop_size_init;

        // 寻找初始最优解
        let mut global_best_ind = popul[0].clone();
        for ind in &popul {
            if ind.fitness < global_best_ind.fitness {
                global_best_ind = ind.clone();
            }
        }

        // === 主循环 ===
        while nf_eval < max_feval {
            let current_sol = Solution {
                genome: global_best_ind.genome.clone(),
                fitness: global_best_ind.fitness,
            };
            // 触发回调，检查是否需要提前终止
            if !callback(&current_sol, nf_eval) {
                break;
            }

            // 5. 排序 (为了 LPSR 和 rank-based selection)
            popul.sort_by(|a, b| a.fitness.total_cmp(&b.fitness));

            // 6. 种群缩减 (LPSR) - 逻辑与 C++ 一致
            // 根据评估进度线性减少种群规模
            let progress = nf_eval as f64 / max_feval as f64;
            let next_size_f = ((n_inds_min as f64 - n_inds_front_max as f64) * progress)
                + n_inds_front_max as f64;
            let next_size = next_size_f as usize;

            if popul.len() > n_inds_front {
                popul.truncate(n_inds_front);
            }
            n_inds_front = next_size.max(n_inds_min).min(popul.len());

            // 更新最优解（因为排序后 index 0 也是当前种群最优，但 global_best 可能更好）
            if popul[0].fitness < global_best_ind.fitness {
                global_best_ind = popul[0].clone();
            }

            let popul_front = popul.clone();

            // 7. 自适应参数 (完全对齐 C++ 逻辑)
            let mean_f = 0.4_f64 + (success_rate * 5.0_f64).tanh() * 0.25_f64;

            // --- 修正点 2: 调整 Sigma 为 C++ 原版数值 ---
            let sigma_f = 0.02_f64; // C++: 0.02
            let sigma_cr = 0.05_f64; // C++: 0.05

            // 计算排名权重 (Weighted Selection)
            let dist_rank = if n_inds_front > 1 {
                let weights: Vec<f64> = (0..n_inds_front)
                    .map(|i| (-(i as f64) / n_inds_front as f64 * 3.0_f64).exp())
                    .collect();
                WeightedIndex::new(&weights).ok()
            } else {
                None
            };

            // P-Best 大小计算
            let p_size_val =
                (n_inds_front as f64 * 0.7_f64 * (-success_rate * 7.0_f64).exp()) as usize;
            let p_size_val = p_size_val.max(2).min(n_inds_front);

            // 8. 并行生成新一代
            // 预先生成种子以保证并行确定性
            let seeds: Vec<u64> = (0..n_inds_front).map(|_| master_rng.random()).collect();

            let results: Vec<_> = (0..n_inds_front)
                .into_par_iter()
                .zip(seeds.into_par_iter())
                .map(|(i, seed)| {
                    let mut local_rng = StdRng::seed_from_u64(seed);

                    // 注意：Rust 这里使用 i 作为 target (遍历模式)，比 C++ 的随机选取更稳定
                    let target_idx = i;
                    let mem_idx = local_rng.random_range(0..self.memory_size);

                    // 选择 p-best
                    let mut prand_idx;
                    loop {
                        prand_idx = local_rng.random_range(0..p_size_val);
                        if prand_idx != target_idx {
                            break;
                        }
                    }

                    // 选择 r1 (基于排名权重)
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

                    // 选择 r2 (随机)
                    let mut rand2_idx;
                    loop {
                        rand2_idx = local_rng.random_range(0..n_inds_front);
                        if rand2_idx != prand_idx && rand2_idx != rand1_idx {
                            break;
                        }
                    }

                    // 参数生成 F
                    let mut f_val;
                    loop {
                        let z: f64 = local_rng.sample(StandardNormal);
                        f_val = mean_f + sigma_f * z;
                        if f_val >= 0.0 {
                            f_val = f_val.min(1.0);
                            break;
                        }
                    }

                    // CR 查 Memory 表并变异
                    let z_cr: f64 = local_rng.sample(StandardNormal);
                    let mut cr_val = memory_cr[mem_idx] + sigma_cr * z_cr;
                    cr_val = cr_val.clamp(0.0, 1.0);

                    // 变异交叉 (current-to-pbest/1)
                    let x_target = &popul_front[target_idx].genome;
                    let x_pbest = &popul_front[prand_idx].genome;
                    let x_r1 = &popul_front[rand1_idx].genome;
                    let x_r2 = &popul_front[rand2_idx].genome;

                    let mut trial_genome = x_target.clone();
                    let j_rand = local_rng.random_range(0..n_vars);

                    for j in 0..n_vars {
                        if local_rng.random_bool(cr_val.into()) || j == j_rand {
                            let val = x_target[j]
                                + f_val * (x_pbest[j] - x_target[j])
                                + f_val * (x_r1[j] - x_r2[j]);

                            let (min_j, max_j) = self.problem.get_bounds(j);
                            // 边界处理：随机重置
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

            // 9. 更新逻辑 (串行处理结果)
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

            // 10. Memory 更新 (仅更新 CR，基于 Weighted Lehmer Mean)
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

        Solution {
            genome: global_best_ind.genome,
            fitness: global_best_ind.fitness,
        }
    }
}
