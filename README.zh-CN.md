# l_srtde

[English](README.md) | [简体中文](README.zh-CN.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)

`l_srtde` 是 L-SRTDE（Large Scale Random Topology Differential Evolution，大规模随机拓扑差分进化）算法的 Rust 实现，面向大规模连续数值优化问题。

这个库的核心目标是：保留一个高性能 Rust 算法内核，同时提供 C ABI 动态库能力，让 C、C++、Python `ctypes` 等非 Rust 项目也能调用同一份算法实现。

## 特性

- 使用 `rayon` 并行评估种群
- 基于成功率自适应调整缩放因子 `F`
- 线性种群规模缩减（LPSR）
- 面向大规模搜索空间的随机拓扑策略
- 原生 Rust API
- C ABI 动态库接口，方便非 Rust 语言调用

## 安装

Rust 项目中直接使用：

```toml
[dependencies]
l_srtde = "0.1.3"
```

## Rust 快速开始

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

## 参数配置

```rust
let solver = Lsrtde::new(&problem)
    .with_max_evaluations(1_000_000)
    .with_pop_size_multiplier(18)
    .with_memory_size(5)
    .with_seed(12345);
```

也可以通过 callback 监控优化进度，或者提前停止：

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

## C ABI / 动态库

这个 crate 可以构建成动态库，供 C、C++、Python `ctypes` 以及其他能加载 C ABI 动态库的语言调用。

构建 release 动态库：

```bash
cargo build --release
```

构建产物在 `target/release`：

- Windows: `l_srtde.dll`
- Linux: `libl_srtde.so`
- macOS: `libl_srtde.dylib`

C 语言使用 `include/l_srtde.h`：

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

callback 收到的是 row-major 连续数组，长度为 `point_count * dim`。第 `i` 个候选解的第 `j` 个变量访问方式是：

```c
points[i * dim + j]
```

callback 需要写出 `point_count` 个 fitness：

```c
fitness_out[i] = value;
```

返回 `0` 表示成功；返回非 `0` 会中止优化，并让 `lsrtde_minimize` 返回 `LSRTDE_CALLBACK_ERROR`。

编译链接示例：

```bash
# Linux/macOS
cc main.c -Iinclude -Ltarget/release -ll_srtde -o main

# Windows MSVC
cl main.c /I include target\release\l_srtde.dll.lib

# Windows MinGW gcc
gcc main.c -Iinclude target\release\l_srtde.dll -o target\release\main.exe
```

Windows 上运行时，确保 `l_srtde.dll` 在可执行文件同目录，或在 `PATH` 中。

## Python ctypes 示例

当前版本不是 Python 包，所以 Python 里不是 `import l_srtde`，而是用 `ctypes` 加载 Rust 编译出来的动态库。

```python
import ctypes
from pathlib import Path


class LsrtdeConfig(ctypes.Structure):
    _fields_ = [
        ("dim", ctypes.c_size_t),
        ("lower_bounds", ctypes.POINTER(ctypes.c_double)),
        ("upper_bounds", ctypes.POINTER(ctypes.c_double)),
        ("max_evaluations", ctypes.c_size_t),
        ("memory_size", ctypes.c_size_t),
        ("pop_size_multiplier", ctypes.c_size_t),
        ("seed", ctypes.c_uint64),
        ("use_seed", ctypes.c_uint8),
    ]


CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
)


@CALLBACK
def sphere_batch(points, point_count, dim, fitness_out, user_data):
    for i in range(point_count):
        total = 0.0
        for j in range(dim):
            x = points[i * dim + j]
            total += x * x
        fitness_out[i] = total
    return 0


lib = ctypes.CDLL(str(Path("target/release/l_srtde.dll").resolve()))

lib.lsrtde_minimize.argtypes = [
    ctypes.POINTER(LsrtdeConfig),
    CALLBACK,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
lib.lsrtde_minimize.restype = ctypes.c_int32

lib.lsrtde_error_message.argtypes = [ctypes.c_int32]
lib.lsrtde_error_message.restype = ctypes.c_char_p

dim = 10
DoubleArray = ctypes.c_double * dim

lower = DoubleArray(*([-100.0] * dim))
upper = DoubleArray(*([100.0] * dim))
best_x = DoubleArray()
best_fitness = ctypes.c_double()

config = LsrtdeConfig(
    dim=dim,
    lower_bounds=lower,
    upper_bounds=upper,
    max_evaluations=100_000,
    memory_size=5,
    pop_size_multiplier=18,
    seed=42,
    use_seed=1,
)

status = lib.lsrtde_minimize(
    ctypes.byref(config),
    sphere_batch,
    None,
    best_x,
    ctypes.byref(best_fitness),
)

if status != 0:
    message = lib.lsrtde_error_message(status).decode()
    raise RuntimeError(f"l_srtde failed: {message}")

print("best fitness:", best_fitness.value)
print("best x:", list(best_x))
```

## 校验规则和评估预算

`run()` 和 `run_with_callback()` 返回 `Result<_, LsrtdeError>`。求解器会在并行评估开始前拒绝非法配置。

当前校验规则：

- `dimension() > 0`
- `memory_size > 0`
- `dimension * pop_size_multiplier` 不能发生整数溢出
- 初始种群规模至少为 `3`
- 每一维的 `(lower, upper)` 必须是有限数，并且满足 `lower < upper`

`with_max_evaluations()` 是软预算，不是严格硬上限：

- 初始种群总是完整评估
- 每一代 trial vector 会按整批并行评估
- 实际目标函数调用次数可能超过配置的预算
- 超出的数量最多不超过当前代种群规模

## 参考文献

本 crate 实现的是 V. Stanovov 和 E. Semenkin 提出的 L-SRTDE 算法。如果你在研究中使用这个算法或代码，请引用原论文：

> V. Stanovov and E. Semenkin, "Success Rate-based Adaptive Differential Evolution L-SRTDE for CEC 2024 Competition," 2024 IEEE Congress on Evolutionary Computation (CEC), Yokohama, Japan, 2024, pp. 1-8, doi: [10.1109/CEC60901.2024.10611907](https://doi.org/10.1109/CEC60901.2024.10611907).

## 许可证

本项目使用 MIT license。详见 [LICENSE](LICENSE)。
