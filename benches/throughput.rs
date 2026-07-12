use std::hint::black_box;
use std::time::Instant;

use l_srtde::{Lsrtde, Problem};

struct Sphere {
    dim: usize,
}

impl Problem for Sphere {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn get_bounds(&self, _index: usize) -> (f64, f64) {
        (-5.0, 5.0)
    }

    fn evaluate(&self, genome: &[f64]) -> f64 {
        genome.iter().map(|x| x * x).sum()
    }
}

fn main() {
    for dim in [16, 64, 256] {
        let problem = Sphere { dim };
        let population = dim * 3;
        let budget = population * 8;
        let mut samples = Vec::with_capacity(20);

        for run in 0..20 {
            let start = Instant::now();
            let solution = Lsrtde::new(&problem)
                .with_pop_size_multiplier(3)
                .with_max_evaluations(budget)
                .with_seed(run as u64)
                .run()
                .expect("benchmark problem is valid");
            black_box(solution);
            samples.push(start.elapsed().as_secs_f64() * 1_000.0);
        }

        samples.sort_by(f64::total_cmp);
        println!(
            "dim={dim:>3} population={population:>4} budget={budget:>5} median_ms={:.3} min_ms={:.3} max_ms={:.3}",
            samples[samples.len() / 2],
            samples[0],
            samples[samples.len() - 1]
        );
    }
}
