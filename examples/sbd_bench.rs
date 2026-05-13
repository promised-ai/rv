use rayon::prelude::*;
use rv::traits::{HasDensity, Sampleable};
use std::hint::black_box;
use std::time::Instant;

use rv::experimental::stick::StickBreakingDiscrete;

pub fn format_with_commas(val: f64) -> String {
    if !val.is_finite() {
        return val.to_string();
    }

    let s = format!("{:.2}", val);
    let mut parts = s.split('.');

    let int_part = parts.next().unwrap();
    // {:.2} formatting on a finite float guarantees a fractional part
    let frac_part = parts.next().unwrap();

    let (sign, digits) = if let Some(stripped) = int_part.strip_prefix('-') {
        ("-", stripped)
    } else {
        ("", int_part)
    };

    let mut result = String::with_capacity(s.len() + s.len() / 3);
    result.push_str(sign);

    let len = digits.len();
    for (i, c) in digits.chars().enumerate() {
        result.push(c);
        let rem = len - i - 1;
        // Insert a comma if there are multiples of 3 digits remaining
        if rem > 0 && rem % 3 == 0 {
            result.push(',');
        }
    }

    result.push('.');
    result.push_str(frac_part);

    result
}

fn main() {
    let n_reps = 4_u64;
    let n_evals = 1_000_000_usize;

    let mut rng = rand::rng();
    let src = StickBreakingDiscrete::from_alpha(1.0, Some(1337)).unwrap();

    let mut times = Vec::new();
    for n_threads in 1..=8 {
        for i in 0..n_reps {
            let seed = 1337 + i;
            let sbd =
                StickBreakingDiscrete::from_alpha(1.0, Some(seed)).unwrap();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .unwrap();

            let xs: Vec<usize> = src.sample(n_evals, &mut rng);
            pool.install(|| {
                let t_start = Instant::now();
                xs.par_chunks(n_evals.div_ceil(n_threads))
                    .for_each(|chunk| {
                        for x in chunk {
                            let ln_f = sbd.ln_f(x);
                            black_box(ln_f);
                        }
                    });
                let t_end = t_start.elapsed();
                let ln_f_sec = n_evals as f64 / t_end.as_secs_f64();
                eprintln!(
                    "{n_threads}: {t_end:?}\t{} f/sec",
                    format_with_commas(ln_f_sec)
                );
                times.push((n_threads, t_end));
            });
        }
    }
}
