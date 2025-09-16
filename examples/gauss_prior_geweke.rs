// Runs the Geweke test many times on the Gaussian conjugate priors and compares
// the test performance by way of mean and variance. Remember, a lower error is
// better.
use rv::dist::{NormalGamma, NormalInvChiSquared, NormalInvGamma};
use rv::test::GewekeTester;
use std::collections::BTreeMap;
use std::io::{stdout, Write};

// Number of tests to run per prior
const N_RUNS: usize = 100;
// Number of forward and posterior samples to collect
const N_STEPS: usize = 5_000;
// Number of steps between posterior chain sample collection. This helps to
// reduce some of the auto-correlation that can mess with the test results.
const THINNING: usize = 10;

// Print a thing immediately
macro_rules! print_flush {
    ( $($t:tt)* ) => {
        {
            let mut h = stdout();
            write!(h, $($t)* ).unwrap();
            h.flush().unwrap();
        }
    }
}

// Compute the mean and variance
fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var)
}

// A macro is a little easier to write than a generic function, because the
// compiler does type inference.
macro_rules! geweke {
    ($pr: expr) => {{
        let mut rng = rand::rng();
        let mut results: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for i in 0..N_RUNS {
            print_flush!(".");
            let mut tester = GewekeTester::new($pr.clone(), 20);
            tester.run_chains(N_STEPS, THINNING, &mut rng);
            let errs = tester.errs();
            if i == 0 {
                errs.iter().for_each(|(key, _)| {
                    results.insert(key.clone(), Vec::with_capacity(N_RUNS));
                });
            }
            errs.iter().for_each(|(key, val)| {
                results.get_mut(key).unwrap().push(*val);
            });
        }
        println!("+");
        results.iter().for_each(|(k, xs)| {
            let (mean, var) = mean_var(xs);
            println!("{} (m, v): ({}, {})", k, mean, var);
        })
    }};
}

fn main() {
    // Note that we have chose the same parameter values for each distribution.
    // This may not exactly be apples-to-apples because the parameters have
    // different meanings in each distribution.
    println!("Running NormalGamma");
    let ng = NormalGamma::new(0.0, 1.0, 2.0, 3.0).unwrap();
    geweke!(ng);

    println!("Running NormalInvGamma");
    let nig = NormalInvGamma::new(0.0, 1.0, 2.0, 3.0).unwrap();
    geweke!(nig);

    println!("Running NormalInvChiSquared");
    let nix = NormalInvChiSquared::new(0.0, 1.0, 2.0, 3.0).unwrap();
    geweke!(nix);
}
