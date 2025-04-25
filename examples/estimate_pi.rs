// Use a Monte Carlo method known as 'rejection sampling to estimate the value
// of pi. We will draw samples from a 2-by-2 square, within which there is a
// perfectly inscribed circle. We then use an estimate of the ratio of the
// areas to estimate pi.
//
//  A_circle      pi * r^2    pi         # in circle
// ----------  =  -------- = ----  => 4 ------------- ~= pi
//  A_square      4 * r^2      4         # in square
//
use rv::dist::Uniform;
use rv::traits::Sampleable;
use std::f64::consts::PI;

fn main() {
    // The number of samples to use for the Monte Carlo estimate
    let n_samples: usize = 1_000_000;

    // Two samples from this gets us uniform samples in a 2-by-2 square
    let u = Uniform::new(-1.0, 1.0).unwrap();

    // The rand number steam consumes the rng
    let mut rng1 = rand::thread_rng();
    let mut rng2 = rand::thread_rng();

    let pi_est = u
        .sample_stream(&mut rng1)
        .zip(u.sample_stream(&mut rng2))
        .take(n_samples)
        .map(
            // Count the number of times the random point lands in the circle
            |(x, y): (f64, f64)| {
                if x * x + y * y < 1.0 {
                    1.0
                } else {
                    0.0
                }
            },
        )
        .sum::<f64>()
        // divide by the total number of samples in the square
        / (n_samples as f64)
        // and multiply by 4 according to the above solution
        * 4.0;

    println!(
        "π_est: {}, π_true: {}, absolute error: {}",
        pi_est,
        PI,
        (pi_est - PI).abs()
    );
}
