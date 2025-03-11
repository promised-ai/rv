#[cfg(feature = "experimental")]
use rv::dist::BetaPrime;

#[cfg(feature = "experimental")]
use rand::SeedableRng;
#[cfg(feature = "experimental")]
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "experimental")]
use rv::experimental::stick_breaking_process::{
    StickBreaking, StickBreakingDiscrete, StickBreakingDiscreteSuffStat,
};
#[cfg(feature = "experimental")]
use rv::prelude::*;

// Simulation-based calibration
// For details see http://www.stat.columbia.edu/~gelman/research/unpublished/sbc.pdf
#[cfg(feature = "experimental")]
fn main() {
    let mut rng = Xoshiro256Plus::seed_from_u64(123);
    let n_samples = 10000;
    let n_obs = 10;
    let n_bins = 100;
    let mut hist = vec![0_usize; n_bins + 1];

    let alpha_prior = BetaPrime::new(1.0, 1.0).unwrap();

    // Comments in this section are from Algorithm 1 of the SBC paper
    for _ in 0..n_samples {
        // Draw a prior sample, θ̃ ∼ π(θ)
        let alpha = alpha_prior.draw(&mut rng);

        // Draw a simulated data set, ỹ ∼ π(y | θ̃)
        let mut stat = StickBreakingDiscreteSuffStat::new();
        for _ in 0..n_obs {
            let stick_breaking =
                StickBreaking::new(UnitPowerLaw::new(alpha).unwrap());
            let sbd: StickBreakingDiscrete = stick_breaking.draw(&mut rng);
            let x = sbd.draw(&mut rng);
            stat.observe(&x);
        }

        let posterior = alpha_prior.posterior(&DataOrSuffStat::SuffStat(&stat));

        let mut q = 0;
        for _ in 0..n_bins {
            // Draw posterior samples {θ₁, . . . , θₗ} ∼ π(θ | ỹ)
            let alpha_hat: f64 = posterior.draw(&mut rng);

            // Compute the rank statistic
            if alpha_hat < alpha {
                q += 1;
            }
        }

        // Increment the histogram
        hist[q] += 1;
    }

    // Should be uniform
    println!("{:?}", hist);
}

#[cfg(not(feature = "experimental"))]
fn main() {
    println!("This example requires the 'experimental' feature");
}
