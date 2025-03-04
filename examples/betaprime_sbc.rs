use rv::dist::BetaPrime;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rv::experimental::stick_breaking_process::{
    StickBreaking, StickBreakingDiscrete, StickBreakingDiscreteSuffStat,
};
use rv::prelude::*;

// Simulation-based calibration
// For details see http://www.stat.columbia.edu/~gelman/research/unpublished/sbc.pdf
fn main() {
    let mut rng = Xoshiro256Plus::seed_from_u64(123);
    let n_samples = 100000;
    let n_obs = 1;
    let n_bins = 100 ;
    let mut hist = vec![0_usize; n_bins + 1];

    let alpha_prior = BetaPrime::new(1.0, 1.0).unwrap();
    for _ in 0..n_samples {
        let alpha = alpha_prior.draw(&mut rng);
        let stick_breaking =
            StickBreaking::new(UnitPowerLaw::new(alpha).unwrap());
        let sbd: StickBreakingDiscrete = stick_breaking.draw(&mut rng);

        let mut stat = StickBreakingDiscreteSuffStat::new();
        for _ in 0..n_obs {
            let x = sbd.draw(&mut rng);
            stat.observe(&x);
        }

        let posterior = alpha_prior.posterior(&DataOrSuffStat::SuffStat(&stat));

        let mut q = 0;
        for _ in 0..n_bins {
            let alpha_hat: f64 = posterior.draw(&mut rng);
            if alpha_hat < alpha {
                q += 1;
            }
        }

        // Increment histogram bin
        hist[q] += 1;
    }

    // Should be uniform
    println!("{:?}", hist);
}
