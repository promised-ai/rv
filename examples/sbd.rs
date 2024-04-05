#[cfg(feature = "experimental")]
use rv::experimental::{
    StickBreaking, StickBreakingDiscrete, StickBreakingDiscreteSuffStat,
    StickSequence,
};

fn main() {
    #[cfg(feature = "experimental")]
    {
        // Instantiate a stick-breaking process
        let alpha = 5.0;
        let sbp = StickBreaking::new(UnitPowerLaw::new(alpha).unwrap());

        // Sample from it to get a StickSequence
        let sticks: StickSequence = sbp.draw(&mut rand::thread_rng());

        // Use the StickSequence to instantiate a stick-breaking discrete distribution
        let sbd = StickBreakingDiscrete::new(sticks.clone());

        // Now sample from the StickBreakingDiscrete and find its sufficient statistic
        let n = 10000;
        let xs = sbd.sample(n, &mut rand::thread_rng());
        let stat = StickBreakingDiscreteSuffStat::from(&xs[..]);

        // Use the sufficient statistic to find the posterior
        let post = sbp.posterior_from_suffstat(&stat);

        let mut break_iter = post.break_dists();

        // Print the posterior parameters of each Beta distribution.
        while let Some(Either::Left(p)) = break_iter.next() {
            let mean: f64 = p.mean().unwrap();
            println!(
                "alpha: {}\t beta: {}\t mean: {}",
                p.alpha(),
                p.beta(),
                mean
            );
        }

        let cache = sbp.ln_pp_cache(&DataOrSuffStat::SuffStat(&stat));
        let sum: f64 = (0..99).map(|y| sbp.pp_with_cache(&cache, &y)).sum();
        println!("P(posterior predictive < 100) {}", sum);
    }
}
