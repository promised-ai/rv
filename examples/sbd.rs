use rv::experimental::Sbd;
use rv::experimental::SbdSuffStat;
use rv::experimental::StickBreaking;
use rv::experimental::StickSequence;
use rv::prelude::UnitPowerLaw;
use rv::traits::*;
use rv::prelude::DataOrSuffStat;

fn main() {
    // Instantiate a stick-breaking process
    let alpha = 5.0;
    let sbp = StickBreaking::new(UnitPowerLaw::new(alpha).unwrap());

    // Sample from it to get a StickSequence
    let sticks: StickSequence = sbp.draw(&mut rand::thread_rng());

    // Use the StickSequence to instantiate a stick-breaking discrete distribution
    let sbd = Sbd::new(sticks.clone());

    // Now sample from the Sbd and find its sufficient statistic
    let n = 10000;
    let xs = sbd.sample(n, &mut rand::thread_rng());
    let stat = SbdSuffStat::from(&xs[..]);

    // Use the sufficient statistic to find the posterior
    let post = sbp.posterior_from_suffstat(&stat);

    // Print the posterior parameters of each Beta distribution.
    post.prefix.iter().for_each(|p| {
        let alpha = p.alpha();
        let beta = p.beta();
        let p = alpha / (alpha + beta);
        println!("alpha: {}\t beta: {}\t mean: {}", alpha, beta, p);
    });

    let cache = sbp.ln_pp_cache(&DataOrSuffStat::SuffStat(&stat));
    let sum: f64 = (0..99).map(|y| sbp.pp_with_cache(&cache, &y)).sum();
    println!("P(posterior predictive < 100) {}", sum);

}
