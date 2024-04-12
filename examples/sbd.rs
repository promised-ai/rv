use rand::SeedableRng;
use rv::prelude::*;

#[cfg(feature = "experimental")]
use rv::experimental::stick_breaking::{
    StickBreaking, StickBreakingDiscrete, StickSequence,
};

fn main() {
    #[cfg(feature = "experimental")]
    {
        // Instantiate a stick-breaking process
        let alpha = 10.0;
        let sbp = StickBreaking::new(UnitPowerLaw::new(alpha).unwrap());

        // Sample from it to get a StickSequence
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let sticks: StickSequence = sbp.draw(&mut rng);

        // Use the StickSequence to instantiate a stick-breaking discrete distribution
        let sbd = StickBreakingDiscrete::new(sticks.clone());

        let start = std::time::Instant::now();
        let entropy = sbd.entropy();
        let duration = start.elapsed();
        println!("Entropy: {}", entropy);
        println!("Time elapsed in entropy() is: {:?}", duration);

        let num_weights = sbd.stick_sequence().num_weights_unstable();

        println!("num weights: {}", num_weights);
    }
}
