#![feature(type_ascription)]
extern crate rand;
extern crate rv;

use rv::data::DataOrSuffStat;
use rv::dist::{Categorical, SymmetricDirichlet};
use rv::prelude::CategoricalData;
use rv::traits::*;

fn main() {
    let mut rng = rand::thread_rng();

    // Roll a die (0...5) that comes up 5 half the time
    let ctgrl = Categorical::new(&vec![1.0, 1.0, 1.0, 1.0, 1.0, 5.0]).unwrap();
    let rolls: Vec<u8> = ctgrl.sample(1000, &mut rng);

    // Use the Jeffreys prior of Dir(1/2, ..., 1/2)
    let prior = SymmetricDirichlet::jeffreys(6).unwrap();
    let obs: CategoricalData<u8> = DataOrSuffStat::Data(&rolls);

    // Log marginal likelihood of the observed rolls
    println!("Log P(rolls) = {}", prior.ln_m(&obs));

    // Posterior predictive probability of the next die roll being 5 given the
    // observed rolls.
    let pp_5: f64 = prior.pp(&5_u8, &obs);
    println!("P(y = 5 | rolls) = {}", pp_5);

    // Draw a sample from the posterior distributoin P(θ|x)
    let weights: Vec<f64> = prior.posterior(&obs).draw(&mut rng);
    println!("Die weight sample: {:?}", weights);
}
