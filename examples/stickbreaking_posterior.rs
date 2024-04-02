use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use peroxide::statistics::stat::Statistics;
use rv::experimental::*;
use rv::prelude::*;
use rv::experimental::stick_breaking::BreakSequence;     
use rv::traits::*;

fn main() {
    let mut rng = rand::thread_rng();
    let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());

    let num_samples = 1_000_000;

    // Our computed posterior
    let data = [10];
    let dist = sb.posterior(&DataOrSuffStat::Data(&data[..]));
    // let dist = sb.clone();

    // An approximation using rejection sampling
    let mut approx: Vec<Vec<f64>> = Vec::new();
    while approx.len() < num_samples {
        let seq: StickSequence = sb.draw(&mut rng);
        let sbd = Sbd::new(seq.clone());
        if sbd.draw(&mut rng) == 10 {
            approx.push(BreakSequence::from(&seq.weights(20)).0);
        }
    }


    let mut counts: Vec<Vec<f64>> = vec![];
    for j in 0..20 {
        counts.push(approx.iter().map(|breaks: &Vec<f64>| *breaks.get(j).unwrap()).collect())
    };



    let break_dists: Vec<Beta> = (0..19)
        .map(|n| {
            if n < dist.break_prefix.len() {
                dist.break_prefix[n].clone()
            } else {
                Beta::new_unchecked(dist.break_tail.alpha(), 1.0)
            }
        })
        .collect();

    counts.iter().enumerate().for_each(|(n, c)| {
        let data_mean: f64 = c.mean();
        let beta_mean: f64 = break_dists[n].mean().unwrap();
        println!(
            "n: {}\tmean: {:.3} (pred {:.3})\t var: {:.3} (pred {:.3})",
            n,
            data_mean,
            beta_mean,
            c.var(),
            break_dists[n].variance().unwrap(),
        );
    });

}
