use crate::experimental::StickSequence;
use crate::traits::*;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct StickBreaking<B: Rv<f64> + Clone> {
    pub breaker: B,
}

impl<B:Rv<f64> + Clone> StickBreaking<B> {
    pub fn new(breaker: B) -> Self {
        Self { breaker }
    }
}

// impl<UnitPowerLaw> HasDensity<&[f64]> for StickBreaking<UnitPowerLaw> {
//     fn ln_f(&self, x: &&[f64]) -> f64 {
//         let stat = StickBreakingSuffStat::from(x);
//         self.ln_f_stat(&stat)
//     }
// }

impl<B: Rv<f64> + Clone> Sampleable<StickSequence<B>> for StickBreaking<B> {
    fn draw<R: Rng>(&self, rng: &mut R) -> StickSequence<B> {
        let seed: u64 = rng.gen();

        StickSequence::new(self.breaker.clone(), Some(seed))
    }
}


