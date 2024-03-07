use crate::experimental::Sbd;
use crate::experimental::SbdSuffStat;
use crate::experimental::StickBreakingSuffStat;
use crate::experimental::StickSequence;
use crate::prelude::Beta;
use crate::prelude::DataOrSuffStat;
use crate::prelude::UnitPowerLaw;
use crate::suffstat_traits::HasSuffStat;
use crate::suffstat_traits::SuffStat;
use crate::traits::*;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct StickBreaking {
    pub breaker: UnitPowerLaw,
    pub prefix: Vec<Beta>,
}

impl StickBreaking {
    pub fn new(breaker:UnitPowerLaw) -> Self {
        let prefix = Vec::new();
        Self { breaker, prefix }
    }
}

impl HasDensity<&[f64]> for StickBreaking {
    fn ln_f(&self, x: &&[f64]) -> f64 {
        let stat = StickBreakingSuffStat::from(x);
        self.ln_f_stat(&stat)
    }
}

impl Sampleable<StickSequence> for StickBreaking {
    fn draw<R: Rng>(&self, rng: &mut R) -> StickSequence {
        let seed: u64 = rng.gen();

        StickSequence::new(self.breaker.clone(), Some(seed))
    }
}

impl Sampleable<Sbd> for StickBreaking {
    fn draw<R: Rng>(&self, rng: &mut R) -> Sbd {
        Sbd::new(self.draw(rng))
    }
}

impl ConjugatePrior<usize, Sbd> for StickBreaking {
    type Posterior = StickBreaking;
    type LnMCache = ();
    type LnPpCache = ();

    fn ln_m_cache(&self) -> Self::LnMCache {}

    fn ln_pp_cache(&self, x: &DataOrSuffStat<usize, Sbd>) -> Self::LnPpCache {
        todo!()
//         let post = self.posterior(x);
//         // we'll need the alpha for computing 1 / (1 + alpha), which is the
//         // expected likelihood of a new class
//         let alpha = post.dir.alphas().last().unwrap();
//         // Need to norm the alphas to probabilities
//         let ln_norm = post.dir.alphas().iter().sum::<f64>().ln();
//         let ln_weights = post
//             .iter()
//             .map(|(&x, &ix)| (x, post.dir.alphas[ix].ln() - ln_norm))
//             .collect();

//         // // ln (1/(1 + alpha))
//         // let ln_f_new = (1.0 + alpha).recip().ln() - ln_norm;
//         let ln_f_new = (alpha / (1.0 + alpha)).ln() - ln_norm;

//         SbCache {
//             ln_weights,
//             ln_f_new,
//         }
    }

    fn posterior(&self, x: &DataOrSuffStat<usize, Sbd>) -> Self::Posterior {
        todo!()
//         match x {
//             DataOrSuffStat::Data(xs) => {
//                 let mut stat = SbdSuffStat::new();
//                 stat.observe_many(xs);
//                 sbpost_from_stat(self.alpha, &stat)
//             }
//             DataOrSuffStat::SuffStat(stat) => {
//                 sbpost_from_stat(self.alpha, stat)
//             }
//             DataOrSuffStat::None => panic!("Need data for posterior"),
//         }
    }

    fn ln_m_with_cache(
        &self,
        _cache: &Self::LnMCache,
        x: &DataOrSuffStat<usize, Sbd>,
    ) -> f64 {
        todo!()
//         match x {
//             DataOrSuffStat::Data(xs) => {
//                 let mut stat = SbdSuffStat::new();
//                 stat.observe_many(xs);
//                 sbm_from_stat(self.alpha, &stat)
//             }
//             DataOrSuffStat::SuffStat(stat) => sbm_from_stat(self.alpha, stat),
//             DataOrSuffStat::None => panic!("Need data for posterior"),
//         }
    }

    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &usize) -> f64 {
        todo!()
//         // FIXME: I feel like this isn't quite right
//         cache.ln_weights.get(y).copied().unwrap_or(cache.ln_f_new)
    }
}
