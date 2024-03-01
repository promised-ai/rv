use crate::experimental::Sbd;
use crate::experimental::SbdSuffStat;
use crate::experimental::StickBreakingBetaSuffStat;
use crate::experimental::StickBreakingUnitPowerLawSuffStat;
use crate::experimental::StickSequence;
use crate::prelude::Beta;
use crate::prelude::DataOrSuffStat;
use crate::prelude::UnitPowerLaw;
use crate::suffstat_traits::HasSuffStat;
use crate::suffstat_traits::SuffStat;
use crate::traits::*;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct StickBreaking<B: Rv<f64> + Clone> {
    pub breaker: B,
    pub breaks: Vec<f64>,
}

impl<B: Rv<f64> + Clone> StickBreaking<B> {
    pub fn new(breaker: B) -> Self {
        let breaks = Vec::new();
        Self { breaker, breaks }
    }
}

impl HasDensity<&[f64]> for StickBreaking<Beta> {
    fn ln_f(&self, x: &&[f64]) -> f64 {
        let stat = StickBreakingBetaSuffStat::from(x);
        self.ln_f_stat(&stat)
    }
}

impl HasDensity<&[f64]> for StickBreaking<UnitPowerLaw> {
    fn ln_f(&self, x: &&[f64]) -> f64 {
        let stat = StickBreakingUnitPowerLawSuffStat::from(x);
        self.ln_f_stat(&stat)
    }
}

impl<B: Rv<f64> + Clone> Sampleable<StickSequence<B>> for StickBreaking<B> {
    fn draw<R: Rng>(&self, rng: &mut R) -> StickSequence<B> {
        let seed: u64 = rng.gen();

        StickSequence::new(self.breaker.clone(), Some(seed))
    }
}

// impl ConjugatePrior<usize, Sbd<Beta>> for StickBreaking<Beta> {
//     type Posterior = SbPosterior;
//     type LnMCache = ();
//     type LnPpCache = SbCache;

//     fn ln_m_cache(&self) -> Self::LnMCache {}

//     fn ln_pp_cache(&self, x: &DataOrSuffStat<usize, Sbd<Beta>>) -> Self::LnPpCache {
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
//     }

//     fn posterior(&self, x: &DataOrSuffStat<usize, Sbd<Beta>>) -> Self::Posterior {
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
//     }

//     fn ln_m_with_cache(
//         &self,
//         _cache: &Self::LnMCache,
//         x: &DataOrSuffStat<usize, Sbd<Beta>>,
//     ) -> f64 {
//         match x {
//             DataOrSuffStat::Data(xs) => {
//                 let mut stat = SbdSuffStat::new();
//                 stat.observe_many(xs);
//                 sbm_from_stat(self.alpha, &stat)
//             }
//             DataOrSuffStat::SuffStat(stat) => sbm_from_stat(self.alpha, stat),
//             DataOrSuffStat::None => panic!("Need data for posterior"),
//         }
//     }

//     fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &usize) -> f64 {
//         // FIXME: I feel like this isn't quite right
//         cache.ln_weights.get(y).copied().unwrap_or(cache.ln_f_new)
//     }
// }
