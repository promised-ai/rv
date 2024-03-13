use crate::experimental::Sbd;
use crate::experimental::SbdSuffStat;
use crate::experimental::StickBreakingSuffStat;
use crate::experimental::StickSequence;
use crate::prelude::Beta;
use crate::prelude::BetaBinomial;
use crate::prelude::DataOrSuffStat;
use crate::prelude::UnitPowerLaw;
use crate::suffstat_traits::*;
use crate::traits::*;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct StickBreaking {
    pub breaker: UnitPowerLaw,
    pub prefix: Vec<Beta>,
}

impl StickBreaking {
    pub fn new(breaker: UnitPowerLaw) -> Self {
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
    type LnPpCache = Self::Posterior;

    fn ln_m_cache(&self) -> Self::LnMCache {}

    fn ln_pp_cache(&self, x: &DataOrSuffStat<usize, Sbd>) -> Self::LnPpCache {
        self.posterior(x)
    }

    fn posterior_from_suffstat(&self, stat: &SbdSuffStat) -> Self::Posterior {
        let pairs = stat.break_pairs();
        let new_prefix = self
            .prefix
            .iter()
            .zip_longest(pairs)
            .map(|pair| match pair {
                Left(beta) => beta.clone(),
                Right((b, a)) => {
                    Beta::new(self.breaker.alpha() + a as f64, 1.0 + b as f64)
                        .unwrap()
                }
                Both(beta, (b, a)) => {
                    Beta::new(beta.alpha() + a as f64, beta.beta() + b as f64)
                        .unwrap()
                }
            })
            .collect();
        StickBreaking {
            breaker: self.breaker.clone(),
            prefix: new_prefix,
        }
    }

    fn ln_m(&self, x: &DataOrSuffStat<usize, Sbd>) -> f64 {
        self.m(x).ln()
    }

    fn m(&self, x: &DataOrSuffStat<usize, Sbd>) -> f64 {
        let count_pairs = match x {
            DataOrSuffStat::Data(xs) => {
                let mut stat = SbdSuffStat::new();
                stat.observe_many(xs);
                stat.break_pairs()
            }
            DataOrSuffStat::SuffStat(stat) => stat.break_pairs(),
        };
        let params = self
            .prefix
            .iter()
            .map(|b| (b.alpha(), b.beta()))
            .chain(std::iter::repeat((self.breaker.alpha(), 1.0)));
        count_pairs
            .iter()
            .zip(params)
            .map(|(counts, params)| {
                let n = counts.0 + counts.1;
                BetaBinomial::new(n as u32, params.0, params.1)
                    .unwrap()
                    .f(&(counts.1 as u32))
            })
            .product()
    }

    fn ln_m_with_cache(
        &self,
        _cache: &Self::LnMCache,
        x: &DataOrSuffStat<usize, Sbd>,
    ) -> f64 {
        self.ln_m(x)
    }

    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &usize) -> f64 {
        cache.ln_m(&DataOrSuffStat::Data(&[*y]))
    }
            
    fn pp(&self, y: &usize, x: &DataOrSuffStat<usize, Sbd>) -> f64 {
        let post = self.posterior(x);
        post.m(&DataOrSuffStat::Data(&[*y]))
    }
}
