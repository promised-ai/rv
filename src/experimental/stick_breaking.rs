use crate::experimental::Sbd;
use crate::experimental::SbdSuffStat;
use crate::experimental::StickBreakingSuffStat;
use crate::experimental::StickSequence;
use crate::prelude::Beta;
use crate::prelude::BetaBinomial;
use crate::prelude::UnitPowerLaw;
use crate::suffstat_traits::*;
use crate::traits::*;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
/// Represents a stick-breaking process.
pub struct StickBreaking {
    pub breaker: UnitPowerLaw,
    pub prefix: Vec<Beta>,
}

/// Implementation of the `StickBreaking` struct.
impl StickBreaking {
    /// Creates a new instance of `StickBreaking` with the given `breaker`.
    ///
    /// # Arguments
    ///
    /// * `breaker` - The `UnitPowerLaw` used for stick breaking.
    ///
    /// # Returns
    ///
    /// A new instance of `StickBreaking`.
    ///
    /// # Example
    /// ```
    /// use rv::prelude::*;
    /// use rv::experimental::StickBreaking;
    ///
    /// let alpha = 5.0;
    /// let breaker = UnitPowerLaw::new(alpha).unwrap();
    /// let stick_breaking = StickBreaking::new(breaker);
    /// ```
    pub fn new(breaker: UnitPowerLaw) -> Self {
        let prefix = Vec::new();
        Self { breaker, prefix }
    }
}

/// Implements the `HasDensity` trait for `StickBreaking`.
impl HasDensity<&[f64]> for StickBreaking {
    /// Calculates the natural logarithm of the density function for the given input `x`.
    ///
    /// # Arguments
    ///
    /// * `x` - A reference to a slice of `f64` values.
    ///
    /// # Returns
    ///
    /// The natural logarithm of the density function.
    fn ln_f(&self, x: &&[f64]) -> f64 {
        let stat = StickBreakingSuffStat::from(x);
        self.ln_f_stat(&stat)
    }
}

impl Sampleable<StickSequence> for StickBreaking {
    /// Draws a sample from the StickBreaking distribution.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to the random number generator.
    ///
    /// # Returns
    ///
    /// A `StickSequence` representing the drawn sample.
    fn draw<R: Rng>(&self, rng: &mut R) -> StickSequence {
        let seed: u64 = rng.gen();

        StickSequence::new(self.breaker.clone(), Some(seed))
    }
}

/// Implements the `Sampleable` trait for `StickBreaking`.
impl Sampleable<Sbd> for StickBreaking {
    /// Draws a sample from the `StickBreaking` distribution.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to the random number generator.
    ///
    /// # Returns
    ///
    /// A sample from the `StickBreaking` distribution.
    fn draw<R: Rng>(&self, rng: &mut R) -> Sbd {
        Sbd::new(self.draw(rng))
    }
}

/// Implementation of the `ConjugatePrior` trait for the `StickBreaking` struct.
impl ConjugatePrior<usize, Sbd> for StickBreaking {
    type Posterior = StickBreaking;
    type MCache = ();
    type PpCache = Self::Posterior;

    /// Computes the logarithm of the marginal likelihood cache.
    fn ln_m_cache(&self) -> Self::MCache {}

    /// Computes the logarithm of the predictive probability cache.
    fn ln_pp_cache(&self, x: &DataOrSuffStat<usize, Sbd>) -> Self::PpCache {
        self.posterior(x)
    }

    /// Computes the posterior distribution from the sufficient statistic.
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

    /// Computes the logarithm of the marginal likelihood.
    fn ln_m(&self, x: &DataOrSuffStat<usize, Sbd>) -> f64 {
        self.m(x).ln()
    }

    /// Computes the marginal likelihood.
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

    /// Computes the logarithm of the marginal likelihood with cache.
    fn ln_m_with_cache(
        &self,
        _cache: &Self::MCache,
        x: &DataOrSuffStat<usize, Sbd>,
    ) -> f64 {
        self.ln_m(x)
    }

    /// Computes the logarithm of the predictive probability with cache.
    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &usize) -> f64 {
        cache.ln_m(&DataOrSuffStat::Data(&[*y]))
    }

    /// Computes the predictive probability.
    fn pp(&self, y: &usize, x: &DataOrSuffStat<usize, Sbd>) -> f64 {
        let post = self.posterior(x);
        post.m(&DataOrSuffStat::Data(&[*y]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use rand::thread_rng;

    #[test]
    fn test_stick_breaking() {
        let alpha = 5.0;
        let breaker = UnitPowerLaw::new(alpha).unwrap();
        let stick_breaking = StickBreaking::new(breaker);
        let mut rng = thread_rng();
        let n = 10000;
        let xs = stick_breaking.sample(n, &mut rng);
        assert_eq!(xs.len(), n);
        assert!(&0.0 < xs.first().unwrap());
        assert!(xs.last().unwrap() < &1.0);
        assert!(xs.windows(2).all(|w| w[0] <= w[1]));
        let mean = xs.iter().sum::<f64>() / n as f64;
        assert!(mean > 0.49 && mean < 0.51);
        let var = xs.iter().map(|x| (x - 0.5).powi(2)).sum::<f64>() / n as f64;
        assert!(var > 0.08 && var < 0.09);
    }
}
