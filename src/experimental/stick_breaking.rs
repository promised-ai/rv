use crate::experimental::Sbd;
use crate::experimental::SbdSuffStat;
use crate::experimental::StickBreakingSuffStat;
use crate::experimental::StickSequence;
use crate::prelude::*;
use crate::suffstat_traits::*;
use crate::traits::*;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use rand::Rng;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
/// Represents a stick-breaking process.
pub struct StickBreaking {
    pub break_prefix: Vec<Beta>,
    pub break_tail: UnitPowerLaw,
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
    /// let stick_breaking = StickBreaking::new(UnitPowerLaw::new(alpha).unwrap());
    /// ```
    pub fn new(breaker: UnitPowerLaw) -> Self {
        let break_prefix = Vec::new();
        Self {
            break_prefix,
            break_tail: breaker,
        }
    }
}

pub struct PartialWeights(pub Vec<f64>);
pub struct BreakSequence(pub Vec<f64>);

impl From<&BreakSequence> for PartialWeights {
    fn from(seq: &BreakSequence) -> Self {
        let mut total = 1.0;
        let ws = seq
            .0
            .iter()
            .map(|b| {
                let w = b * total;
                total *= b;
                w
            })
            .collect();
        PartialWeights(ws)
    }
}

impl From<&PartialWeights> for BreakSequence {
    fn from(ws: &PartialWeights) -> Self {
        let mut last_w = 1.0;
        let bs =
            ws.0.iter()
                .map(|w| {
                    let b = w / last_w;
                    last_w = *w;
                    b
                })
                .collect();
        BreakSequence(bs)
    }
}

/// Implements the `HasDensity` trait for `StickBreaking`.
impl HasDensity<PartialWeights> for StickBreaking {
    /// Calculates the natural logarithm of the density function for the given input `x`.
    ///
    /// # Arguments
    ///
    /// * `x` - A reference to a slice of `f64` values.
    ///
    /// # Returns
    ///
    /// The natural logarithm of the density function.
    fn ln_f(&self, w: &PartialWeights) -> f64 {
        // let pairs = x.break_pairs();
        // let new_prefix = self
        //     .prefix
        //     .iter()
        //     .zip_longest(pairs)
        //     .map(|pair| match pair {
        //         Left(beta) => beta.clone(),
        //         Right((a, b)) => {
        //             Beta::new(self.breaker.alpha() + a as f64, 1.0 + b as f64)
        //                 .unwrap()
        //         }
        //         Both(beta, (a, b)) => {
        //             Beta::new(beta.alpha() + a as f64, beta.beta() + b as f64)
        //                 .unwrap()
        //         }
        //     })

        todo!()
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

        // TODO: Account for the `break_prefix`
        StickSequence::new(self.break_tail.clone(), Some(seed))
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
            .break_prefix
            .iter()
            .zip_longest(pairs)
            .map(|pair| match pair {
                Left(beta) => beta.clone(),
                Right((a, b)) => Beta::new(
                    self.break_tail.alpha() + a as f64,
                    1.0 + b as f64,
                )
                .unwrap(),
                Both(beta, (a, b)) => {
                    Beta::new(beta.alpha() + a as f64, beta.beta() + b as f64)
                        .unwrap()
                }
            })
            .collect();
        StickBreaking {
            break_prefix: new_prefix,
            break_tail: self.break_tail.clone(),
        }
    }
    fn posterior(&self, x: &DataOrSuffStat<usize, Sbd>) -> Self::Posterior {
        match x {
            DataOrSuffStat::Data(xs) => {
                let mut stat = SbdSuffStat::new();
                stat.observe_many(xs);
                self.posterior_from_suffstat(&stat)
            }
            DataOrSuffStat::SuffStat(stat) => {
                self.posterior_from_suffstat(stat)
            }
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
            .break_prefix
            .iter()
            .map(|b| (b.alpha(), b.beta()))
            .chain(std::iter::repeat((self.break_tail.alpha(), 1.0)));
        count_pairs
            .iter()
            .zip(params)
            .map(|(counts, params)| {
                let n = counts.0 + counts.1;
                BetaBinomial::new(n as u32, params.0, params.1)
                    .unwrap()
                    .f(&(counts.0 as u32))
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

    #[test]
    fn partial_weights_to_break_sequence() {
        let ws = PartialWeights(vec![0.4, 0.3, 0.2]);
        let bs = BreakSequence::from(&ws);
        assert::close(ws.0, PartialWeights::from(&bs).0, 1e-10);
    }

    #[test]
    fn break_sequence_to_partial_weights() {
        let bs = BreakSequence(vec![0.4, 0.3, 0.2]);
        let ws = PartialWeights::from(&bs);
        assert::close(bs.0, BreakSequence::from(&ws).0, 1e-10);
    }

    #[test]
    fn sb_ln_m_vs_monte_carlo() {
        use crate::misc::logsumexp;

        let n_samples = 1000;
        let xs: Vec<usize> = vec![1, 2, 3];

        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let obs = DataOrSuffStat::Data(&xs);
        let ln_m = sb.ln_m(&obs);

        let mc_est = {
            let ln_fs: Vec<f64> = sb
                .sample_stream(&mut rand::thread_rng())
                .take(n_samples)
                .map(|sbd: Sbd| xs.iter().map(|x| sbd.ln_f(x)).sum::<f64>())
                .collect();
            logsumexp(&ln_fs) - (n_samples as f64).ln()
        };
        // high error tolerance. MC estimation is not the most accurate...
        assert::close(ln_m, mc_est, 1e-2);
    }

    #[test]
    fn sb_pp_posterior() {
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_pp = sb.pp(&3, &DataOrSuffStat::Data(&vec![1, 2]));
        let post = sb.posterior(&DataOrSuffStat::Data(&vec![1, 2]));
        let post_f =
            post.pp(&3, &DataOrSuffStat::SuffStat(&SbdSuffStat::new()));
        assert::close(sb_pp, post_f, 1e-10);
    }

    #[test]
    fn sb_repeated_obs_more_likely() {
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_m = sb.m(&DataOrSuffStat::Data(&vec![1, 2]));
        let post = sb.posterior(&DataOrSuffStat::Data(&vec![1, 2]));
        let post_m = post.m(&DataOrSuffStat::Data(&vec![1, 2]));
        assert!(post_m > sb_m);
    }

    // FIXME
    #[test]
    fn sb_bayes_law() {
        let mut rng = rand::thread_rng();

        // Prior
        let prior = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let par: StickSequence = prior.draw(&mut rng);
        let par_data = par.weights(3);
        let prior_f = prior.f(&par_data);

        // Likelihood
        let lik = Sbd::new(par);
        let lik_data: &usize = &0;
        let lik_f = lik.f(lik_data);

        // Evidence
        let ev = prior.m(&DataOrSuffStat::Data(&[*lik_data]));

        // Posterior
        let post = prior.posterior(&DataOrSuffStat::Data(&[*lik_data]));
        let post_f = post.f(&par_data);

        // Bayes' law
        assert::close(post_f, prior_f * lik_f / ev, 1e-12);
    }

    // FIXME
    #[test]
    fn sb_pp_is_quotient_of_marginals() {
        // pp(x|y) = m({x, y})/m(x)
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_pp = sb.pp(&1, &DataOrSuffStat::Data(&vec![0]));

        let m_1 = sb.m(&DataOrSuffStat::Data(&vec![0]));
        let m_1_2 = sb.m(&DataOrSuffStat::Data(&vec![0, 1]));

        assert::close(sb_pp, m_1_2 / m_1, 1e-12);
    }

    #[test]
    fn sb_big_alpha_heavy_tails() {
        let sb_5 = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_2 = StickBreaking::new(UnitPowerLaw::new(2.0).unwrap());
        let sb_pt5 = StickBreaking::new(UnitPowerLaw::new(0.5).unwrap());

        let m_pt5_10 = sb_pt5.m(&DataOrSuffStat::Data(&vec![10]));
        let m_2_10 = sb_2.m(&DataOrSuffStat::Data(&vec![10]));
        let m_5_10 = sb_5.m(&DataOrSuffStat::Data(&vec![10]));

        assert!(m_pt5_10 < m_2_10);
        assert!(m_2_10 < m_5_10);
    }

    #[test]
    fn sb_marginal_zero() {
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let m_0 = sb.m(&DataOrSuffStat::Data(&vec![0]));
        let bern = Bernoulli::new(3.0 / 4.0).unwrap();
        assert::close(m_0, bern.f(&0), 1e-12);
    }

    #[test]
    fn sb_postpred_zero() {
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let pp_0 = sb.pp(&0, &DataOrSuffStat::Data(&vec![0]));
        let bern = Bernoulli::new(3.0 / 5.0).unwrap();
        assert::close(pp_0, bern.f(&0), 1e-12);
    }

    #[test]
    fn sb_pp_zero_marginals() {
        // pp(x|y) = m({x, y})/m(x)
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_pp = sb.pp(&0, &DataOrSuffStat::Data(&vec![0]));

        let m_1 = sb.m(&DataOrSuffStat::Data(&vec![0]));
        let m_1_2 = sb.m(&DataOrSuffStat::Data(&vec![0, 0]));

        assert::close(sb_pp, m_1_2 / m_1, 1e-12);
    }

    #[test]
    fn sb_posterior_obs_one() {
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let post = sb.posterior(&DataOrSuffStat::Data(&vec![1]));

        assert_eq!(post.break_prefix[0], Beta::new(4.0, 1.0).unwrap());
        assert_eq!(post.break_prefix[1], Beta::new(3.0, 2.0).unwrap());
    }

    #[test]
    fn sb_logposterior_diff() {
        // Like Bayes Law, but takes a quotient to cancel evidence

        let mut rng = rand::thread_rng();
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let seq1: StickSequence = sb.draw(&mut rng);
        let seq2: StickSequence = sb.draw(&mut rng);

        let w1 = seq1.weights(2);
        let w2 = seq2.weights(2);

        let logprior_diff = sb.ln_f(&w1) - sb.ln_f(&w2);

        let data = vec![1, 2];
        let stat = SbdSuffStat::from(&data[..]);
        let post = sb.posterior(&DataOrSuffStat::SuffStat(&stat));
        let logpost_diff = post.ln_f(&w1) - post.ln_f(&w2);

        let sbd1 = Sbd::new(seq1);
        let sbd2 = Sbd::new(seq2);
        let loglik_diff = sbd1.ln_f_stat(&stat) - sbd2.ln_f_stat(&stat);

        assert::close(logpost_diff, loglik_diff + logprior_diff, 1e-12);
    }
} // mod tests
