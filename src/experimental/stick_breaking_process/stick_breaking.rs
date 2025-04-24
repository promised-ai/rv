use crate::experimental::stick_breaking_process::StickBreakingDiscrete;
use crate::experimental::stick_breaking_process::StickBreakingDiscreteSuffStat;
use crate::experimental::stick_breaking_process::StickSequence;
use crate::prelude::*;
use crate::traits::{ConjugatePrior, HasDensity, HasSuffStat, Sampleable, SuffStat};
use itertools::Either;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use rand::Rng;
use special::Beta as BetaFn;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Represents a stick-breaking process.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct StickBreaking {
    break_prefix: Vec<Beta>,
    break_tail: UnitPowerLaw,
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
    /// use rv::experimental::stick_breaking_process::StickBreaking;
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

    pub fn new_with_prefix(
        break_prefix: Vec<Beta>,
        break_tail: UnitPowerLaw,
    ) -> Self {
        Self {
            break_prefix,
            break_tail,
        }
    }

    pub fn from_alpha(alpha: f64) -> Result<Self, UnitPowerLawError> {
        let breaker = UnitPowerLaw::new(alpha)?;
        Ok(Self::new(breaker))
    }

    /// Sets the alpha parameter for both the `break_tail` and all Beta distributions in `break_prefix`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The new alpha value to set.
    ///
    /// # Returns
    ///
    /// A result indicating success or containing a `UnitPowerLawError` if setting alpha on `break_tail` fails,
    /// or a `BetaError` if setting alpha on any `Beta` distribution in `break_prefix` fails.
    pub fn set_alpha(&mut self, alpha: f64) -> Result<(), BetaError> {
        let old_alpha = self.alpha();
        self.break_tail.set_alpha(alpha).map_err(|e| match e {
            UnitPowerLawError::AlphaNotFinite { alpha } => {
                BetaError::AlphaNotFinite { alpha }
            }
            UnitPowerLawError::AlphaTooLow { alpha } => {
                BetaError::AlphaTooLow { alpha }
            }
        })?;
        let d_alpha = alpha - old_alpha;
        for b in &mut self.break_prefix {
            b.set_alpha(b.alpha() + d_alpha)?;
        }
        Ok(())
    }

    pub fn break_prefix(&self) -> &Vec<Beta> {
        &self.break_prefix
    }

    pub fn break_tail(&self) -> &UnitPowerLaw {
        &self.break_tail
    }

    pub fn break_dists(
        &self,
    ) -> impl Iterator<Item = Either<&Beta, &UnitPowerLaw>> {
        self.break_prefix
            .iter()
            .map(Either::Left)
            .chain(std::iter::repeat(Either::Right(&self.break_tail)))
    }

    pub fn alpha(&self) -> f64 {
        self.break_tail.alpha()
    }
}

pub struct PartialWeights(pub Vec<f64>);
pub struct BreakSequence(pub Vec<f64>);

impl From<&BreakSequence> for PartialWeights {
    fn from(bs: &BreakSequence) -> Self {
        let mut remaining = 1.0;
        let ws =
            bs.0.iter()
                .map(|b| {
                    debug_assert!((0.0..=1.0).contains(b));
                    let w = (1.0 - b) * remaining;
                    debug_assert!((0.0..=1.0).contains(&w));
                    remaining -= w;
                    debug_assert!((0.0..=1.0).contains(&remaining));
                    w
                })
                .collect();
        PartialWeights(ws)
    }
}

impl From<&PartialWeights> for BreakSequence {
    fn from(ws: &PartialWeights) -> Self {
        let mut r_new = 1.0;
        let mut r_old = 1.0;
        let mut b = f64::NAN;
        let bs: Vec<f64> =
            ws.0.iter()
                .map(|w| {
                    debug_assert!((0.0..=1.0).contains(w));
                    r_new = r_old - w;
                    debug_assert!((0.0..=1.0).contains(&r_new));
                    b = r_new / r_old;
                    debug_assert!((0.0..=1.0).contains(&b));
                    r_old = r_new;
                    b
                })
                .collect();
        assert!(
            (0.0..=1.0).contains(bs.last().unwrap()),
            "Weights cannot sum to more than one."
        );
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
        self.break_dists()
            .zip(BreakSequence::from(w).0.iter())
            .map(|(b, p)| match b {
                Either::Left(beta) => beta.ln_f(p),
                Either::Right(unit_powlaw) => unit_powlaw.ln_f(p),
            })
            .sum()
    }
}

impl Sampleable<StickSequence> for StickBreaking {
    /// Draws a sample from the `StickBreaking` distribution.
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

        let seq = StickSequence::new(self.break_tail.clone(), Some(seed));
        for beta in &self.break_prefix {
            let p = beta.draw(rng);
            seq.push_break(p);
        }
        seq
    }
}

fn rising_beta_prod(x: f64, a: usize, y: f64, b: usize) -> f64 {
    let x_y = x + y;
    let mut r = 1.0;
    for k in 0..a {
        let k = k as f64;
        r *= x + k;
        r /= x_y + k;
    }
    let x_y_a = x_y + a as f64;
    for k in 0..b {
        let k = k as f64;
        r *= y + k;
        r /= x_y_a + k;
    }
    r
}

/// Implements the `Sampleable` trait for `StickBreaking`.
impl Sampleable<StickBreakingDiscrete> for StickBreaking {
    /// Draws a sample from the `StickBreaking` distribution.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to the random number generator.
    ///
    /// # Returns
    ///
    /// A sample from the `StickBreaking` distribution.
    fn draw<R: Rng>(&self, rng: &mut R) -> StickBreakingDiscrete {
        StickBreakingDiscrete::new(self.draw(rng))
    }
}

/// Implementation of the `ConjugatePrior` trait for the `StickBreaking` struct.
impl ConjugatePrior<usize, StickBreakingDiscrete> for StickBreaking {
    type Posterior = StickBreaking;
    type MCache = ();
    type PpCache = Self::Posterior;

    fn empty_stat(
        &self,
    ) -> <StickBreakingDiscrete as HasSuffStat<usize>>::Stat {
        StickBreakingDiscreteSuffStat::new()
    }

    /// Computes the logarithm of the marginal likelihood cache.
    fn ln_m_cache(&self) -> Self::MCache {}

    /// Computes the logarithm of the predictive probability cache.
    fn ln_pp_cache(
        &self,
        x: &DataOrSuffStat<usize, StickBreakingDiscrete>,
    ) -> Self::PpCache {
        self.posterior(x)
    }

    /// Computes the posterior distribution from the sufficient statistic.
    fn posterior_from_suffstat(
        &self,
        stat: &StickBreakingDiscreteSuffStat,
    ) -> Self::Posterior {
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

    /// Computes the logarithm of the marginal likelihood.
    fn ln_m(&self, x: &DataOrSuffStat<usize, StickBreakingDiscrete>) -> f64 {
        let count_pairs = match x {
            DataOrSuffStat::Data(xs) => {
                let mut stat = StickBreakingDiscreteSuffStat::new();
                stat.observe_many(xs);
                stat.break_pairs()
            }
            DataOrSuffStat::SuffStat(stat) => stat.break_pairs(),
        };
        let alpha = self.break_tail.alpha();
        let params = self.break_prefix.iter().map(|b| (b.alpha(), b.beta()));
        count_pairs
            .iter()
            .zip_longest(params)
            .map(|pair| match pair {
                Left((num_pass, num_fail)) => {
                    let (num_pass, num_fail) =
                        (*num_pass as f64, *num_fail as f64);

                    // TODO: Simplify this after everything is working
                    (num_pass + alpha).ln_beta(num_fail + 1.0)
                        - alpha.ln_beta(1.0)
                    // num_pass * alpha.ln() - (num_pass + num_fail) * (alpha + 1.0).ln()
                }
                Right((_a, _b)) => 0.0,
                Both((num_pass, num_fail), (a, b)) => {
                    // let (num_pass, num_fail) = (*num_pass as f64, *num_fail as f64);
                    // num_pass * a.ln() + num_fail * b.ln() - (num_pass + num_fail) * (a + b).ln()
                    // (num_pass + a).ln_beta(num_fail + b) - a.ln_beta(b)

                    // rising_pow(a, *num_pass).ln() + rising_pow(b, *num_fail).ln()
                    // - rising_pow(a + b, num_pass + num_fail).ln()

                    rising_beta_prod(a, *num_pass, b, *num_fail).ln()
                }
            })
            .sum()
    }

    /// Computes the logarithm of the marginal likelihood with cache.
    fn ln_m_with_cache(
        &self,
        _cache: &Self::MCache,
        x: &DataOrSuffStat<usize, StickBreakingDiscrete>,
    ) -> f64 {
        self.ln_m(x)
    }

    /// Computes the logarithm of the predictive probability with cache.
    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &usize) -> f64 {
        cache.ln_m(&DataOrSuffStat::Data(&[*y]))
    }

    /// Computes the predictive probability.
    fn pp(
        &self,
        y: &usize,
        x: &DataOrSuffStat<usize, StickBreakingDiscrete>,
    ) -> f64 {
        let post = self.posterior(x);
        post.m(&DataOrSuffStat::Data(&[*y]))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn partial_weights_to_break_sequence(v in prop::collection::vec(0.0..=1.0, 1..100), m in 0.0..=1.0) {
            // we want the sum of ws to be in the range [0, 1]
            let multiplier: f64 = m / v.iter().sum::<f64>();
            let ws = PartialWeights(v.iter().map(|w| w * multiplier).collect());
            let bs = BreakSequence::from(&ws);
            assert::close(ws.0, PartialWeights::from(&bs).0, 1e-10);
        }
    }

    proptest! {
        #[test]
        fn break_sequence_to_partial_weights(v in prop::collection::vec(0.0..=1.0, 1..100)) {
            let bs = BreakSequence(v);
            let ws = PartialWeights::from(&bs);
            let bs2 = BreakSequence::from(&ws);
            assert::close(bs.0, bs2.0, 1e-10);
        }
    }

    #[test]
    fn sb_ln_m_vs_monte_carlo() {
        use crate::misc::func::LogSumExp;

        let n_samples = 1_000_000;
        let xs: Vec<usize> = vec![1, 2, 3];

        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let obs = DataOrSuffStat::Data(&xs);
        let ln_m = sb.ln_m(&obs);

        let mc_est = {
            sb.sample_stream(&mut rand::thread_rng())
                .take(n_samples)
                .map(|sbd: StickBreakingDiscrete| {
                    xs.iter().map(|x| sbd.ln_f(x)).sum::<f64>()
                })
                .logsumexp()
                - (n_samples as f64).ln()
        };
        // high error tolerance. MC estimation is not the most accurate...
        assert::close(ln_m, mc_est, 1e-2);
    }

    #[test]
    fn sb_pp_posterior() {
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_pp = sb.pp(&3, &DataOrSuffStat::Data(&[1, 2]));
        let post = sb.posterior(&DataOrSuffStat::Data(&[1, 2]));
        let post_f = post.pp(
            &3,
            &DataOrSuffStat::SuffStat(&StickBreakingDiscreteSuffStat::new()),
        );
        assert::close(sb_pp, post_f, 1e-10);
    }

    #[test]
    fn sb_repeated_obs_more_likely() {
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_m = sb.ln_m(&DataOrSuffStat::Data(&[10]));
        let post = sb.posterior(&DataOrSuffStat::Data(&[10]));
        let post_m = post.ln_m(&DataOrSuffStat::Data(&[10]));
        assert!(post_m > sb_m);
    }

    #[test]
    fn sb_bayes_law() {
        let mut rng = rand::thread_rng();

        // Prior
        let prior = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let par: StickSequence = prior.draw(&mut rng);
        let par_data = par.weights(7);
        let prior_lnf = prior.ln_f(&par_data);

        // Likelihood
        let lik = StickBreakingDiscrete::new(par);
        let lik_data: &usize = &5;
        let lik_lnf = lik.ln_f(lik_data);

        // Evidence
        let ln_ev = prior.ln_m(&DataOrSuffStat::Data(&[*lik_data]));

        // Posterior
        let post = prior.posterior(&DataOrSuffStat::Data(&[*lik_data]));
        let post_lnf = post.ln_f(&par_data);

        // Bayes' law
        assert::close(post_lnf, prior_lnf + lik_lnf - ln_ev, 1e-12);
    }

    #[test]
    fn sb_pp_is_quotient_of_marginals() {
        // pp(x|y) = m({x, y})/m(x)
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_pp = sb.pp(&1, &DataOrSuffStat::Data(&[0]));

        let m_1 = sb.m(&DataOrSuffStat::Data(&[0]));
        let m_1_2 = sb.m(&DataOrSuffStat::Data(&[0, 1]));

        assert::close(sb_pp, m_1_2 / m_1, 1e-12);
    }

    #[test]
    fn sb_big_alpha_heavy_tails() {
        let sb_5 = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_2 = StickBreaking::new(UnitPowerLaw::new(2.0).unwrap());
        let sb_pt5 = StickBreaking::new(UnitPowerLaw::new(0.5).unwrap());

        let m_pt5_10 = sb_pt5.m(&DataOrSuffStat::Data(&[10]));
        let m_2_10 = sb_2.m(&DataOrSuffStat::Data(&[10]));
        let m_5_10 = sb_5.m(&DataOrSuffStat::Data(&[10]));

        assert!(m_pt5_10 < m_2_10);
        assert!(m_2_10 < m_5_10);
    }

    #[test]
    fn sb_marginal_zero() {
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let m_0 = sb.m(&DataOrSuffStat::Data(&[0]));
        let bern = Bernoulli::new(3.0 / 4.0).unwrap();
        assert::close(m_0, bern.f(&0), 1e-12);
    }

    #[test]
    fn sb_postpred_zero() {
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let pp_0 = sb.pp(&0, &DataOrSuffStat::Data(&[0]));
        let bern = Bernoulli::new(3.0 / 5.0).unwrap();
        assert::close(pp_0, bern.f(&0), 1e-12);
    }

    #[test]
    fn sb_pp_zero_marginals() {
        // pp(x|y) = m({x, y})/m(x)
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_pp = sb.pp(&0, &DataOrSuffStat::Data(&[0]));

        let m_1 = sb.m(&DataOrSuffStat::Data(&[0]));
        let m_1_2 = sb.m(&DataOrSuffStat::Data(&[0, 0]));

        assert::close(sb_pp, m_1_2 / m_1, 1e-12);
    }

    #[test]
    fn sb_posterior_obs_one() {
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let post = sb.posterior(&DataOrSuffStat::Data(&[2]));

        assert_eq!(post.break_prefix[0], Beta::new(4.0, 1.0).unwrap());
        assert_eq!(post.break_prefix[1], Beta::new(4.0, 1.0).unwrap());
        assert_eq!(post.break_prefix[2], Beta::new(3.0, 2.0).unwrap());
    }

    #[test]
    fn sb_logposterior_diff() {
        // Like Bayes Law, but takes a quotient to cancel evidence

        let mut rng = rand::thread_rng();
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let seq1: StickSequence = sb.draw(&mut rng);
        let seq2: StickSequence = sb.draw(&mut rng);

        let w1 = seq1.weights(3);
        let w2 = seq2.weights(3);

        let logprior_diff = sb.ln_f(&w1) - sb.ln_f(&w2);

        let data = [1, 2];
        let stat = StickBreakingDiscreteSuffStat::from(&data[..]);
        let post = sb.posterior(&DataOrSuffStat::SuffStat(&stat));
        let logpost_diff = post.ln_f(&w1) - post.ln_f(&w2);

        let sbd1 = StickBreakingDiscrete::new(seq1);
        let sbd2 = StickBreakingDiscrete::new(seq2);
        let loglik_diff = sbd1.ln_f_stat(&stat) - sbd2.ln_f_stat(&stat);

        assert::close(logpost_diff, loglik_diff + logprior_diff, 1e-12);
    }

    #[test]
    fn sb_posterior_rejection_sampling() {
        let mut rng = rand::thread_rng();
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());

        let num_samples = 1000;

        // Our computed posterior
        let data = [10];
        let post = sb.posterior(&DataOrSuffStat::Data(&data[..]));

        // An approximation using rejection sampling
        let mut stat = StickBreakingDiscreteSuffStat::new();
        let mut n = 0;
        while n < num_samples {
            let seq: StickSequence = sb.draw(&mut rng);
            let sbd = StickBreakingDiscrete::new(seq.clone());
            if sbd.draw(&mut rng) == 10 {
                stat.observe(&sbd.draw(&mut rng));
                n += 1;
            }
        }

        let counts = stat.counts();

        // This would be counts.len() - 1, but the current implementation has a
        // trailing zero we need to ignore
        let dof = (counts.len() - 2) as f64;

        // Chi-square test is not exact, so we'll trim to only consider cases
        // where expected count is at least 5.
        let expected_counts = (0..)
            .map(|j| post.m(&DataOrSuffStat::Data(&[j])) * f64::from(num_samples))
            .take_while(|x| *x > 5.0);

        let ts = counts
            .iter()
            .zip(expected_counts)
            .map(|(o, e)| ((*o as f64) - e).powi(2) / e);

        let t: &f64 = &ts.clone().sum();
        let p = ChiSquared::new(dof).unwrap().sf(t);

        assert!(p > 0.001, "p-value = {p}");
    }

    #[test]
    fn test_set_alpha() {
        // Step 1: Generate a new StickBreaking instance with alpha=3
        let mut sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());

        // Step 2: Set the prefix to [Beta(4, 3), Beta(3, 2), Beta(2, 1)]
        sb.break_prefix = vec![
            Beta::new(4.0, 2.0).unwrap(),
            Beta::new(3.0, 2.0).unwrap(),
            Beta::new(2.0, 2.0).unwrap(),
        ];

        // Step 3: Call set_alpha(2.0)
        sb.set_alpha(2.0).unwrap();

        // Step 4: Check that the prefix is now [Beta(3, 3), Beta(2, 2), Beta(1, 1)]
        assert_eq!(sb.break_prefix[0], Beta::new(3.0, 2.0).unwrap());
        assert_eq!(sb.break_prefix[1], Beta::new(2.0, 2.0).unwrap());
        assert_eq!(sb.break_prefix[2], Beta::new(1.0, 2.0).unwrap());
        assert_eq!(sb.break_tail, UnitPowerLaw::new(2.0).unwrap());
    }
} // mod tests
