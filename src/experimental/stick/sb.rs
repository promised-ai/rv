use crate::data::DataOrSuffStat;
use crate::dist::Beta;
use crate::dist::BetaError;
use crate::dist::UnitPowerLawError;
use crate::experimental::stick::HalfBeta;
use crate::experimental::stick::StickBreakingDiscrete;
use crate::experimental::stick::StickBreakingDiscreteSuffStat;
use crate::experimental::stick::StickSequence;
use crate::traits::{
    ConjugatePrior, HasDensity, HasSuffStat, Sampleable, SuffStat,
};
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
    break_tail: HalfBeta,
}

#[derive(Debug, Clone)]
pub enum PrefixOrTail<'a> {
    Prefix(&'a Beta),
    Tail(&'a HalfBeta),
}

/// Implementation of the `StickBreaking` struct.
impl StickBreaking {
    /// Creates a new instance of `StickBreaking` with the given `breaker`.
    ///
    /// # Example
    /// ```
    /// use rv::experimental::stick::StickBreaking;
    /// use rv::experimental::stick::HalfBeta;
    ///
    /// let alpha = 5.0;
    /// let stick_breaking = StickBreaking::new(HalfBeta::new(alpha).unwrap());
    /// ```
    pub fn new(breaker: HalfBeta) -> Self {
        let break_prefix = Vec::new();
        Self {
            break_prefix,
            break_tail: breaker,
        }
    }

    pub fn new_with_prefix(
        break_prefix: Vec<Beta>,
        break_tail: HalfBeta,
    ) -> Self {
        Self {
            break_prefix,
            break_tail,
        }
    }

    pub fn from_alpha(alpha: f64) -> Result<Self, UnitPowerLawError> {
        let breaker = HalfBeta::new(alpha)?;
        Ok(Self::new(breaker))
    }

    /// Sets the alpha parameter for both the `break_tail` and all Beta
    /// distributions in `break_prefix`.
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

    pub fn break_tail(&self) -> &HalfBeta {
        &self.break_tail
    }

    pub fn break_dists(&self) -> impl Iterator<Item = PrefixOrTail<'_>> {
        self.break_prefix
            .iter()
            .map(PrefixOrTail::Prefix)
            .chain(std::iter::repeat(PrefixOrTail::Tail(&self.break_tail)))
    }

    pub fn alpha(&self) -> f64 {
        self.break_tail.alpha()
    }
}

pub struct StickWeights(pub Vec<f64>);
pub struct BreakSequence(pub Vec<f64>);

impl From<&BreakSequence> for StickWeights {
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
        StickWeights(ws)
    }
}

impl From<&StickWeights> for BreakSequence {
    fn from(ws: &StickWeights) -> Self {
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

impl HasDensity<StickWeights> for StickBreaking {
    fn ln_f(&self, w: &StickWeights) -> f64 {
        let mut mass = 1.0;
        w.0.iter()
            .zip(self.break_dists())
            .map(|(w_i, beta)| {
                let p = w_i / mass;
                mass -= w_i;
                let ln_f = match beta {
                    PrefixOrTail::Prefix(beta) => beta.ln_f(&p),
                    PrefixOrTail::Tail(unit_powlaw) => unit_powlaw.ln_f(&p),
                };
                ln_f
            })
            .sum()
    }
}

impl Sampleable<StickSequence> for StickBreaking {
    fn draw<R: Rng>(&self, rng: &mut R) -> StickSequence {
        let seed: u64 = rng.random();

        let seq = StickSequence::new(self.break_tail.clone(), Some(seed));
        // In the event that this is a posterior distribution with a
        // break_prefix, we need to construct the initial weights from the break
        // prefix.
        for beta in &self.break_prefix {
            let p: f64 = beta.draw(rng);
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

impl Sampleable<StickBreakingDiscrete> for StickBreaking {
    fn draw<R: Rng>(&self, rng: &mut R) -> StickBreakingDiscrete {
        StickBreakingDiscrete::new(self.draw(rng))
    }
}

impl ConjugatePrior<usize, StickBreakingDiscrete> for StickBreaking {
    type Posterior = StickBreaking;
    type MCache = ();
    type PpCache = Self::Posterior;

    fn empty_stat(
        &self,
    ) -> <StickBreakingDiscrete as HasSuffStat<usize>>::Stat {
        StickBreakingDiscreteSuffStat::new()
    }

    fn ln_m_cache(&self) -> Self::MCache {}

    fn ln_pp_cache(
        &self,
        x: DataOrSuffStat<usize, StickBreakingDiscrete>,
    ) -> Self::PpCache {
        self.posterior(x)
    }

    fn posterior_from_suffstat(
        &self,
        stat: &StickBreakingDiscreteSuffStat,
    ) -> Self::Posterior {
        use itertools::EitherOrBoth::{Both, Left, Right};
        use itertools::Itertools;
        let pairs = stat.break_pairs();
        let new_prefix = self
            .break_prefix
            .iter()
            .zip_longest(pairs)
            .map(|pair| match pair {
                Left(beta) => beta.clone(),
                Right((t, f)) => Beta::new(
                    1.0 + f as f64,
                    self.break_tail.alpha() + t as f64,
                )
                .unwrap(),
                Both(beta, (t, f)) => {
                    Beta::new(beta.alpha() + f as f64, beta.beta() + t as f64)
                        .unwrap()
                }
            })
            .collect();

        StickBreaking {
            break_prefix: new_prefix,
            break_tail: self.break_tail.clone(),
        }
    }

    fn ln_m(&self, x: DataOrSuffStat<usize, StickBreakingDiscrete>) -> f64 {
        use itertools::EitherOrBoth::{Both, Left, Right};
        use itertools::Itertools;

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

                    // Γ(α)/Γ(α+n) Γ(c+1) α/(α+gk)
                    // TODO: Simplify this after everything is working
                    (num_pass + alpha).ln_beta(num_fail + 1.0)
                        - alpha.ln_beta(1.0)
                }
                Right((_a, _b)) => 0.0,
                Both((num_pass, num_fail), (a, b)) => {
                    rising_beta_prod(b, *num_pass, a, *num_fail).ln()
                }
            })
            .sum()
    }

    fn ln_m_with_cache(
        &self,
        _cache: &Self::MCache,
        x: DataOrSuffStat<usize, StickBreakingDiscrete>,
    ) -> f64 {
        self.ln_m(x)
    }

    /// Computes the logarithm of the predictive probability with cache.
    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &usize) -> f64 {
        cache.ln_m(DataOrSuffStat::Data(&[*y]))
    }

    /// Computes the predictive probability.
    fn pp(
        &self,
        y: &usize,
        x: DataOrSuffStat<usize, StickBreakingDiscrete>,
    ) -> f64 {
        let post = self.posterior(x);
        post.m(DataOrSuffStat::Data(&[*y]))
    }
}

#[cfg(test)]
mod tests {
    use crate::dist::Bernoulli;
    use crate::dist::ChiSquared;
    use crate::traits::Cdf;
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn partial_weights_to_break_sequence(v in prop::collection::vec(0.0..=1.0, 1..100), m in 0.0..=1.0) {
            // we want the sum of ws to be in the range [0, 1]
            let multiplier: f64 = m / v.iter().sum::<f64>();
            let ws = StickWeights(v.iter().map(|w| w * multiplier).collect());
            let bs = BreakSequence::from(&ws);
            assert::close(ws.0, StickWeights::from(&bs).0, 1e-10);
        }
    }

    proptest! {
        #[test]
        fn break_sequence_to_partial_weights(v in prop::collection::vec(0.0..=1.0, 1..100)) {
            let bs = BreakSequence(v);
            let ws = StickWeights::from(&bs);
            let bs2 = BreakSequence::from(&ws);
            assert::close(bs.0, bs2.0, 1e-10);
        }
    }

    #[test]
    fn sb_ln_m_vs_monte_carlo() {
        use crate::misc::func::LogSumExp;

        let n_samples = 1_000_000;
        let xs: Vec<usize> = vec![1, 2, 3];

        let sb = StickBreaking::new(HalfBeta::new(5.0).unwrap());
        let obs = DataOrSuffStat::Data(&xs);
        let ln_m = sb.ln_m(obs);

        let mc_est = {
            sb.sample_stream(&mut rand::rng())
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
        let sb = StickBreaking::new(HalfBeta::new(5.0).unwrap());
        let sb_pp = sb.pp(&3, DataOrSuffStat::Data(&[1, 2]));

        let post = sb.posterior(DataOrSuffStat::Data(&[1, 2]));
        let post_f = post.pp(
            &3,
            DataOrSuffStat::SuffStat(&StickBreakingDiscreteSuffStat::new()),
        );
        assert::close(sb_pp, post_f, 1e-10);
    }

    #[test]
    fn sb_repeated_obs_more_likely() {
        let sb = StickBreaking::new(HalfBeta::new(5.0).unwrap());
        let sb_m = sb.ln_m(DataOrSuffStat::Data(&[10]));
        let post = sb.posterior(DataOrSuffStat::Data(&[10]));
        let post_m = post.ln_m(DataOrSuffStat::Data(&[10]));
        assert!(post_m > sb_m);
    }

    #[test]
    fn sb_bayes_law() {
        let mut rng = rand::rng();
        let alpha = 5.0;

        // Prior
        let prior = StickBreaking::from_alpha(alpha).unwrap();
        let par: StickSequence = prior.draw(&mut rng);
        let weights = par.weights(Some(7));
        // p(weights)
        let log_prior = prior.ln_f(&weights);

        // Likelihood p(x|weights)
        let lik = StickBreakingDiscrete::new(par);
        let x = 5_usize;
        let log_like = lik.ln_f(&x);

        // Evidence p(x)
        let log_marg = prior.ln_m(DataOrSuffStat::Data(&[x]));

        // Posterior p(weights | x)
        let post = prior.posterior(DataOrSuffStat::Data(&[x]));
        let log_post = post.ln_f(&weights);

        // Bayes' law
        assert::close(log_post, log_like + log_prior - log_marg, 1e-12);
    }

    #[test]
    fn sb_pp_is_quotient_of_marginals() {
        // pp(x|y) = m({x, y})/m(x)
        let sb = StickBreaking::new(HalfBeta::new(5.0).unwrap());
        let sb_pp = sb.pp(&1, DataOrSuffStat::Data(&[0]));

        let m_1 = sb.m(DataOrSuffStat::Data(&[0]));
        let m_1_2 = sb.m(DataOrSuffStat::Data(&[0, 1]));

        assert::close(sb_pp, m_1_2 / m_1, 1e-12);
    }

    #[test]
    fn sb_big_alpha_heavy_tails() {
        let sb_5 = StickBreaking::new(HalfBeta::new(5.0).unwrap());
        let sb_2 = StickBreaking::new(HalfBeta::new(2.0).unwrap());
        let sb_pt5 = StickBreaking::new(HalfBeta::new(0.5).unwrap());

        let m_pt5_10 = sb_pt5.m(DataOrSuffStat::Data(&[10]));
        let m_2_10 = sb_2.m(DataOrSuffStat::Data(&[10]));
        let m_5_10 = sb_5.m(DataOrSuffStat::Data(&[10]));

        assert!(m_pt5_10 < m_2_10);
        assert!(m_2_10 < m_5_10);
    }

    #[test]
    fn sb_marginal_zero() {
        let sb = StickBreaking::new(HalfBeta::new(3.0).unwrap());
        let m_0 = sb.m(DataOrSuffStat::Data(&[0]));
        let bern = Bernoulli::new(3.0 / 4.0).unwrap();
        assert::close(m_0, bern.f(&0), 1e-12);
    }

    #[test]
    fn sb_postpred_zero() {
        // Γ(α)/Γ(α+1) * Γ(2) * α/(α+1)
        // Γ(3)/Γ(4) * Γ(2) * 3/4
        use DataOrSuffStat::Data;
        let sb = StickBreaking::new(HalfBeta::new(3.0).unwrap());
        let post = sb.posterior(Data(&[0]));
        let pp = post.m(Data(&[0]));
        eprintln!("{post:?}: {pp}");
        let pp_0 = sb.pp(&0, Data(&[0]));
        let bern = Bernoulli::new(3.0 / 5.0).unwrap();
        assert::close(pp_0, bern.f(&0), 1e-12);
    }

    #[test]
    fn sb_pp_zero_marginals() {
        // pp(x|y) = m({x, y})/m(x)
        let sb = StickBreaking::new(HalfBeta::new(5.0).unwrap());
        let sb_pp = sb.pp(&0, DataOrSuffStat::Data(&[0]));

        let m_1 = sb.m(DataOrSuffStat::Data(&[0]));
        let m_1_2 = sb.m(DataOrSuffStat::Data(&[0, 0]));

        assert::close(sb_pp, m_1_2 / m_1, 1e-12);
    }

    #[test]
    fn sb_posterior_obs_one() {
        let sb = StickBreaking::new(HalfBeta::new(3.0).unwrap());
        let post = sb.posterior(DataOrSuffStat::Data(&[2]));

        assert_eq!(post.break_prefix[0], Beta::new(1.0, 4.0).unwrap());
        assert_eq!(post.break_prefix[1], Beta::new(1.0, 4.0).unwrap());
        assert_eq!(post.break_prefix[2], Beta::new(2.0, 3.0).unwrap());
    }

    #[test]
    fn sb_logposterior_diff() {
        // Like Bayes Law, but takes a quotient to cancel evidence

        let mut rng = rand::rng();
        let sb = StickBreaking::new(HalfBeta::new(3.0).unwrap());
        let seq1: StickSequence = sb.draw(&mut rng);
        let seq2: StickSequence = sb.draw(&mut rng);

        let w1 = seq1.weights(Some(3));
        let w2 = seq2.weights(Some(3));

        let logprior_diff = sb.ln_f(&w1) - sb.ln_f(&w2);

        let data = [1, 2];
        let stat = StickBreakingDiscreteSuffStat::from(&data[..]);
        let post = sb.posterior(DataOrSuffStat::SuffStat(&stat));
        let logpost_diff = post.ln_f(&w1) - post.ln_f(&w2);

        let sbd1 = StickBreakingDiscrete::new(seq1);
        let sbd2 = StickBreakingDiscrete::new(seq2);
        let loglik_diff = sbd1.ln_f_stat(&stat) - sbd2.ln_f_stat(&stat);

        assert::close(logpost_diff, loglik_diff + logprior_diff, 1e-12);
    }

    #[test]
    fn sb_posterior_rejection_sampling() {
        let mut rng = rand::rng();
        let sb = StickBreaking::new(HalfBeta::new(3.0).unwrap());

        let num_samples = 1000;

        // Our computed posterior
        let data = [10];
        let post = sb.posterior(DataOrSuffStat::Data(&data[..]));

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
            .map(|j| {
                post.m(DataOrSuffStat::Data(&[j])) * f64::from(num_samples)
            })
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
        let mut sb = StickBreaking::new(HalfBeta::new(3.0).unwrap());

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
        assert_eq!(sb.break_tail, HalfBeta::new(2.0).unwrap());
    }
}
