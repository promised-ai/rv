//! Beta prime distribution over x in (0, ∞)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::traits::*;
use rand::Rng;
use special::Beta;
use std::f64;
use std::fmt;
use std::sync::OnceLock;

/// [Beta prime distribution](https://en.wikipedia.org/wiki/Beta_prime_distribution),
/// BetaPrime(α, β) over x in (0, ∞).
///
/// # Examples
///
/// ```
/// use rv::prelude::*;
///
/// // Create a beta prime distribution with parameters α=2, β=3
/// let betaprime = BetaPrime::new(2.0, 3.0).unwrap();
///
/// // Calculate some properties
/// let mean = betaprime.mean().unwrap();
/// let variance = betaprime.variance().unwrap();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct BetaPrime {
    alpha: f64,
    beta: f64,
    #[cfg_attr(feature = "serde1", serde(skip))]
    /// Cached ln(Beta(a, b))
    ln_beta_ab: OnceLock<f64>,
}

pub struct BetaPrimeParameters {
    pub alpha: f64,
    pub beta: f64,
}

impl Parameterized for BetaPrime {
    type Parameters = BetaPrimeParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            alpha: self.alpha(),
            beta: self.beta(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.alpha, params.beta)
    }
}

impl PartialEq for BetaPrime {
    fn eq(&self, other: &BetaPrime) -> bool {
        self.alpha == other.alpha && self.beta == other.beta
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum BetaPrimeError {
    /// The alpha parameter is less than or equal too zero
    AlphaTooLow { alpha: f64 },
    /// The alpha parameter is infinite or NaN
    AlphaNotFinite { alpha: f64 },
    /// The beta parameter is less than or equal to zero
    BetaTooLow { beta: f64 },
    /// The beta parameter is infinite or NaN
    BetaNotFinite { beta: f64 },
}

impl BetaPrime {
    /// Create a new `BetaPrime` distribution with parameters α and β.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::BetaPrime;
    /// let bp = BetaPrime::new(2.0, 3.0);
    /// assert!(bp.is_ok());
    ///
    /// // Invalid negative parameter
    /// let bp_nope = BetaPrime::new(-5.0, 1.0);
    /// assert!(bp_nope.is_err());
    /// ```
    pub fn new(alpha: f64, beta: f64) -> Result<Self, BetaPrimeError> {
        if alpha <= 0.0 {
            Err(BetaPrimeError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(BetaPrimeError::AlphaNotFinite { alpha })
        } else if beta <= 0.0 {
            Err(BetaPrimeError::BetaTooLow { beta })
        } else if !beta.is_finite() {
            Err(BetaPrimeError::BetaNotFinite { beta })
        } else {
            Ok(Self::new_unchecked(alpha, beta))
        }
    }

    /// Creates a new BetaPrime without checking whether the parameters are valid.
    #[inline]
    pub fn new_unchecked(alpha: f64, beta: f64) -> Self {
        BetaPrime {
            alpha,
            beta,
            ln_beta_ab: OnceLock::new(),
        }
    }

    /// Get the alpha parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::BetaPrime;
    /// let bp = BetaPrime::new(2.0, 3.0).unwrap();
    /// assert_eq!(bp.alpha(), 2.0);
    /// ```
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set the alpha parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::BetaPrime;
    /// let mut bp = BetaPrime::new(2.0, 3.0).unwrap();
    ///
    /// bp.set_alpha(2.5).unwrap();
    /// assert_eq!(bp.alpha(), 2.5);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::BetaPrime;
    /// # let mut bp = BetaPrime::new(2.0, 3.0).unwrap();
    /// assert!(bp.set_alpha(0.1).is_ok());
    /// assert!(bp.set_alpha(0.0).is_err());
    /// assert!(bp.set_alpha(-1.0).is_err());
    /// assert!(bp.set_alpha(f64::INFINITY).is_err());
    /// assert!(bp.set_alpha(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_alpha(&mut self, alpha: f64) -> Result<(), BetaPrimeError> {
        if alpha <= 0.0 {
            Err(BetaPrimeError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(BetaPrimeError::AlphaNotFinite { alpha })
        } else {
            self.set_alpha_unchecked(alpha);
            Ok(())
        }
    }

    /// Set alpha without input validation
    #[inline]
    pub fn set_alpha_unchecked(&mut self, alpha: f64) {
        self.alpha = alpha;
        self.ln_beta_ab = OnceLock::new();
    }

    /// Get the beta parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::BetaPrime;
    /// let bp = BetaPrime::new(2.0, 3.0).unwrap();
    /// assert_eq!(bp.beta(), 3.0);
    /// ```
    #[inline]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Set the beta parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::BetaPrime;
    /// let mut bp = BetaPrime::new(2.0, 3.0).unwrap();
    ///
    /// bp.set_beta(3.5).unwrap();
    /// assert_eq!(bp.beta(), 3.5);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::BetaPrime;
    /// # let mut bp = BetaPrime::new(2.0, 3.0).unwrap();
    /// assert!(bp.set_beta(0.1).is_ok());
    /// assert!(bp.set_beta(0.0).is_err());
    /// assert!(bp.set_beta(-1.0).is_err());
    /// assert!(bp.set_beta(f64::INFINITY).is_err());
    /// assert!(bp.set_beta(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_beta(&mut self, beta: f64) -> Result<(), BetaPrimeError> {
        if beta <= 0.0 {
            Err(BetaPrimeError::BetaTooLow { beta })
        } else if !beta.is_finite() {
            Err(BetaPrimeError::BetaNotFinite { beta })
        } else {
            self.set_beta_unchecked(beta);
            Ok(())
        }
    }

    /// Set beta without input validation
    #[inline]
    pub fn set_beta_unchecked(&mut self, beta: f64) {
        self.beta = beta;
        self.ln_beta_ab = OnceLock::new();
    }

    /// Evaluate or fetch cached ln(Beta(a, b))
    #[inline]
    fn ln_beta_ab(&self) -> f64 {
        *self
            .ln_beta_ab
            .get_or_init(|| self.alpha.ln_beta(self.beta))
    }
}

impl Default for BetaPrime {
    fn default() -> Self {
        BetaPrime::new(1.0, 1.0).unwrap()
    }
}

impl From<&BetaPrime> for String {
    fn from(bp: &BetaPrime) -> String {
        format!("BetaPrime(α: {}, β: {})", bp.alpha, bp.beta)
    }
}

impl_display!(BetaPrime);

impl HasDensity<f64> for BetaPrime {
    fn ln_f(&self, x: &f64) -> f64 {
        let alpha = self.alpha;
        let beta = self.beta;
        (alpha - 1.0).mul_add(x.ln(), -((alpha + beta) * x.ln_1p()))
            - self.ln_beta_ab()
    }
}

impl Sampleable<f64> for BetaPrime {
    fn draw<R: Rng>(&self, rng: &mut R) -> f64 {
        // Generate a beta random variable and transform it
        let b = rand_distr::Beta::new(self.alpha, self.beta).unwrap();
        let beta_sample = rng.sample(b);
        beta_sample / (1.0 - beta_sample)
    }

    fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<f64> {
        let b = rand_distr::Beta::new(self.alpha, self.beta).unwrap();
        (0..n)
            .map(|_| {
                let beta_sample = rng.sample(b);
                beta_sample / (1.0 - beta_sample)
            })
            .collect()
    }
}

use crate::data::DataOrSuffStat;
#[cfg(feature = "experimental")]
use crate::experimental::stick_breaking_process::{
    StickBreakingDiscrete, StickBreakingDiscreteSuffStat,
};
use crate::traits::ConjugatePrior;

#[cfg(feature = "experimental")]
use crate::experimental::stick_breaking_process::StickBreaking;

use crate::prelude::UnitPowerLaw;

#[cfg(feature = "experimental")]
impl Sampleable<StickBreakingDiscrete> for BetaPrime {
    fn draw<R: Rng>(&self, rng: &mut R) -> StickBreakingDiscrete {
        // Draw a random alpha from the BetaPrime distribution
        let alpha: f64 = self.draw(rng);

        // Use the alpha to construct and draw from a StickBreakingProcess
        let stick_breaking =
            StickBreaking::new(UnitPowerLaw::new(alpha).unwrap());
        stick_breaking.draw(rng)
    }
}

impl Support<f64> for BetaPrime {
    fn supports(&self, x: &f64) -> bool {
        // TODO: Should this also check x.isfinite()?
        *x > 0.0
    }
}

impl ContinuousDistr<f64> for BetaPrime {}

impl Cdf<f64> for BetaPrime {
    fn cdf(&self, x: &f64) -> f64 {
        let t = *x / (1.0 + *x);
        t.inc_beta(self.alpha, self.beta, self.ln_beta_ab())
    }
}

impl Mean<f64> for BetaPrime {
    fn mean(&self) -> Option<f64> {
        if self.beta > 1.0 {
            Some(self.alpha / (self.beta - 1.0))
        } else {
            None
        }
    }
}

impl Mode<f64> for BetaPrime {
    fn mode(&self) -> Option<f64> {
        if self.alpha >= 1.0 {
            Some((self.alpha - 1.0) / (self.beta + 1.0))
        } else {
            Some(0.0)
        }
    }
}

impl Variance<f64> for BetaPrime {
    fn variance(&self) -> Option<f64> {
        if self.beta > 2.0 {
            let beta_m1 = self.beta - 1.0;
            let numer = self.alpha * (self.alpha + beta_m1);
            let denom = (beta_m1 - 1.0) * beta_m1 * beta_m1;
            Some(numer / denom)
        } else {
            None
        }
    }
}

// TODO: This is correct for non-integers. But for integers, we need to take limits to avoid NaNs
// impl Entropy for BetaPrime {
//     fn entropy(&self) -> f64 {
//         let apb = self.alpha + self.beta;
//         let pi = std::f64::consts::PI;
//         let a_pi = self.alpha * pi;
//         let b_pi = self.beta * pi;
//         self.ln_beta_ab()
//             + (self.alpha - 1.0) * (self.beta.digamma() - self.alpha.digamma())
//             + apb
//                 * ((1.0 - apb).digamma() - (1.0 - self.beta).digamma()
//                     + pi * (a_pi.sin()
//                         / (b_pi.sin() * (a_pi + b_pi).sin())))
//     }
// }

impl std::error::Error for BetaPrimeError {}

impl fmt::Display for BetaPrimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlphaTooLow { alpha } => {
                write!(f, "alpha ({}) must be greater than zero", alpha)
            }
            Self::AlphaNotFinite { alpha } => {
                write!(f, "alpha ({}) was non finite", alpha)
            }
            Self::BetaTooLow { beta } => {
                write!(f, "beta ({}) must be greater than zero", beta)
            }
            Self::BetaNotFinite { beta } => {
                write!(f, "beta ({}) was non finite", beta)
            }
        }
    }
}

#[cfg(feature = "experimental")]
impl ConjugatePrior<usize, StickBreakingDiscrete> for BetaPrime {
    type Posterior = Self;
    type MCache = f64;
    type PpCache = f64;

    fn empty_stat(
        &self,
    ) -> <StickBreakingDiscrete as HasSuffStat<usize>>::Stat {
        StickBreakingDiscreteSuffStat::new()
    }

    fn posterior(
        &self,
        data: &DataOrSuffStat<usize, StickBreakingDiscrete>,
    ) -> Self {
        match data {
            DataOrSuffStat::Data(xs) => {
                let stat = StickBreakingDiscreteSuffStat::from(xs.as_ref());
                self.posterior(&DataOrSuffStat::SuffStat(&stat))
            }
            DataOrSuffStat::SuffStat(stat) => {
                let mut alpha = self.alpha;
                let mut beta = self.beta;

                for (j, count) in stat.counts().iter().enumerate() {
                    alpha += (j * count) as f64;
                    beta += *count as f64;
                }

                Self::new_unchecked(alpha, beta)
            }
        }
    }

    fn ln_m_cache(&self) -> Self::MCache {
        -self.ln_beta_ab()
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::MCache,
        data: &DataOrSuffStat<usize, StickBreakingDiscrete>,
    ) -> f64 {
        let post = self.posterior(data);
        post.ln_beta_ab() + cache
    }

    fn ln_pp_cache(
        &self,
        data: &DataOrSuffStat<usize, StickBreakingDiscrete>,
    ) -> Self::PpCache {
        let post = self.posterior(data);
        post.alpha / post.beta
    }

    fn ln_pp_with_cache(&self, cache: &Self::PpCache, _y: &usize) -> f64 {
        cache.ln()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    use super::*;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;

    test_basic_impls!(f64, BetaPrime, BetaPrime::new(1.0, 1.0).unwrap());

    #[test]
    fn new() {
        let bp = BetaPrime::new(1.2, 3.4).unwrap();
        assert::close(bp.alpha, 1.2, TOL);
        assert::close(bp.beta, 3.4, TOL);

        assert!(BetaPrime::new(0.0, 1.0).is_err());
        assert!(BetaPrime::new(-1.0, 1.0).is_err());
        assert!(BetaPrime::new(1.0, 0.0).is_err());
        assert!(BetaPrime::new(1.0, -1.0).is_err());
        assert!(BetaPrime::new(f64::INFINITY, 1.0).is_err());
        assert!(BetaPrime::new(1.0, f64::INFINITY).is_err());
        assert!(BetaPrime::new(f64::NAN, 1.0).is_err());
        assert!(BetaPrime::new(1.0, f64::NAN).is_err());
    }

    #[test]
    fn mean_when_beta_gt_one() {
        let mut rng = Xoshiro256Plus::seed_from_u64(123);
        let bp = BetaPrime::new(2.0, 3.0).unwrap();

        // Theoretical mean
        let mu_theory: f64 = bp.mean().unwrap();

        // Sample mean
        let n = 1000;
        let mut s = 0.0;
        for _ in 0..n {
            let x: f64 = bp.draw(&mut rng);
            s += x;
        }
        let mu_sample = s / n as f64;

        assert::close(mu_sample, mu_theory, 0.03);
    }

    #[test]
    fn mean_when_beta_leq_one() {
        let bp = BetaPrime::new(2.0, 1.0).unwrap();
        assert!(bp.mean().is_none());
    }

    #[test]
    fn variance_when_beta_gt_two() {
        let mut rng = Xoshiro256Plus::seed_from_u64(123);
        let bp = BetaPrime::new(10.0, 15.0).unwrap();

        // Theoretical variance
        let var_theory: f64 = bp.variance().unwrap();

        // Sample variance assuming correct mean
        let mean = bp.mean().unwrap();
        let n = 1000;
        let mut sse = 0.0;
        for _ in 0..n {
            let x: f64 = bp.draw(&mut rng);
            sse += (x - mean).powi(2);
        }

        // Calculate variance
        let var_sample: f64 = sse / n as f64;

        assert::close(var_sample, var_theory, 0.01);
    }

    #[test]
    fn variance_when_beta_leq_two() {
        let bp = BetaPrime::new(2.0, 2.0).unwrap();
        assert!(bp.variance().is_none());
    }

    #[test]
    fn mode_when_alpha_geq_one() {
        let bp = BetaPrime::new(2.0, 3.0).unwrap();
        assert::close(bp.mode().unwrap(), 0.25, TOL);
    }

    #[test]
    fn mode_when_alpha_lt_one() {
        let bp = BetaPrime::new(0.5, 3.0).unwrap();
        assert!(bp.mode().unwrap() == 0.0);
    }

    #[test]
    fn draw_should_return_positive_values() {
        let mut rng = rand::thread_rng();
        let bp = BetaPrime::new(2.0, 3.0).unwrap();
        for _ in 0..100 {
            let x: f64 = bp.draw(&mut rng);
            assert!(x > 0.0);
        }
    }

    // TODO: Uncomment once limiting behavior is corrected
    // #[test]
    // fn entropy_finite() {
    //     let bp = BetaPrime::new(2.0, 3.0).unwrap();
    //     println!("{}", bp.entropy());
    //     assert!(bp.entropy().is_finite());
    // }

    #[test]
    fn cdf_values() {
        let bp = BetaPrime::new(2.0, 3.0).unwrap();
        let beta = crate::dist::Beta::new(2.0, 3.0).unwrap();

        // Take points in (0,1) and compare Beta CDF with BetaPrime CDF on transformed points
        let points = vec![0.1, 0.3, 0.5, 0.7, 0.9];

        for x in points {
            // For x in (0,1), transform to y = x/(1-x) to get corresponding BetaPrime point
            let y = x / (1.0 - x);

            // The CDFs should match at these corresponding points
            assert::close(beta.cdf(&x), bp.cdf(&y), 1e-12);
        }
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn test_ln_m_cache_consistency() {
        let prior = BetaPrime::new(2.0, 3.0).unwrap();
        let data = StickBreakingDiscreteSuffStat::new();
        let cache = prior.ln_m_cache();

        // The cache should be the negative of ln_beta_ab
        assert_eq!(cache, -prior.ln_beta_ab());

        // Using the cache with empty data should give same result
        let ln_m =
            prior.ln_m_with_cache(&cache, &DataOrSuffStat::SuffStat(&data));
        let post = prior.posterior(&DataOrSuffStat::SuffStat(&data));
        assert_eq!(ln_m, post.ln_beta_ab() - prior.ln_beta_ab());
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn test_posterior() {
        let prior = BetaPrime::new(2.0, 1.0).unwrap();
        let data = vec![0, 1, 1, 2, 2, 2]; // This gives counts [2, 2, 3]
        let posterior = prior.posterior(&DataOrSuffStat::Data(&data));
        // Our observation is [C₀, C₁, C₂] = [1, 2, 3]
        // So
        //   ∑ j Cⱼ = 0 * 1 + 1 * 2 + 2 * 3 = 8
        // and
        //   ∑ Cⱼ = 1 + 2 + 3 = 6
        //
        // So the posterior is BetaPrime(2 + 8, 1 + 6) = BetaPrime(10, 7)
        //
        // See See https://github.com/cscherrer/stick-breaking for details
        assert_eq!(posterior, BetaPrime::new(10.0, 7.0).unwrap());
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn test_posterior_parameter_updates() {
        let prior = BetaPrime::new(2.0, 3.0).unwrap();
        let mut stat = StickBreakingDiscreteSuffStat::new();

        // Add some observations
        let data = vec![0, 1, 1, 2, 2, 2]; // Counts: [1, 2, 3]
        for x in &data {
            stat.observe(x);
        }

        let posterior = prior.posterior(&DataOrSuffStat::SuffStat(&stat));

        // Alpha should increase by sum(j * count[j])
        // Beta should increase by sum(count[j])
        assert_eq!(posterior.alpha(), prior.alpha() + 8.0); // 0*1 + 1*2 + 2*3 = 8
        assert_eq!(posterior.beta(), prior.beta() + 6.0); // 1 + 2 + 3 = 6
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn test_pp_cache_consistency() {
        let prior = BetaPrime::new(2.0, 3.0).unwrap();
        let data = StickBreakingDiscreteSuffStat::new();

        let cache = prior.ln_pp_cache(&DataOrSuffStat::SuffStat(&data));
        let ln_pp = prior.ln_pp_with_cache(&cache, &0);

        // Cache should be posterior alpha/beta ratio
        let post = prior.posterior(&DataOrSuffStat::SuffStat(&data));
        assert_eq!(cache, post.alpha() / post.beta());

        // ln_pp should be log of the cache
        assert_eq!(ln_pp, cache.ln());
    }
}
