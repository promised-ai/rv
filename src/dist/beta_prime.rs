use crate::impl_display;
use crate::traits::*;
use rand::Rng;
use special::Beta as _;
use std::f64;
use std::fmt;
use std::sync::OnceLock;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// [Beta Prime distribution](https://en.wikipedia.org/wiki/Beta_prime_distribution),
/// BetaPrime(α, β) over x in (0, ∞).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct BetaPrime {
    alpha: f64,
    beta: f64,
    #[cfg_attr(feature = "serde1", serde(skip))]
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

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum BetaPrimeError {
    /// The alpha parameter is less than or equal to zero
    AlphaTooLow { alpha: f64 },
    /// The alpha parameter is infinite or NaN
    AlphaNotFinite { alpha: f64 },
    /// The beta parameter is less than or equal to zero
    BetaTooLow { beta: f64 },
    /// The beta parameter is infinite or NaN
    BetaNotFinite { beta: f64 },
}

impl BetaPrime {
    /// Create a new BetaPrime distribution
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

    #[inline]
    pub(crate) fn new_unchecked(alpha: f64, beta: f64) -> Self {
        BetaPrime {
            alpha,
            beta,
            ln_beta_ab: OnceLock::new(),
        }
    }

    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    #[inline]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    #[inline]
    fn ln_beta_ab(&self) -> f64 {
        *self.ln_beta_ab.get_or_init(|| self.alpha.ln_beta(self.beta))
    }
}

impl From<&BetaPrime> for String {
    fn from(bp: &BetaPrime) -> String {
        format!("BetaPrime(α: {}, β: {})", bp.alpha, bp.beta)
    }
}

impl_display!(BetaPrime);

impl Support<f64> for BetaPrime {
    #[inline]
    fn supports(&self, x: &f64) -> bool {
        x > &0.0
    }
}

impl HasDensity<f64> for BetaPrime {
    fn ln_f(&self, x: &f64) -> f64 {
        if *x <= 0.0 {
            f64::NEG_INFINITY
        } else {
            let alpha = self.alpha;
            let beta = self.beta;
            (alpha - 1.0) * x.ln() - (alpha + beta) * (1.0 + x).ln() - self.ln_beta_ab()
        }
    }
}

impl Sampleable<f64> for BetaPrime {
    fn draw<R: Rng>(&self, rng: &mut R) -> f64 {
        let gamma_alpha = rand_distr::Gamma::new(self.alpha, 1.0).unwrap();
        let gamma_beta = rand_distr::Gamma::new(self.beta, 1.0).unwrap();
        
        let x: f64 = rng.sample(gamma_alpha);
        let y: f64 = rng.sample(gamma_beta);
        x / y
    }
}

impl std::error::Error for BetaPrimeError {}

impl fmt::Display for BetaPrimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlphaTooLow { alpha } => {
                write!(f, "alpha ({}) must be greater than zero", alpha)
            }
            Self::AlphaNotFinite { alpha } => {
                write!(f, "alpha ({}) was non-finite", alpha)
            }
            Self::BetaTooLow { beta } => {
                write!(f, "beta ({}) must be greater than zero", beta)
            }
            Self::BetaNotFinite { beta } => {
                write!(f, "beta ({}) was non-finite", beta)
            }
        }
    }
}

impl PartialEq for BetaPrime {
    fn eq(&self, other: &Self) -> bool {
        self.alpha == other.alpha && self.beta == other.beta
    }
}


use crate::traits::ConjugatePrior;
#[cfg(feature = "experimental")]
use crate::experimental::stick_breaking_process::{StickBreakingDiscrete, StickBreakingDiscreteSuffStat};
use crate::data::DataOrSuffStat;

#[cfg(feature = "experimental")]
impl ConjugatePrior<usize, StickBreakingDiscrete> for BetaPrime {
    type Posterior = Self;
    type MCache = f64;
    type PpCache = f64;

    fn posterior(&self, data: &DataOrSuffStat<usize, StickBreakingDiscrete>) -> Self {
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

    fn ln_m_with_cache(&self, cache: &Self::MCache, data: &DataOrSuffStat<usize, StickBreakingDiscrete>) -> f64 {
        let post = self.posterior(data);
        post.ln_beta_ab() + cache
    }

    fn ln_pp_cache(&self, data: &DataOrSuffStat<usize, StickBreakingDiscrete>) -> Self::PpCache {
        let post = self.posterior(data);
        post.alpha / post.beta
    }

    fn ln_pp_with_cache(&self, cache: &Self::PpCache, _y: &usize) -> f64 {
        cache.ln()
    }
}

#[cfg(feature = "experimental")]
impl Sampleable<StickBreakingDiscrete> for BetaPrime {
    fn draw<R: Rng>(&self, rng: &mut R) -> StickBreakingDiscrete {
        // Convert the BetaPrime sample to a StickBreakingDiscrete
        let p = self.draw(rng);
        StickBreakingDiscrete::new(p)
    }
}

#[cfg(test)]
mod tests {
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
    fn test_posterior() {
        let prior = BetaPrime::new(1.0, 1.0).unwrap();
        let data = vec![1, 2, 1];
        let posterior = prior.posterior(&DataOrSuffStat::Data(&data));
        
        // After observing [1,2,1]:
        // - Two observations of value 1 contribute 0*2 to alpha and 2 to beta
        // - One observation of value 2 contributes 1*1 to alpha and 1 to beta
        assert_eq!(posterior, BetaPrime::new(2.0, 4.0).unwrap());
    }
} 
