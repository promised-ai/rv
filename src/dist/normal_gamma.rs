//! A common conjugate prior for Gaussians
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::consts::HALF_LN_2PI;
use crate::data::GaussianSuffStat;
use crate::dist::{Gamma, Gaussian};
use crate::impl_display;
use crate::traits::*;
use getset::Setters;
use rand::Rng;

/// Prior for Gaussian
///
/// Given `x ~ N(μ, σ)`, the Normal Gamma prior implies that `μ ~ N(m, 1/(rρ))`
/// and `ρ ~ Gamma(ν/2, s/2)`.
#[derive(Debug, Clone, PartialEq, PartialOrd, Setters)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NormalGamma {
    #[set = "pub"]
    m: f64,
    #[set = "pub"]
    r: f64,
    #[set = "pub"]
    s: f64,
    #[set = "pub"]
    v: f64,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum NormalGammaError {
    /// The m parameter is infinite or NaN
    MNotFiniteError,
    /// The r parameter is less than or equal to zero
    RTooLowError,
    /// The r parameter is infinite or NaN
    RNotFiniteError,
    /// The s parameter is less than or equal to zero
    STooLowError,
    /// The s parameter is infinite or NaN
    SNotFiniteError,
    /// The v parameter is less than or equal to zero
    VTooLowError,
    /// The v parameter is infinite or NaN
    VNotFiniteError,
}

impl NormalGamma {
    /// Create a new Normal Gamma distribution
    ///
    /// # Arguments
    /// - m: The prior mean
    /// - r: Relative precision of μ versus data
    /// - s: The mean of rho (the precision) is v/s.
    /// - v: Degrees of freedom of precision of rho
    pub fn new(
        m: f64,
        r: f64,
        s: f64,
        v: f64,
    ) -> Result<Self, NormalGammaError> {
        if !m.is_finite() {
            Err(NormalGammaError::MNotFiniteError)
        } else if !r.is_finite() {
            Err(NormalGammaError::RNotFiniteError)
        } else if !s.is_finite() {
            Err(NormalGammaError::SNotFiniteError)
        } else if !v.is_finite() {
            Err(NormalGammaError::VNotFiniteError)
        } else if r <= 0.0 {
            Err(NormalGammaError::RTooLowError)
        } else if s <= 0.0 {
            Err(NormalGammaError::STooLowError)
        } else if v <= 0.0 {
            Err(NormalGammaError::VTooLowError)
        } else {
            Ok(NormalGamma { m, r, s, v })
        }
    }

    /// Creates a new NormalGamma without checking whether the parameters are
    /// valid.
    pub fn new_unchecked(m: f64, r: f64, s: f64, v: f64) -> Self {
        NormalGamma { m, r, s, v }
    }

    /// Get the m parameter
    pub fn m(&self) -> f64 {
        self.m
    }

    /// Get the r parameter
    pub fn r(&self) -> f64 {
        self.r
    }

    /// Get the s parameter
    pub fn s(&self) -> f64 {
        self.s
    }

    /// Get the v parameter
    pub fn v(&self) -> f64 {
        self.v
    }
}

impl From<&NormalGamma> for String {
    fn from(ng: &NormalGamma) -> String {
        format!(
            "Normal-Gamma(m: {}, r: {}, s: {}, ν: {})",
            ng.m, ng.r, ng.s, ng.v
        )
    }
}

impl_display!(NormalGamma);

impl Rv<Gaussian> for NormalGamma {
    fn ln_f(&self, x: &Gaussian) -> f64 {
        let rho = x.sigma().powi(2).recip();
        let lnf_rho =
            Gamma::new(self.v / 2.0, self.s / 2.0).unwrap().ln_f(&rho);
        let prior_sigma = (self.r * rho).recip().sqrt();
        let lnf_mu = Gaussian::new(self.m, prior_sigma).unwrap().ln_f(&x.mu());
        lnf_rho + lnf_mu - HALF_LN_2PI
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Gaussian {
        // NOTE: The parameter errors in this fn shouldn't happen if the prior
        // parameters are valid.
        let rho: f64 = Gamma::new(self.v / 2.0, self.s / 2.0)
            .expect("Invalid σ posterior params")
            .draw(&mut rng);
        let post_sigma: f64 = (self.r * rho).recip().sqrt();
        let mu: f64 = Gaussian::new(self.m, post_sigma)
            .expect("Invalid μ posterior params")
            .draw(&mut rng);

        Gaussian::new(mu, rho.sqrt().recip()).expect("Invalid params")
    }
}

impl Support<Gaussian> for NormalGamma {
    fn supports(&self, x: &Gaussian) -> bool {
        // NOTE: Could replace this with Gaussian::new(mu, sigma).is_ok(),
        // but this is more explicit.
        x.mu().is_finite() && x.sigma() > 0.0 && x.sigma().is_finite()
    }
}

impl HasSuffStat<f64> for NormalGamma {
    type Stat = GaussianSuffStat;
    fn empty_suffstat(&self) -> Self::Stat {
        GaussianSuffStat::new()
    }
}

impl ContinuousDistr<Gaussian> for NormalGamma {}
