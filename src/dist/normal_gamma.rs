//! A common conjugate prior for Gaussians with unknown mean and variance
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::GaussianSuffStat;
use crate::dist::{Gamma, Gaussian};
use crate::impl_display;
use crate::traits::*;
use rand::Rng;
use std::fmt;

mod gaussian_prior;

/// Prior for Gaussian
///
/// Given `x ~ N(μ, σ)`, the Normal Gamma prior implies that `μ ~ N(m, 1/(rρ))`
/// and `ρ ~ Gamma(ν/2, s/2)`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NormalGamma {
    m: f64,
    r: f64,
    s: f64,
    v: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum NormalGammaError {
    /// The m parameter is infinite or NaN
    MNotFinite { m: f64 },
    /// The r parameter is less than or equal to zero
    RTooLow { r: f64 },
    /// The r parameter is infinite or NaN
    RNotFinite { r: f64 },
    /// The s parameter is less than or equal to zero
    STooLow { s: f64 },
    /// The s parameter is infinite or NaN
    SNotFinite { s: f64 },
    /// The v parameter is less than or equal to zero
    VTooLow { v: f64 },
    /// The v parameter is infinite or NaN
    VNotFinite { v: f64 },
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
            Err(NormalGammaError::MNotFinite { m })
        } else if !r.is_finite() {
            Err(NormalGammaError::RNotFinite { r })
        } else if !s.is_finite() {
            Err(NormalGammaError::SNotFinite { s })
        } else if !v.is_finite() {
            Err(NormalGammaError::VNotFinite { v })
        } else if r <= 0.0 {
            Err(NormalGammaError::RTooLow { r })
        } else if s <= 0.0 {
            Err(NormalGammaError::STooLow { s })
        } else if v <= 0.0 {
            Err(NormalGammaError::VTooLow { v })
        } else {
            Ok(NormalGamma { m, r, s, v })
        }
    }

    /// Creates a new NormalGamma without checking whether the parameters are
    /// valid.
    #[inline]
    pub fn new_unchecked(m: f64, r: f64, s: f64, v: f64) -> Self {
        NormalGamma { m, r, s, v }
    }

    /// Get the m parameter
    #[inline]
    pub fn m(&self) -> f64 {
        self.m
    }

    /// Set the value of m
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::NormalGamma;
    ///
    /// let mut ng = NormalGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(ng.m(), 0.0);
    ///
    /// ng.set_m(-1.1).unwrap();
    /// assert_eq!(ng.m(), -1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalGamma;
    /// # let mut ng = NormalGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(ng.set_m(-1.1).is_ok());
    /// assert!(ng.set_m(std::f64::INFINITY).is_err());
    /// assert!(ng.set_m(std::f64::NEG_INFINITY).is_err());
    /// assert!(ng.set_m(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_m(&mut self, m: f64) -> Result<(), NormalGammaError> {
        if m.is_finite() {
            self.set_m_unchecked(m);
            Ok(())
        } else {
            Err(NormalGammaError::MNotFinite { m })
        }
    }

    /// Set the value of m without input validation
    #[inline]
    pub fn set_m_unchecked(&mut self, m: f64) {
        self.m = m;
    }

    /// Get the r parameter
    #[inline]
    pub fn r(&self) -> f64 {
        self.r
    }

    /// Set the value of r
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::NormalGamma;
    ///
    /// let mut ng = NormalGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(ng.r(), 1.2);
    ///
    /// ng.set_r(2.1).unwrap();
    /// assert_eq!(ng.r(), 2.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalGamma;
    /// # let mut ng = NormalGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(ng.set_r(2.1).is_ok());
    ///
    /// // must be greater than zero
    /// assert!(ng.set_r(0.0).is_err());
    /// assert!(ng.set_r(-1.0).is_err());
    ///
    ///
    /// assert!(ng.set_r(std::f64::INFINITY).is_err());
    /// assert!(ng.set_r(std::f64::NEG_INFINITY).is_err());
    /// assert!(ng.set_r(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_r(&mut self, r: f64) -> Result<(), NormalGammaError> {
        if !r.is_finite() {
            Err(NormalGammaError::RNotFinite { r })
        } else if r <= 0.0 {
            Err(NormalGammaError::RTooLow { r })
        } else {
            self.set_r_unchecked(r);
            Ok(())
        }
    }

    /// Set the value of r without input validation
    #[inline]
    pub fn set_r_unchecked(&mut self, r: f64) {
        self.r = r;
    }

    /// Get the s parameter
    #[inline]
    pub fn s(&self) -> f64 {
        self.s
    }

    /// Set the value of s
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::NormalGamma;
    ///
    /// let mut ng = NormalGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(ng.s(), 2.3);
    ///
    /// ng.set_s(3.2).unwrap();
    /// assert_eq!(ng.s(), 3.2);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalGamma;
    /// # let mut ng = NormalGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(ng.set_s(2.1).is_ok());
    ///
    /// // must be greater than zero
    /// assert!(ng.set_s(0.0).is_err());
    /// assert!(ng.set_s(-1.0).is_err());
    ///
    ///
    /// assert!(ng.set_s(std::f64::INFINITY).is_err());
    /// assert!(ng.set_s(std::f64::NEG_INFINITY).is_err());
    /// assert!(ng.set_s(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_s(&mut self, s: f64) -> Result<(), NormalGammaError> {
        if !s.is_finite() {
            Err(NormalGammaError::SNotFinite { s })
        } else if s <= 0.0 {
            Err(NormalGammaError::STooLow { s })
        } else {
            self.set_s_unchecked(s);
            Ok(())
        }
    }

    /// Set the value of s without input validation
    #[inline]
    pub fn set_s_unchecked(&mut self, s: f64) {
        self.s = s;
    }

    /// Get the v parameter
    #[inline]
    pub fn v(&self) -> f64 {
        self.v
    }

    /// Set the value of v
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::NormalGamma;
    ///
    /// let mut ng = NormalGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(ng.v(), 3.4);
    ///
    /// ng.set_v(4.3).unwrap();
    /// assert_eq!(ng.v(), 4.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalGamma;
    /// # let mut ng = NormalGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(ng.set_v(2.1).is_ok());
    ///
    /// // must be greater than zero
    /// assert!(ng.set_v(0.0).is_err());
    /// assert!(ng.set_v(-1.0).is_err());
    ///
    ///
    /// assert!(ng.set_v(std::f64::INFINITY).is_err());
    /// assert!(ng.set_v(std::f64::NEG_INFINITY).is_err());
    /// assert!(ng.set_v(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_v(&mut self, v: f64) -> Result<(), NormalGammaError> {
        if !v.is_finite() {
            Err(NormalGammaError::VNotFinite { v })
        } else if v <= 0.0 {
            Err(NormalGammaError::VTooLow { v })
        } else {
            self.set_v_unchecked(v);
            Ok(())
        }
    }

    /// Set the value of v without input validation
    #[inline]
    pub fn set_v_unchecked(&mut self, v: f64) {
        self.v = v;
    }

    /// Return (m, r, s, v)
    #[inline]
    pub fn params(&self) -> (f64, f64, f64, f64) {
        (self.m, self.r, self.s, self.v)
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
        // TODO: could cache the gamma and Gaussian distributions
        let rho = (x.sigma() * x.sigma()).recip();
        let lnf_rho =
            Gamma::new_unchecked(self.v / 2.0, self.s / 2.0).ln_f(&rho);
        let prior_sigma = (self.r * rho).recip().sqrt();
        let lnf_mu = Gaussian::new_unchecked(self.m, prior_sigma).ln_f(&x.mu());
        lnf_rho + lnf_mu
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Gaussian {
        // NOTE: The parameter errors in this fn shouldn't happen if the prior
        // parameters are valid.
        // BAX: The problem is, in another library, I've been having trouble
        // with the posterior having invalid parameters, or with a valid
        // posterior drawing Inf sigma, which I suppose means it is drawing a
        // zero rho. Since things seem to go wrong in normal use, I'm using the
        // `new` constructors here rather than `new_unchecked` because I want to
        // catch things when they go wrong here so they don't spread. Of course,
        // all this input validation hurts performance 😞.
        let rho: f64 = Gamma::new(self.v / 2.0, self.s / 2.0)
            .map_err(|err| {
                panic!("Invalid ρ params when drawing Gaussian: {}", err)
            })
            .unwrap()
            .draw(&mut rng);

        let sigma = if rho.is_infinite() {
            std::f64::EPSILON
        } else {
            rho.recip().sqrt()
        };

        let post_sigma: f64 = self.r.recip().sqrt() * sigma;
        let mu: f64 = Gaussian::new(self.m, post_sigma)
            .map_err(|err| {
                panic!("Invalid μ params when drawing Gaussian: {}", err)
            })
            .unwrap()
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

impl std::error::Error for NormalGammaError {}

impl fmt::Display for NormalGammaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MNotFinite { m } => write!(f, "non-finite m: {}", m),
            Self::RNotFinite { r } => write!(f, "non-finite r: {}", r),
            Self::SNotFinite { s } => write!(f, "non-finite s: {}", s),
            Self::VNotFinite { v } => write!(f, "non-finite v: {}", v),
            Self::RTooLow { r } => {
                write!(f, "r ({}) must be greater than zero", r)
            }
            Self::STooLow { s } => {
                write!(f, "s ({}) must be greater than zero", s)
            }
            Self::VTooLow { v } => {
                write!(f, "v ({}) must be greater than zero", v)
            }
        }
    }
}

// TODO: tests!
