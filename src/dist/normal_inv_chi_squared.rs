//! A common conjugate prior for Gaussians with unknown mean and variance
//!
//! For a reference see section 6 of [Kevin Murphy's
//! whitepaper](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf).
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

mod gaussian_prior;

use crate::dist::{Gaussian, ScaledInvChiSquared};
use crate::impl_display;
use crate::misc::ln_gammafn;
use crate::traits::{HasDensity, Parameterized, Sampleable};
use rand::Rng;
use std::sync::OnceLock;

/// Prior for Gaussian
///
/// Given `x ~ N(μ, σ)`, the Normal Inverse Chi Squared prior implies that
/// `μ ~ N(m, σ/√k)` and `σ² ~ ScaledInvChiSquared(v, s2)`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct NormalInvChiSquared {
    m: f64,
    k: f64,
    v: f64,
    s2: f64,
    /// Cached scaled inv X^2
    #[cfg_attr(feature = "serde1", serde(skip))]
    scaled_inv_x2: OnceLock<ScaledInvChiSquared>,
}

pub struct NormalInvChiSquaredParameters {
    pub m: f64,
    pub k: f64,
    pub v: f64,
    pub s2: f64,
}

impl Parameterized for NormalInvChiSquared {
    type Parameters = NormalInvChiSquaredParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            m: self.m(),
            k: self.k(),
            v: self.v(),
            s2: self.s2(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.m, params.k, params.v, params.s2)
    }
}

impl PartialEq for NormalInvChiSquared {
    fn eq(&self, other: &Self) -> bool {
        self.m == other.m
            && self.k == other.k
            && self.v == other.v
            && self.s2 == other.s2
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum NormalInvChiSquaredError {
    /// The m parameter is infinite or NaN
    MNotFinite { m: f64 },
    /// The k parameter is less than or equal to zero
    KTooLow { k: f64 },
    /// The k parameter is infinite or NaN
    KNotFinite { k: f64 },
    /// The v parameter is less than or equal to zero
    VTooLow { v: f64 },
    /// The v parameter is infinite or NaN
    VNotFinite { v: f64 },
    /// The s2 parameter is less than or equal to zero
    S2TooLow { s2: f64 },
    /// The s2 parameter is infinite or NaN
    S2NotFinite { s2: f64 },
}

impl NormalInvChiSquared {
    /// Create a new Normal Inverse Gamma distribution
    ///
    /// # Arguments
    /// - m: The prior mean
    /// - k: How strongly we believe the prior mean (in prior
    ///   pseudo-observations)
    /// - v: How strongly we believe the prior variance (in prior
    ///   pseudo-observations)
    /// - s2: The prior variance
    pub fn new(
        m: f64,
        k: f64,
        v: f64,
        s2: f64,
    ) -> Result<Self, NormalInvChiSquaredError> {
        if !m.is_finite() {
            Err(NormalInvChiSquaredError::MNotFinite { m })
        } else if !k.is_finite() {
            Err(NormalInvChiSquaredError::KNotFinite { k })
        } else if !v.is_finite() {
            Err(NormalInvChiSquaredError::VNotFinite { v })
        } else if !s2.is_finite() {
            Err(NormalInvChiSquaredError::S2NotFinite { s2 })
        } else if v <= 0.0 {
            Err(NormalInvChiSquaredError::VTooLow { v })
        } else if k <= 0.0 {
            Err(NormalInvChiSquaredError::KTooLow { k })
        } else if s2 <= 0.0 {
            Err(NormalInvChiSquaredError::S2TooLow { s2 })
        } else {
            Ok(NormalInvChiSquared {
                m,
                k,
                v,
                s2,
                scaled_inv_x2: OnceLock::new(),
            })
        }
    }

    /// Creates a new `NormalInvChiSquared` without checking whether the
    /// parameters are valid.
    #[inline]
    #[must_use]
    pub fn new_unchecked(m: f64, k: f64, v: f64, s2: f64) -> Self {
        NormalInvChiSquared {
            m,
            k,
            v,
            s2,
            scaled_inv_x2: OnceLock::new(),
        }
    }

    /// Returns (m, k, v, s2)
    #[inline]
    pub fn params(&self) -> (f64, f64, f64, f64) {
        (self.m, self.k, self.v, self.s2)
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
    /// use rv::dist::NormalInvChiSquared;
    ///
    /// let mut nix = NormalInvChiSquared::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(nix.m(), 0.0);
    ///
    /// nix.set_m(-1.1).unwrap();
    /// assert_eq!(nix.m(), -1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalInvChiSquared;
    /// # let mut nix = NormalInvChiSquared::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(nix.set_m(-1.1).is_ok());
    /// assert!(nix.set_m(f64::INFINITY).is_err());
    /// assert!(nix.set_m(f64::NEG_INFINITY).is_err());
    /// assert!(nix.set_m(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_m(&mut self, m: f64) -> Result<(), NormalInvChiSquaredError> {
        if m.is_finite() {
            self.set_m_unchecked(m);
            Ok(())
        } else {
            Err(NormalInvChiSquaredError::MNotFinite { m })
        }
    }

    /// Set the value of m without input validation
    #[inline]
    pub fn set_m_unchecked(&mut self, m: f64) {
        self.m = m;
    }

    /// Get the k parameter
    #[inline]
    pub fn k(&self) -> f64 {
        self.k
    }

    /// Set the value of k
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::NormalInvChiSquared;
    ///
    /// let mut nix = NormalInvChiSquared::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(nix.k(), 1.2);
    ///
    /// nix.set_k(4.3).unwrap();
    /// assert_eq!(nix.k(), 4.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalInvChiSquared;
    /// # let mut nix = NormalInvChiSquared::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(nix.set_k(2.1).is_ok());
    ///
    /// // must be greater than zero
    /// assert!(nix.set_k(0.0).is_err());
    /// assert!(nix.set_k(-1.0).is_err());
    ///
    ///
    /// assert!(nix.set_k(f64::INFINITY).is_err());
    /// assert!(nix.set_k(f64::NEG_INFINITY).is_err());
    /// assert!(nix.set_k(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_k(&mut self, k: f64) -> Result<(), NormalInvChiSquaredError> {
        if !k.is_finite() {
            Err(NormalInvChiSquaredError::KNotFinite { k })
        } else if k <= 0.0 {
            Err(NormalInvChiSquaredError::KTooLow { k })
        } else {
            self.set_k_unchecked(k);
            Ok(())
        }
    }

    /// Set the value of k without input validation
    #[inline]
    pub fn set_k_unchecked(&mut self, k: f64) {
        self.k = k;
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
    /// use rv::dist::NormalInvChiSquared;
    ///
    /// let mut nix = NormalInvChiSquared::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(nix.v(), 2.3);
    ///
    /// nix.set_v(4.3).unwrap();
    /// assert_eq!(nix.v(), 4.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalInvChiSquared;
    /// # let mut nix = NormalInvChiSquared::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(nix.set_v(2.1).is_ok());
    ///
    /// // must be greater than zero
    /// assert!(nix.set_v(0.0).is_err());
    /// assert!(nix.set_v(-1.0).is_err());
    ///
    ///
    /// assert!(nix.set_v(f64::INFINITY).is_err());
    /// assert!(nix.set_v(f64::NEG_INFINITY).is_err());
    /// assert!(nix.set_v(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_v(&mut self, v: f64) -> Result<(), NormalInvChiSquaredError> {
        if !v.is_finite() {
            Err(NormalInvChiSquaredError::VNotFinite { v })
        } else if v <= 0.0 {
            Err(NormalInvChiSquaredError::VTooLow { v })
        } else {
            self.set_v_unchecked(v);
            self.scaled_inv_x2 = OnceLock::new();
            Ok(())
        }
    }

    /// Set the value of v without input validation
    #[inline]
    pub fn set_v_unchecked(&mut self, v: f64) {
        self.v = v;
        self.scaled_inv_x2 = OnceLock::new();
    }

    /// Get the s2 parameter
    #[inline]
    pub fn s2(&self) -> f64 {
        self.s2
    }

    /// Set the value of s2
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::NormalInvChiSquared;
    ///
    /// let mut nix = NormalInvChiSquared::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(nix.s2(), 3.4);
    ///
    /// nix.set_s2(4.3).unwrap();
    /// assert_eq!(nix.s2(), 4.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalInvChiSquared;
    /// # let mut nix = NormalInvChiSquared::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(nix.set_s2(2.1).is_ok());
    ///
    /// // must be greater than zero
    /// assert!(nix.set_s2(0.0).is_err());
    /// assert!(nix.set_s2(-1.0).is_err());
    ///
    ///
    /// assert!(nix.set_s2(f64::INFINITY).is_err());
    /// assert!(nix.set_s2(f64::NEG_INFINITY).is_err());
    /// assert!(nix.set_s2(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_s2(&mut self, s2: f64) -> Result<(), NormalInvChiSquaredError> {
        if !s2.is_finite() {
            Err(NormalInvChiSquaredError::S2NotFinite { s2 })
        } else if s2 <= 0.0 {
            Err(NormalInvChiSquaredError::S2TooLow { s2 })
        } else {
            self.set_s2_unchecked(s2);
            self.scaled_inv_x2 = OnceLock::new();
            Ok(())
        }
    }

    /// Set the value of s2 without input validation
    #[inline]
    pub fn set_s2_unchecked(&mut self, s2: f64) {
        self.s2 = s2;
        self.scaled_inv_x2 = OnceLock::new();
    }

    #[inline]
    pub fn scaled_inv_x2(&self) -> &ScaledInvChiSquared {
        self.scaled_inv_x2
            .get_or_init(|| ScaledInvChiSquared::new_unchecked(self.v, self.s2))
    }

    #[inline]
    pub fn ln_z(&self) -> f64 {
        let k = self.k;
        let v = self.v;
        let s2 = self.s2;
        let ln_gamma_half_v = ln_gammafn(0.5 * self.v);
        // -0.5 * k.ln() + v2.ln_gamma().0 - v2 * (v * s2).ln()
        let term = (v * s2).ln().mul_add(-0.5 * v, ln_gamma_half_v);
        k.ln().mul_add(-0.5, term)
    }
}

impl From<&NormalInvChiSquared> for String {
    fn from(nix: &NormalInvChiSquared) -> String {
        format!(
            "Normal-Inverse-X²(m: {}, k: {}, v: {}, s2: {})",
            nix.m, nix.k, nix.v, nix.s2
        )
    }
}

impl_display!(NormalInvChiSquared);

impl HasDensity<Gaussian> for NormalInvChiSquared {
    fn ln_f(&self, x: &Gaussian) -> f64 {
        let lnf_sigma = self.scaled_inv_x2().ln_f(&(x.sigma() * x.sigma()));
        let prior_sigma = x.sigma() / self.k.sqrt();
        let lnf_mu = Gaussian::new_unchecked(self.m, prior_sigma).ln_f(&x.mu());
        lnf_sigma + lnf_mu
    }
}

impl Sampleable<Gaussian> for NormalInvChiSquared {
    fn draw<R: Rng>(&self, mut rng: &mut R) -> Gaussian {
        let var: f64 = self.scaled_inv_x2().draw(&mut rng);

        let sigma = if var <= 0.0 { f64::EPSILON } else { var.sqrt() };

        let post_sigma: f64 = sigma / self.k.sqrt();
        let mu: f64 = Gaussian::new(self.m, post_sigma)
            .map_err(|err| {
                panic!("Invalid μ params when drawing Gaussian: {err}")
            })
            .unwrap()
            .draw(&mut rng);

        Gaussian::new(mu, var.sqrt()).expect("Invalid params")
    }
}

impl std::error::Error for NormalInvChiSquaredError {}

impl std::fmt::Display for NormalInvChiSquaredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MNotFinite { m } => write!(f, "non-finite m: {m}"),
            Self::KNotFinite { k } => write!(f, "non-finite k: {k}"),
            Self::VNotFinite { v } => write!(f, "non-finite v: {v}"),
            Self::S2NotFinite { s2 } => write!(f, "non-finite s2: {s2}"),
            Self::KTooLow { k } => {
                write!(f, "k ({k}) must be greater than zero")
            }
            Self::VTooLow { v } => {
                write!(f, "v ({v}) must be greater than zero")
            }
            Self::S2TooLow { s2 } => {
                write!(f, "s2 ({s2}) must be greater than zero")
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{test_basic_impls, verify_cache_resets};

    test_basic_impls!(
        Gaussian,
        NormalInvChiSquared,
        NormalInvChiSquared::new(0.1, 1.2, 2.3, 3.4).unwrap()
    );

    verify_cache_resets!(
        [unchecked],
        ln_f_is_same_after_reset_unchecked_v_identically,
        set_v_unchecked,
        NormalInvChiSquared::new(0.1, 1.2, 2.3, 3.4).unwrap(),
        Gaussian::new(-1.2, 0.4).unwrap(),
        2.3,
        std::f64::consts::PI
    );

    verify_cache_resets!(
        [checked],
        ln_f_is_same_after_reset_checked_v_identically,
        set_v,
        NormalInvChiSquared::new(0.1, 1.2, 2.3, 3.4).unwrap(),
        Gaussian::new(-1.2, 0.4).unwrap(),
        2.3,
        std::f64::consts::PI
    );

    verify_cache_resets!(
        [unchecked],
        ln_f_is_same_after_reset_unchecked_s2_identically,
        set_s2_unchecked,
        NormalInvChiSquared::new(0.1, 1.2, 2.3, 3.4).unwrap(),
        Gaussian::new(-1.2, 0.4).unwrap(),
        3.4,
        0.8
    );

    verify_cache_resets!(
        [checked],
        ln_f_is_same_after_reset_checked_s2_identically,
        set_s2,
        NormalInvChiSquared::new(0.1, 1.2, 2.3, 3.4).unwrap(),
        Gaussian::new(-1.2, 0.4).unwrap(),
        3.4,
        0.8
    );
}
