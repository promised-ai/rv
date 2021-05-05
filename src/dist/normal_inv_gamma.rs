//! A common conjugate prior for Gaussians with unknown mean and variance
//!
//! For a reference see section 6 of [Kevin Murphy's
//! whitepaper](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf).
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

mod gaussian_prior;

use crate::dist::{Gaussian, InvGamma};
use crate::impl_display;
use crate::traits::Rv;
use rand::Rng;

/// Prior for Gaussian
///
/// Given `x ~ N(μ, σ)`, the Normal Inverse Gamma prior implies that
/// `μ ~ N(m, sqrt(v)σ)` and `ρ ~ InvGamma(a, b)`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NormalInvGamma {
    m: f64,
    v: f64,
    a: f64,
    b: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum NormalInvGammaError {
    /// The m parameter is infinite or NaN
    MNotFinite { m: f64 },
    /// The v parameter is less than or equal to zero
    VTooLow { v: f64 },
    /// The v parameter is infinite or NaN
    VNotFinite { v: f64 },
    /// The a parameter is less than or equal to zero
    ATooLow { a: f64 },
    /// The a parameter is infinite or NaN
    ANotFinite { a: f64 },
    /// The b parameter is less than or equal to zero
    BTooLow { b: f64 },
    /// The b parameter is infinite or NaN
    BNotFinite { b: f64 },
}

impl NormalInvGamma {
    /// Create a new Normal Inverse Gamma distribution
    ///
    /// # Arguments
    /// - m: The prior mean
    /// - v: Relative variance of μ versus data
    /// - a: The mean of variance is b / (a - 1)
    /// - b: Degrees of freedom of the variance
    pub fn new(
        m: f64,
        v: f64,
        a: f64,
        b: f64,
    ) -> Result<Self, NormalInvGammaError> {
        if !m.is_finite() {
            Err(NormalInvGammaError::MNotFinite { m })
        } else if !v.is_finite() {
            Err(NormalInvGammaError::VNotFinite { v })
        } else if !a.is_finite() {
            Err(NormalInvGammaError::ANotFinite { a })
        } else if !b.is_finite() {
            Err(NormalInvGammaError::BNotFinite { b })
        } else if v <= 0.0 {
            Err(NormalInvGammaError::VTooLow { v })
        } else if a <= 0.0 {
            Err(NormalInvGammaError::ATooLow { a })
        } else if b <= 0.0 {
            Err(NormalInvGammaError::BTooLow { b })
        } else {
            Ok(NormalInvGamma { m, v, a, b })
        }
    }

    /// Creates a new NormalInvGamma without checking whether the parameters are
    /// valid.
    #[inline(always)]
    pub fn new_unchecked(m: f64, v: f64, a: f64, b: f64) -> Self {
        NormalInvGamma { m, v, a, b }
    }

    /// Returns (m, v, a, b)
    #[inline(always)]
    pub fn params(&self) -> (f64, f64, f64, f64) {
        (self.m, self.v, self.a, self.b)
    }

    /// Get the m parameter
    #[inline(always)]
    pub fn m(&self) -> f64 {
        self.m
    }

    /// Set the value of m
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::NormalInvGamma;
    ///
    /// let mut nig = NormalInvGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(nig.m(), 0.0);
    ///
    /// nig.set_m(-1.1).unwrap();
    /// assert_eq!(nig.m(), -1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalInvGamma;
    /// # let mut nig = NormalInvGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(nig.set_m(-1.1).is_ok());
    /// assert!(nig.set_m(std::f64::INFINITY).is_err());
    /// assert!(nig.set_m(std::f64::NEG_INFINITY).is_err());
    /// assert!(nig.set_m(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_m(&mut self, m: f64) -> Result<(), NormalInvGammaError> {
        if m.is_finite() {
            self.set_m_unchecked(m);
            Ok(())
        } else {
            Err(NormalInvGammaError::MNotFinite { m })
        }
    }

    /// Set the value of m without input validation
    #[inline(always)]
    pub fn set_m_unchecked(&mut self, m: f64) {
        self.m = m;
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
    /// use rv::dist::NormalInvGamma;
    ///
    /// let mut nig = NormalInvGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(nig.v(), 1.2);
    ///
    /// nig.set_v(4.3).unwrap();
    /// assert_eq!(nig.v(), 4.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalInvGamma;
    /// # let mut nig = NormalInvGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(nig.set_v(2.1).is_ok());
    ///
    /// // must be greater than zero
    /// assert!(nig.set_v(0.0).is_err());
    /// assert!(nig.set_v(-1.0).is_err());
    ///
    ///
    /// assert!(nig.set_v(std::f64::INFINITY).is_err());
    /// assert!(nig.set_v(std::f64::NEG_INFINITY).is_err());
    /// assert!(nig.set_v(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_v(&mut self, v: f64) -> Result<(), NormalInvGammaError> {
        if !v.is_finite() {
            Err(NormalInvGammaError::VNotFinite { v })
        } else if v <= 0.0 {
            Err(NormalInvGammaError::VTooLow { v })
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

    /// Get the a parameter
    #[inline]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Set the value of a
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::NormalInvGamma;
    ///
    /// let mut nig = NormalInvGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(nig.a(), 2.3);
    ///
    /// nig.set_a(4.3).unwrap();
    /// assert_eq!(nig.a(), 4.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalInvGamma;
    /// # let mut nig = NormalInvGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(nig.set_a(2.1).is_ok());
    ///
    /// // must be greater than zero
    /// assert!(nig.set_a(0.0).is_err());
    /// assert!(nig.set_a(-1.0).is_err());
    ///
    ///
    /// assert!(nig.set_a(std::f64::INFINITY).is_err());
    /// assert!(nig.set_a(std::f64::NEG_INFINITY).is_err());
    /// assert!(nig.set_a(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_a(&mut self, a: f64) -> Result<(), NormalInvGammaError> {
        if !a.is_finite() {
            Err(NormalInvGammaError::ANotFinite { a })
        } else if a <= 0.0 {
            Err(NormalInvGammaError::ATooLow { a })
        } else {
            self.set_a_unchecked(a);
            Ok(())
        }
    }

    /// Set the value of a without input validation
    #[inline]
    pub fn set_a_unchecked(&mut self, a: f64) {
        self.a = a;
    }

    /// Get the b parameter
    #[inline]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Set the value of b
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::NormalInvGamma;
    ///
    /// let mut nig = NormalInvGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(nig.b(), 3.4);
    ///
    /// nig.set_b(4.3).unwrap();
    /// assert_eq!(nig.b(), 4.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NormalInvGamma;
    /// # let mut nig = NormalInvGamma::new(0.0, 1.2, 2.3, 3.4).unwrap();
    /// assert!(nig.set_b(2.1).is_ok());
    ///
    /// // must be greater than zero
    /// assert!(nig.set_b(0.0).is_err());
    /// assert!(nig.set_b(-1.0).is_err());
    ///
    ///
    /// assert!(nig.set_b(std::f64::INFINITY).is_err());
    /// assert!(nig.set_b(std::f64::NEG_INFINITY).is_err());
    /// assert!(nig.set_b(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_b(&mut self, b: f64) -> Result<(), NormalInvGammaError> {
        if !b.is_finite() {
            Err(NormalInvGammaError::BNotFinite { b })
        } else if b <= 0.0 {
            Err(NormalInvGammaError::BTooLow { b })
        } else {
            self.set_b_unchecked(b);
            Ok(())
        }
    }

    /// Set the value of b without input validation
    #[inline(always)]
    pub fn set_b_unchecked(&mut self, b: f64) {
        self.b = b;
    }
}

impl From<&NormalInvGamma> for String {
    fn from(nig: &NormalInvGamma) -> String {
        format!(
            "Normal-Inverse-Gamma(m: {}, v: {}, a: {}, b: {})",
            nig.m, nig.v, nig.a, nig.b
        )
    }
}

impl_display!(NormalInvGamma);

impl Rv<Gaussian> for NormalInvGamma {
    fn ln_f(&self, x: &Gaussian) -> f64 {
        // TODO: could cache the gamma and Gaussian distributions
        let mu = x.mu();
        let sigma = x.sigma();
        let lnf_sigma =
            InvGamma::new_unchecked(self.a, self.b).ln_f(&sigma.powi(2));
        let prior_sigma = self.v.sqrt() * sigma;
        let lnf_mu = Gaussian::new_unchecked(self.m, prior_sigma).ln_f(&mu);
        lnf_sigma + lnf_mu
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Gaussian {
        // NOTE: The parameter errors in this fn shouldn't happen if the prior
        // parameters are valid.
        let var: f64 = InvGamma::new(self.a, self.b)
            .map_err(|err| {
                panic!("Invalid σ² params when drawing Gaussian: {}", err)
            })
            .unwrap()
            .draw(&mut rng);

        let sigma = if var <= 0.0 {
            std::f64::EPSILON
        } else {
            var.sqrt()
        };

        let post_sigma: f64 = self.v.sqrt() * sigma;
        let mu: f64 = Gaussian::new(self.m, post_sigma)
            .map_err(|err| {
                panic!("Invalid μ params when drawing Gaussian: {}", err)
            })
            .unwrap()
            .draw(&mut rng);

        Gaussian::new(mu, sigma).expect("Invalid params")
    }
}
