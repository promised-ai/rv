//! Beta Binomial distribution of x in {0, ..., n}
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use rand::Rng;
use special::Beta as _;
use std::f64;
use std::fmt;
use std::sync::OnceLock;

use crate::impl_display;
use crate::misc::{ln_binom, ln_pflips};
use crate::traits::*;

/// [Beta Binomial distribution](https://en.wikipedia.org/wiki/Beta-binomial_distribution)
/// over k in {0, ..., n}
///
/// # Example
///
/// ```
/// use std::f64;
/// use rv::prelude::*;
///
/// let a = 3.0;
/// let b = 2.0;
/// let n = 20;
///
/// let beta = Beta::new(a, b).unwrap();
/// let beta_binom = BetaBinomial::new(n, a, b).unwrap();
///
/// let beta_mean: f64 = beta.mean().unwrap();
/// let beta_binom_mean: f64 = beta_binom.mean().unwrap();
/// assert!( (beta_mean * f64::from(n) - beta_binom_mean).abs() < 1E-12 );
/// ```
///
/// Some functions will panic when given data outside the supported range:
/// [0, n]
///
/// ```
/// # use rv::prelude::*;
/// let beta_binom = BetaBinomial::new(20, 3.0, 2.0).unwrap();
/// assert!(!beta_binom.supports(&21_u32));
/// ```
///
/// PMF calls will return 0 for out-of-support data
///
/// ```
/// # use rv::prelude::*;
/// # let beta_binom = BetaBinomial::new(20, 3.0, 2.0).unwrap();
/// let f = beta_binom.pmf(&21_u32);
/// assert_eq!(f, 0.0);
/// ```

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct BetaBinomial {
    /// Total number of trials
    n: u32,
    /// Analogous to Beta Distribution α parameter.
    alpha: f64,
    /// Analogous to Beta Distribution β parameter
    beta: f64,
    // ln_beta(alpha, beta)
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_beta_ab: OnceLock<f64>,
}

pub struct BetaBinomialParameters {
    pub n: u32,
    pub alpha: f64,
    pub beta: f64,
}

impl Parameterized for BetaBinomial {
    type Parameters = BetaBinomialParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            n: self.n(),
            alpha: self.alpha(),
            beta: self.beta(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.n, params.alpha, params.beta)
    }
}

impl PartialEq for BetaBinomial {
    fn eq(&self, other: &BetaBinomial) -> bool {
        self.n == other.n
            && self.alpha == other.alpha
            && self.beta == other.beta
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum BetaBinomialError {
    /// The alpha parameter is less than zero
    AlphaTooLow { alpha: f64 },
    /// The alpha parameter is infinite or NaN
    AlphaNotFinite { alpha: f64 },
    /// The beta parameter is less than zero
    BetaTooLow { beta: f64 },
    /// The beta parameter is infinite or NaN
    BetaNotFinite { beta: f64 },
    /// The number of trails is zero
    NIsZero,
}

impl BetaBinomial {
    /// Create a beta-binomal distirbution
    ///
    /// # Arguments
    ///
    /// - n: the total number of trials
    /// - alpha: the prior pseudo obersvations of success
    /// - beta: the prior pseudo obersvations of failure
    pub fn new(
        n: u32,
        alpha: f64,
        beta: f64,
    ) -> Result<Self, BetaBinomialError> {
        if alpha <= 0.0 {
            Err(BetaBinomialError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(BetaBinomialError::AlphaNotFinite { alpha })
        } else if beta <= 0.0 {
            Err(BetaBinomialError::BetaTooLow { beta })
        } else if !beta.is_finite() {
            Err(BetaBinomialError::BetaNotFinite { beta })
        } else if n == 0 {
            Err(BetaBinomialError::NIsZero)
        } else {
            Ok(BetaBinomial {
                n,
                alpha,
                beta,
                ln_beta_ab: OnceLock::new(),
            })
        }
    }

    /// Creates a new BetaBinomial without checking whether the parameters are
    /// valid.
    #[inline]
    pub fn new_unchecked(n: u32, alpha: f64, beta: f64) -> Self {
        BetaBinomial {
            n,
            alpha,
            beta,
            ln_beta_ab: OnceLock::new(),
        }
    }

    /// Evaluate or fetch cached log sigma
    #[inline]
    fn ln_beta_ab(&self) -> f64 {
        *self
            .ln_beta_ab
            .get_or_init(|| self.alpha.ln_beta(self.beta))
    }

    /// Get `n`, the number of trials.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::BetaBinomial;
    /// let bb = BetaBinomial::new(10, 1.0, 2.0).unwrap();
    /// assert_eq!(bb.n(), 10);
    /// ```
    #[inline]
    pub fn n(&self) -> u32 {
        self.n
    }

    /// Get the `alpha` parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::BetaBinomial;
    /// let bb = BetaBinomial::new(10, 1.0, 2.0).unwrap();
    /// assert_eq!(bb.alpha(), 1.0);
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
    /// use rv::dist::BetaBinomial;
    ///
    /// let mut bb = BetaBinomial::new(10, 1.0, 5.0).unwrap();
    ///
    /// bb.set_alpha(2.0).unwrap();
    /// assert_eq!(bb.alpha(), 2.0);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::BetaBinomial;
    /// # let mut bb = BetaBinomial::new(10, 1.0, 5.0).unwrap();
    /// assert!(bb.set_alpha(0.1).is_ok());
    /// assert!(bb.set_alpha(0.0).is_err());
    /// assert!(bb.set_alpha(-1.0).is_err());
    /// assert!(bb.set_alpha(f64::INFINITY).is_err());
    /// assert!(bb.set_alpha(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_alpha(&mut self, alpha: f64) -> Result<(), BetaBinomialError> {
        if alpha <= 0.0 {
            Err(BetaBinomialError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(BetaBinomialError::AlphaNotFinite { alpha })
        } else {
            self.set_alpha_unchecked(alpha);
            Ok(())
        }
    }

    /// Set alpha without input validation
    #[inline]
    pub fn set_alpha_unchecked(&mut self, alpha: f64) {
        self.ln_beta_ab = OnceLock::new();
        self.alpha = alpha
    }

    /// Get the `beta` parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::BetaBinomial;
    /// let bb = BetaBinomial::new(10, 1.0, 2.0).unwrap();
    /// assert_eq!(bb.beta(), 2.0);
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
    /// # use rv::dist::BetaBinomial;
    /// let mut bb = BetaBinomial::new(10, 1.0, 5.0).unwrap();
    ///
    /// bb.set_beta(2.0).unwrap();
    /// assert_eq!(bb.beta(), 2.0);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::BetaBinomial;
    /// # let mut bb = BetaBinomial::new(10, 1.0, 5.0).unwrap();
    /// assert!(bb.set_beta(0.1).is_ok());
    /// assert!(bb.set_beta(0.0).is_err());
    /// assert!(bb.set_beta(-1.0).is_err());
    /// assert!(bb.set_beta(f64::INFINITY).is_err());
    /// assert!(bb.set_beta(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_beta(&mut self, beta: f64) -> Result<(), BetaBinomialError> {
        if beta <= 0.0 {
            Err(BetaBinomialError::BetaTooLow { beta })
        } else if !beta.is_finite() {
            Err(BetaBinomialError::BetaNotFinite { beta })
        } else {
            self.set_beta_unchecked(beta);
            Ok(())
        }
    }

    /// Set beta without input validation
    #[inline]
    pub fn set_beta_unchecked(&mut self, beta: f64) {
        self.ln_beta_ab = OnceLock::new();
        self.beta = beta
    }

    /// Set the value of the n parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::BetaBinomial;
    ///
    /// let mut bb = BetaBinomial::new(10, 0.5, 0.5).unwrap();
    ///
    /// bb.set_n(11).unwrap();
    ///
    /// assert_eq!(bb.n(), 11);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::BetaBinomial;
    /// # let mut bb = BetaBinomial::new(10, 0.5, 0.5).unwrap();
    /// assert!(bb.set_n(11).is_ok());
    /// assert!(bb.set_n(1).is_ok());
    /// assert!(bb.set_n(0).is_err());
    /// ```
    #[inline]
    pub fn set_n(&mut self, n: u32) -> Result<(), BetaBinomialError> {
        if n == 0 {
            Err(BetaBinomialError::NIsZero)
        } else {
            self.set_n_unchecked(n);
            Ok(())
        }
    }

    /// Set the value of n without input validation
    #[inline]
    pub fn set_n_unchecked(&mut self, n: u32) {
        self.n = n
    }
}

impl From<&BetaBinomial> for String {
    fn from(bb: &BetaBinomial) -> String {
        format!("BetaBinomial({}; α: {}, β: {})", bb.n, bb.alpha, bb.beta)
    }
}

impl_display!(BetaBinomial);

macro_rules! impl_int_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for BetaBinomial {
            fn ln_f(&self, k: &$kind) -> f64 {
                let nf = f64::from(self.n);
                let kf = *k as f64;
                ln_binom(nf, kf)
                    + (kf + self.alpha).ln_beta(nf - kf + self.beta)
                    - self.ln_beta_ab()
            }
        }

        impl Sampleable<$kind> for BetaBinomial {
            fn draw<R: Rng>(&self, mut rng: &mut R) -> $kind {
                self.sample(1, &mut rng)[0]
            }

            fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<$kind> {
                // NOTE: Could speed this up if we didn't compute the
                // k-independent terms in ln_f. But if we're caching well w/in
                // the distribution objects, this is negligible.
                let ln_weights: Vec<f64> =
                    (0..=self.n).map(|x| self.ln_f(&x)).collect();

                ln_pflips(&ln_weights, n, true, &mut rng)
                    .iter()
                    .map(|k| *k as $kind)
                    .collect()
            }
        }

        impl Support<$kind> for BetaBinomial {
            #[allow(unused_comparisons)]
            fn supports(&self, k: &$kind) -> bool {
                *k >= 0 && *k <= self.n as $kind
            }
        }

        impl DiscreteDistr<$kind> for BetaBinomial {}

        impl Cdf<$kind> for BetaBinomial {
            fn cdf(&self, k: &$kind) -> f64 {
                // XXX: Slow and awful. Could make this faster with
                // hypergeometric function, but the `special` crate doesn't
                // implement it
                (0..=*k).fold(0.0, |acc, x| acc + self.pmf(&x))
            }
        }
    };
}

impl Mean<f64> for BetaBinomial {
    fn mean(&self) -> Option<f64> {
        let nf = f64::from(self.n);
        let m = self.alpha / (self.alpha + self.beta);
        Some(nf * m)
    }
}

impl Variance<f64> for BetaBinomial {
    fn variance(&self) -> Option<f64> {
        let nf = f64::from(self.n);
        let apb = self.alpha + self.beta;
        let v_numer = nf * self.alpha * self.beta * (apb + nf);
        let v_denom = apb * apb * (apb + 1.0);
        Some(v_numer / v_denom)
    }
}

impl_int_traits!(u8);
impl_int_traits!(u16);
impl_int_traits!(u32);
impl_int_traits!(u64);
impl_int_traits!(usize);

impl_int_traits!(i8);
impl_int_traits!(i16);
impl_int_traits!(i32);
impl_int_traits!(i64);

impl std::error::Error for BetaBinomialError {}

impl fmt::Display for BetaBinomialError {
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
            Self::NIsZero => write!(f, "n was zero"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;

    test_basic_impls!(
        u32,
        BetaBinomial,
        BetaBinomial::new(10, 0.2, 0.7).unwrap()
    );

    #[test]
    fn new() {
        let beta_binom = BetaBinomial::new(10, 0.1, 0.2).unwrap();
        assert_eq!(beta_binom.n, 10);
        assert::close(beta_binom.alpha(), 0.1, TOL);
        assert::close(beta_binom.beta(), 0.2, TOL);
    }

    #[test]
    fn pmf() {
        let beta_binom = BetaBinomial::new(10, 0.5, 2.0).unwrap();
        // Values from wolfram alpha
        let target = vec![
            0.387_765,
            0.176_257,
            0.118_973,
            0.088_128_3,
            0.067_473_2,
            0.052_050_8,
            0.039_761,
            0.029_536_8,
            0.020_768,
            0.013_076_2,
            0.006_211_18,
        ];
        let pmfs: Vec<f64> = (0..=10).map(|k| beta_binom.pmf(&k)).collect();
        assert::close(pmfs, target, 1E-6);
    }
}
