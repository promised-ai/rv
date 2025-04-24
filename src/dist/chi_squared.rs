//! Χ<sup>2</sup> over x in (0, ∞)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::misc::ln_gammafn;
use crate::traits::{Cdf, ContinuousDistr, HasDensity, Kurtosis, Mean, Mode, Parameterized, Sampleable, Scalable, Shiftable, Skewness, Support, Variance};
use rand::Rng;
use special::Gamma;
use std::f64::consts::LN_2;
use std::fmt;

/// [Χ<sup>2</sup> distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution)
/// Χ<sup>2</sup>(k).
///
/// # Example
///
/// ```
/// use rv::prelude::*;
///
/// let x2 = ChiSquared::new(2.0).unwrap();
/// ```
///
/// Parameters struct for the Chi-squared distribution
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct ChiSquaredParameters {
    /// Degrees of freedom in (0, ∞)
    pub k: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct ChiSquared {
    /// Degrees of freedom in (0, ∞)
    k: f64,
}

impl Parameterized for ChiSquared {
    type Parameters = ChiSquaredParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters { k: self.k() }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.k)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum ChiSquaredError {
    /// k parameter is less than or equal to zero
    KTooLow { k: f64 },
    /// k parameter is infinite or NaN
    KNotFinite { k: f64 },
}

impl ChiSquared {
    /// Create a new Chi-squared distribution
    ///
    /// # Arguments
    /// - k: Degrees of freedom in (0, ∞)
    #[inline]
    pub fn new(k: f64) -> Result<Self, ChiSquaredError> {
        if k <= 0.0 {
            Err(ChiSquaredError::KTooLow { k })
        } else if !k.is_finite() {
            Err(ChiSquaredError::KNotFinite { k })
        } else {
            Ok(ChiSquared { k })
        }
    }

    /// Create a new `ChiSquared` without checking whether the parameters are
    /// valid.
    #[inline]
    #[must_use] pub fn new_unchecked(k: f64) -> Self {
        ChiSquared { k }
    }

    /// Get the degrees of freedom, `k`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::ChiSquared;
    /// let x2 = ChiSquared::new(1.2).unwrap();
    /// assert_eq!(x2.k(), 1.2);
    /// ```
    #[inline]
    #[must_use] pub fn k(&self) -> f64 {
        self.k
    }

    /// Set the degrees of freedom, `k`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::ChiSquared;
    /// let mut x2 = ChiSquared::new(1.2).unwrap();
    ///
    /// x2.set_k(2.2).unwrap();
    /// assert_eq!(x2.k(), 2.2);
    /// ```
    ///
    /// Will error given invalid values.
    ///
    /// ```rust
    /// # use rv::dist::ChiSquared;
    /// # let mut x2 = ChiSquared::new(1.2).unwrap();
    /// assert!(x2.set_k(2.2).is_ok());
    /// assert!(x2.set_k(0.0).is_err());
    /// assert!(x2.set_k(-1.0).is_err());
    /// assert!(x2.set_k(f64::NAN).is_err());
    /// assert!(x2.set_k(f64::INFINITY).is_err());
    /// ```
    #[inline]
    pub fn set_k(&mut self, k: f64) -> Result<(), ChiSquaredError> {
        if !k.is_finite() {
            Err(ChiSquaredError::KNotFinite { k })
        } else if k > 0.0 {
            self.set_k_unchecked(k);
            Ok(())
        } else {
            Err(ChiSquaredError::KTooLow { k })
        }
    }

    #[inline]
    pub fn set_k_unchecked(&mut self, k: f64) {
        self.k = k;
    }
}

impl From<&ChiSquared> for String {
    fn from(x2: &ChiSquared) -> String {
        format!("χ²({})", x2.k)
    }
}

impl_display!(ChiSquared);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for ChiSquared {
            fn ln_f(&self, x: &$kind) -> f64 {
                let k2 = self.k / 2.0;
                let xf = f64::from(*x);
                // TODO: cache (k2 - LN_2 - k2.ln_gamma().0)
                k2.mul_add(-LN_2, (k2 - 1.0).mul_add(xf.ln(), -xf / 2.0))
                    - ln_gammafn(k2)
            }
        }

        impl Sampleable<$kind> for ChiSquared {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let x2 = rand_distr::ChiSquared::new(self.k).unwrap();
                rng.sample(x2) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let x2 = rand_distr::ChiSquared::new(self.k).unwrap();
                (0..n).map(|_| rng.sample(x2) as $kind).collect()
            }
        }

        impl Support<$kind> for ChiSquared {
            fn supports(&self, x: &$kind) -> bool {
                *x > 0.0 && x.is_finite()
            }
        }

        impl ContinuousDistr<$kind> for ChiSquared {}

        impl Mean<$kind> for ChiSquared {
            fn mean(&self) -> Option<$kind> {
                Some(self.k as $kind)
            }
        }

        impl Mode<$kind> for ChiSquared {
            fn mode(&self) -> Option<$kind> {
                Some(0.0_f64.max(self.k - 2.0) as $kind)
            }
        }

        impl Variance<$kind> for ChiSquared {
            fn variance(&self) -> Option<$kind> {
                Some((self.k * 2.0) as $kind)
            }
        }

        impl Cdf<$kind> for ChiSquared {
            fn cdf(&self, x: &$kind) -> f64 {
                (f64::from(*x) / 2.0).inc_gamma(self.k / 2.0)
            }
        }
    };
}

crate::impl_shiftable!(ChiSquared);
crate::impl_scalable!(ChiSquared);

impl Skewness for ChiSquared {
    fn skewness(&self) -> Option<f64> {
        Some((8.0 / self.k).sqrt())
    }
}

impl Kurtosis for ChiSquared {
    fn kurtosis(&self) -> Option<f64> {
        Some(12.0 / self.k)
    }
}

impl_traits!(f64);
impl_traits!(f32);

impl std::error::Error for ChiSquaredError {}

impl fmt::Display for ChiSquaredError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KTooLow { k } => {
                write!(f, "k ({k}) must be greater than zero")
            }
            Self::KNotFinite { k } => write!(f, "k ({k}) must be finite"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;
    use crate::test_basic_impls;
    use std::f64;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!(f64, ChiSquared, ChiSquared::new(3.2).unwrap());

    #[test]
    fn new() {
        let x2 = ChiSquared::new(3.2).unwrap();
        assert::close(x2.k, 3.2, TOL);
    }

    #[test]
    fn new_should_reject_k_leq_zero() {
        assert!(ChiSquared::new(f64::MIN_POSITIVE).is_ok());
        assert!(ChiSquared::new(0.0).is_err());
        assert!(ChiSquared::new(-f64::MIN_POSITIVE).is_err());
        assert!(ChiSquared::new(-1.0).is_err());
    }

    #[test]
    fn new_should_reject_non_finite_k() {
        assert!(ChiSquared::new(f64::INFINITY).is_err());
        assert!(ChiSquared::new(-f64::NAN).is_err());
        assert!(ChiSquared::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn ln_pdf() {
        let x2 = ChiSquared::new(2.5).unwrap();
        assert::close(x2.ln_pdf(&1.2_f64), -1.322_581_750_079_63, TOL);
        assert::close(x2.ln_pdf(&3.4_f64), -2.162_218_281_372_589_4, TOL);
    }

    #[test]
    fn cdf() {
        let x2 = ChiSquared::new(2.5).unwrap();
        assert::close(x2.cdf(&1.2_f64), 0.338_593_843_799_828_5, TOL);
        assert::close(x2.cdf(&3.4_f64), 0.744_305_104_870_633_3, TOL);
    }

    #[test]
    fn mean() {
        let m: f64 = ChiSquared::new(2.5).unwrap().mean().unwrap();
        assert::close(m, 2.5, TOL);
    }

    #[test]
    fn variance() {
        let v: f64 = ChiSquared::new(2.5).unwrap().variance().unwrap();
        assert::close(v, 5.0, TOL);
    }

    #[test]
    fn kurtosis() {
        let k = ChiSquared::new(2.5).unwrap().kurtosis().unwrap();
        assert::close(k, 4.8, TOL);
    }

    #[test]
    fn skewness() {
        let s = ChiSquared::new(2.5).unwrap().skewness().unwrap();
        assert::close(s, 1.788_854_381_999_831_7, TOL);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let x2 = ChiSquared::new(2.5).unwrap();
        let cdf = |x: f64| x2.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = x2.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });
        assert!(passes > 0);
    }
}
