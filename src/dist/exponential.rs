//! Exponential distribution over x in [0, ∞)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::traits::*;
use rand::Rng;
use rand_distr::Exp;
use std::f64;
use std::f64::consts::LN_2;
use std::fmt;

/// [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution),
/// Exp(λ) over x in [0, ∞).
///
/// # Examples
///
/// Compute 50% confidence interval
///
/// ```rust
/// use rv::prelude::*;
///
/// let expon = Exponential::new(1.5).unwrap();
/// let interval: (f64, f64) = expon.interval(0.5);  // (0.19, 0.92)
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Exponential {
    /// λ > 0, rate or inverse scale
    rate: f64,
}

crate::impl_shiftable!(Exponential);

impl Scalable for Exponential {
    type Output = Exponential;
    type Error = ExponentialError;

    fn scaled(self, scale: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Exponential::new(self.rate() / scale)
    }

    fn scaled_unchecked(self, scale: f64) -> Self::Output
    where
        Self: Sized,
    {
        Exponential::new_unchecked(self.rate() / scale)
    }
}

impl Default for Exponential {
    fn default() -> Self {
        Self::new_unchecked(1.0)
    }
}

impl Parameterized for Exponential {
    type Parameters = f64;

    fn emit_params(&self) -> Self::Parameters {
        self.rate()
    }

    fn from_params(rate: Self::Parameters) -> Self {
        Self::new_unchecked(rate)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum ExponentialError {
    /// rate parameter is less than or equal to zero
    RateTooLow { rate: f64 },
    /// rate parameter is infinite or zero
    RateNotFinite { rate: f64 },
}

impl Exponential {
    /// Create a new exponential distribution
    ///
    /// # Arguments
    /// - rate: λ > 0, rate or inverse scale
    #[inline]
    pub fn new(rate: f64) -> Result<Self, ExponentialError> {
        if rate <= 0.0 {
            Err(ExponentialError::RateTooLow { rate })
        } else if !rate.is_finite() {
            Err(ExponentialError::RateNotFinite { rate })
        } else {
            Ok(Exponential { rate })
        }
    }

    /// Creates a new Exponential without checking whether the parameter is
    /// valid.
    #[inline]
    pub fn new_unchecked(rate: f64) -> Self {
        Exponential { rate }
    }

    /// Get the rate parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Exponential;
    /// let expon = Exponential::new(1.3).unwrap();
    /// assert_eq!(expon.rate(), 1.3);
    /// ```
    #[inline]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Set the rate parameter
    ///
    /// # Example
    /// ```rust
    /// # use rv::dist::Exponential;
    /// let mut expon = Exponential::new(1.3).unwrap();
    /// assert_eq!(expon.rate(), 1.3);
    ///
    /// expon.set_rate(2.1).unwrap();
    /// assert_eq!(expon.rate(), 2.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Exponential;
    /// # let mut expon = Exponential::new(1.3).unwrap();
    /// assert!(expon.set_rate(2.1).is_ok());
    /// assert!(expon.set_rate(0.1).is_ok());
    /// assert!(expon.set_rate(0.0).is_err());
    /// assert!(expon.set_rate(-1.0).is_err());
    /// assert!(expon.set_rate(f64::INFINITY).is_err());
    /// assert!(expon.set_rate(f64::NEG_INFINITY).is_err());
    /// assert!(expon.set_rate(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_rate(&mut self, rate: f64) -> Result<(), ExponentialError> {
        if rate <= 0.0 {
            Err(ExponentialError::RateTooLow { rate })
        } else if !rate.is_finite() {
            Err(ExponentialError::RateNotFinite { rate })
        } else {
            self.set_rate_unchecked(rate);
            Ok(())
        }
    }

    /// Set the rate parameter without input validation
    #[inline]
    pub fn set_rate_unchecked(&mut self, rate: f64) {
        self.rate = rate;
    }
}

impl From<&Exponential> for String {
    fn from(expon: &Exponential) -> String {
        format!("Expon(λ: {})", expon.rate)
    }
}

impl_display!(Exponential);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for Exponential {
            fn ln_f(&self, x: &$kind) -> f64 {
                // TODO: could cache ln(rate)
                if x < &0.0 {
                    f64::NEG_INFINITY
                } else {
                    self.rate.mul_add(-f64::from(*x), self.rate.ln())
                }
            }
        }

        impl Sampleable<$kind> for Exponential {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let expdist = Exp::new(self.rate).unwrap();
                rng.sample(expdist) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let expdist = Exp::new(self.rate).unwrap();
                (0..n).map(|_| rng.sample(expdist) as $kind).collect()
            }
        }

        impl Support<$kind> for Exponential {
            fn supports(&self, x: &$kind) -> bool {
                *x >= 0.0 && x.is_finite()
            }
        }

        impl ContinuousDistr<$kind> for Exponential {}

        impl Cdf<$kind> for Exponential {
            fn cdf(&self, x: &$kind) -> f64 {
                1.0 - (-self.rate * f64::from(*x)).exp()
            }
        }

        impl InverseCdf<$kind> for Exponential {
            fn invcdf(&self, p: f64) -> $kind {
                let x = -(1.0 - p).ln() / self.rate;
                x as $kind
            }
        }

        impl Mean<$kind> for Exponential {
            fn mean(&self) -> Option<$kind> {
                Some(self.rate.recip() as $kind)
            }
        }

        impl Median<$kind> for Exponential {
            fn median(&self) -> Option<$kind> {
                Some((LN_2 / self.rate) as $kind)
            }
        }

        impl Mode<$kind> for Exponential {
            fn mode(&self) -> Option<$kind> {
                Some(0.0)
            }
        }

        impl Variance<$kind> for Exponential {
            fn variance(&self) -> Option<$kind> {
                let std = self.rate.recip();
                Some((std * std) as $kind)
            }
        }
    };
}

impl Skewness for Exponential {
    fn skewness(&self) -> Option<f64> {
        Some(2.0)
    }
}

impl Kurtosis for Exponential {
    fn kurtosis(&self) -> Option<f64> {
        Some(6.0)
    }
}

impl Entropy for Exponential {
    fn entropy(&self) -> f64 {
        1.0 - self.rate.ln()
    }
}

impl KlDivergence for Exponential {
    fn kl(&self, other: &Self) -> f64 {
        self.rate.ln() - other.rate.ln() + self.rate / other.rate - 1.0
    }
}

impl_traits!(f64);
impl_traits!(f32);

impl std::error::Error for ExponentialError {}

impl fmt::Display for ExponentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RateTooLow { rate } => {
                write!(f, "rate ({}) must be greater than zero", rate)
            }
            Self::RateNotFinite { rate } => {
                write!(f, "non-finite rate: {}", rate)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!(f64, Exponential);

    #[test]
    fn new() {
        let expon = Exponential::new(1.5).unwrap();
        assert::close(expon.rate, 1.5, TOL);
    }

    #[test]
    fn new_should_reject_non_finite_rate() {
        assert!(Exponential::new(1.5).is_ok());
        assert!(Exponential::new(f64::NAN).is_err());
        assert!(Exponential::new(f64::INFINITY).is_err());
    }

    #[test]
    fn new_should_reject_leq_0_rate() {
        assert!(Exponential::new(f64::MIN_POSITIVE).is_ok());
        assert!(Exponential::new(0.0).is_err());
        assert!(Exponential::new(-f64::MIN_POSITIVE).is_err());
    }

    #[test]
    fn ln_f() {
        let expon = Exponential::new_unchecked(1.5);
        assert::close(expon.ln_f(&1.2_f64), -1.394_534_891_891_835_7, TOL);
        assert::close(expon.ln_f(&0.2_f64), 0.105_465_108_108_164_4, TOL);
        assert::close(expon.ln_f(&4.4_f64), -6.194_534_891_891_836, TOL);
        assert_eq!(expon.ln_f(&-1.0_f64), f64::NEG_INFINITY);
    }

    #[test]
    fn ln_pdf() {
        let expon = Exponential::new(1.5).unwrap();
        assert::close(expon.ln_pdf(&1.2_f64), -1.394_534_891_891_835_7, TOL);
        assert::close(expon.ln_pdf(&0.2_f64), 0.105_465_108_108_164_4, TOL);
        assert::close(expon.ln_pdf(&4.4_f64), -6.194_534_891_891_836, TOL);
    }

    #[test]
    fn cdf() {
        let expon = Exponential::new(1.5).unwrap();
        assert::close(expon.cdf(&1.2_f64), 0.834_701_111_778_413_4, TOL);
        assert::close(expon.cdf(&0.2_f64), 0.259_181_779_318_282_2, TOL);
        assert::close(expon.cdf(&4.4_f64), 0.998_639_631_962_452_1, TOL);
    }

    #[test]
    fn mean() {
        let m: f64 = Exponential::new(1.5).unwrap().mean().unwrap();
        assert::close(m, 0.666_666_666_666_666_6, TOL);
    }

    #[test]
    fn median() {
        let m: f64 = Exponential::new(1.5).unwrap().median().unwrap();
        assert::close(m, 0.462_098_120_373_296_84, TOL);
    }

    #[test]
    fn mode() {
        let m: f64 = Exponential::new(1.5).unwrap().mode().unwrap();
        assert::close(m, 0.0, TOL);
    }

    #[test]
    fn variance() {
        let v: f64 = Exponential::new(1.5).unwrap().variance().unwrap();
        assert::close(v, 0.444_444_444_444_444_4, TOL);
    }

    #[test]
    fn skewness() {
        let s = Exponential::new(1.5).unwrap().skewness().unwrap();
        assert::close(s, 2.0, TOL);
    }

    #[test]
    fn kurtosis() {
        let k = Exponential::new(1.5).unwrap().kurtosis().unwrap();
        assert::close(k, 6.0, TOL);
    }

    #[test]
    fn entropy() {
        let h = Exponential::new(1.5).unwrap().entropy();
        assert::close(h, 0.594_534_891_891_835_6, TOL);
    }

    #[test]
    fn quantile() {
        let expon = Exponential::new(1.5).unwrap();
        let q25: f64 = expon.quantile(0.25);
        let q75: f64 = expon.quantile(0.75);
        assert::close(q25, 0.191_788_048_301_187_26, TOL);
        assert::close(q75, 0.924_196_240_746_593_7, TOL);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let expon = Exponential::new(1.5).unwrap();
        let cdf = |x: f64| expon.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = expon.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });
        assert!(passes > 0);
    }

    use crate::test_scalable_cdf;
    use crate::test_scalable_density;
    use crate::test_scalable_entropy;
    use crate::test_scalable_invcdf;
    use crate::test_scalable_method;

    test_scalable_method!(Exponential::new(2.0).unwrap(), mean);
    test_scalable_method!(Exponential::new(2.0).unwrap(), median);
    test_scalable_method!(Exponential::new(2.0).unwrap(), variance);
    test_scalable_method!(Exponential::new(2.0).unwrap(), skewness);
    test_scalable_method!(Exponential::new(2.0).unwrap(), kurtosis);
    test_scalable_density!(Exponential::new(2.0).unwrap());
    test_scalable_entropy!(Exponential::new(2.0).unwrap());
    test_scalable_cdf!(Exponential::new(2.0).unwrap());
    test_scalable_invcdf!(Exponential::new(2.0).unwrap());
}
