//! Laplace (double exponential) distribution
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::traits::*;
use rand::Rng;
use std::f64::consts::{E, FRAC_1_SQRT_2, LN_2};
use std::fmt;

/// [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution), or double
/// exponential, distribution over x in (-∞, ∞).
///
/// # Example
///
/// ```
/// use rv::prelude::*;
///
/// let laplace = Laplace::new(0.0, 1.0).expect("Invalid params");
///
/// // 100 draws from Laplace
/// let mut rng = rand::thread_rng();
/// let xs: Vec<f64> = laplace.sample(100, &mut rng);
/// assert_eq!(xs.len(), 100);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Laplace {
    /// Location in (-∞, ∞)
    mu: f64,
    /// Scale in (0, ∞)
    b: f64,
}

impl Shiftable for Laplace {
    type Output = Laplace;
    type Error = LaplaceError;

    fn shifted(self, dx: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Laplace::new(self.mu() + dx, self.b())
    }

    fn shifted_unchecked(self, dx: f64) -> Self::Output
    where
        Self: Sized,
    {
        Laplace::new_unchecked(self.mu() + dx, self.b())
    }
}

impl Scalable for Laplace {
    type Output = Laplace;
    type Error = LaplaceError;

    fn scaled(self, scale: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Laplace::new(self.mu() * scale, self.b() * scale)
    }

    fn scaled_unchecked(self, scale: f64) -> Self::Output
    where
        Self: Sized,
    {
        Laplace::new_unchecked(self.mu() * scale, self.b() * scale)
    }
}

pub struct LaplaceParameters {
    pub mu: f64,
    pub b: f64,
}

impl Parameterized for Laplace {
    type Parameters = LaplaceParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            mu: self.mu(),
            b: self.b(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.mu, params.b)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum LaplaceError {
    /// The mu parameter is infinite or NaN
    MuNotFinite { mu: f64 },
    /// The b parameter less than or equal to zero
    BTooLow { b: f64 },
    /// The b parameter is infinite or NaN
    BNotFinite { b: f64 },
}

impl Laplace {
    /// Create a new Laplace distribution.
    pub fn new(mu: f64, b: f64) -> Result<Self, LaplaceError> {
        if !mu.is_finite() {
            Err(LaplaceError::MuNotFinite { mu })
        } else if !b.is_finite() {
            Err(LaplaceError::BNotFinite { b })
        } else if b <= 0.0 {
            Err(LaplaceError::BTooLow { b })
        } else {
            Ok(Laplace { mu, b })
        }
    }

    /// Creates a new Laplace without checking whether the parameters are
    /// valid.
    #[inline]
    pub fn new_unchecked(mu: f64, b: f64) -> Self {
        Laplace { mu, b }
    }

    /// Get the mu parameter
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rv::dist::Laplace;
    /// let laplace = Laplace::new(-1.0, 2.0).unwrap();
    /// assert_eq!(laplace.mu(), -1.0);
    /// ```
    #[inline]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Set the value of the mu parameter
    ///
    /// # Example
    /// ```rust
    /// # use rv::dist::Laplace;
    /// let mut laplace = Laplace::new(-1.0, 2.0).unwrap();
    /// assert_eq!(laplace.mu(), -1.0);
    ///
    /// laplace.set_mu(2.3).unwrap();
    /// assert_eq!(laplace.mu(), 2.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Laplace;
    /// # let mut laplace = Laplace::new(-1.0, 2.0).unwrap();
    /// assert!(laplace.set_mu(0.0).is_ok());
    /// assert!(laplace.set_mu(f64::INFINITY).is_err());
    /// assert!(laplace.set_mu(f64::NEG_INFINITY).is_err());
    /// assert!(laplace.set_mu(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_mu(&mut self, mu: f64) -> Result<(), LaplaceError> {
        if mu.is_finite() {
            self.set_mu_unchecked(mu);
            Ok(())
        } else {
            Err(LaplaceError::MuNotFinite { mu })
        }
    }

    /// Set the value of the mu parameter without input validation
    #[inline]
    pub fn set_mu_unchecked(&mut self, mu: f64) {
        self.mu = mu;
    }

    /// Get the b parameter
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rv::dist::Laplace;
    /// let laplace = Laplace::new(-1.0, 2.0).unwrap();
    /// assert_eq!(laplace.b(), 2.0);
    /// ```
    #[inline]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Set the value of the b parameter
    ///
    /// # Example
    /// ```rust
    /// # use rv::dist::Laplace;
    /// let mut laplace = Laplace::new(-1.0, 2.0).unwrap();
    /// assert_eq!(laplace.b(), 2.0);
    ///
    /// laplace.set_b(2.3).unwrap();
    /// assert_eq!(laplace.b(), 2.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Laplace;
    /// # let mut laplace = Laplace::new(-1.0, 2.0).unwrap();
    /// assert!(laplace.set_b(2.3).is_ok());
    /// assert!(laplace.set_b(0.0).is_err());
    /// assert!(laplace.set_b(f64::INFINITY).is_err());
    /// assert!(laplace.set_b(f64::NEG_INFINITY).is_err());
    /// assert!(laplace.set_b(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_b(&mut self, b: f64) -> Result<(), LaplaceError> {
        if b <= 0.0 {
            Err(LaplaceError::BTooLow { b })
        } else if !b.is_finite() {
            Err(LaplaceError::BNotFinite { b })
        } else {
            self.set_b_unchecked(b);
            Ok(())
        }
    }

    /// Set the value of the b parameter without input validation
    #[inline]
    pub fn set_b_unchecked(&mut self, b: f64) {
        self.b = b;
    }
}

/// Laplace with mean 0 and variance 1
impl Default for Laplace {
    fn default() -> Self {
        Laplace::new_unchecked(0.0, FRAC_1_SQRT_2)
    }
}

impl From<&Laplace> for String {
    fn from(laplace: &Laplace) -> String {
        format!("Laplace(μ: {}, b: {})", laplace.mu, laplace.b)
    }
}

impl_display!(Laplace);

#[inline]
fn laplace_partial_draw(u: f64) -> f64 {
    let r = u - 0.5;
    r.signum() * 2.0_f64.mul_add(-r.abs(), 1.0).ln()
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for Laplace {
            fn ln_f(&self, x: &$kind) -> f64 {
                // TODO: could cache ln(b)
                -(f64::from(*x) - self.mu).abs() / self.b - self.b.ln() - LN_2
            }
        }

        impl Sampleable<$kind> for Laplace {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let u = rng.sample(rand_distr::OpenClosed01);
                self.b.mul_add(-laplace_partial_draw(u), self.mu) as $kind
            }
        }

        impl Support<$kind> for Laplace {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite()
            }
        }

        impl ContinuousDistr<$kind> for Laplace {}

        impl Cdf<$kind> for Laplace {
            fn cdf(&self, x: &$kind) -> f64 {
                let xf: f64 = f64::from(*x);
                if xf < self.mu {
                    0.5 * ((xf - self.mu) / self.b).exp()
                } else {
                    0.5_f64.mul_add(-(-(xf - self.mu) / self.b).exp(), 1.0)
                }
            }
        }

        impl Mean<$kind> for Laplace {
            fn mean(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl Median<$kind> for Laplace {
            fn median(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl Mode<$kind> for Laplace {
            fn mode(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl Variance<$kind> for Laplace {
            fn variance(&self) -> Option<$kind> {
                Some((2.0 * self.b * self.b) as $kind)
            }
        }
    };
}

impl Skewness for Laplace {
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl Kurtosis for Laplace {
    fn kurtosis(&self) -> Option<f64> {
        Some(3.0)
    }
}

impl Entropy for Laplace {
    fn entropy(&self) -> f64 {
        (2.0 * self.b * E).ln()
    }
}

impl_traits!(f64);
impl_traits!(f32);

impl std::error::Error for LaplaceError {}

impl fmt::Display for LaplaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MuNotFinite { mu } => write!(f, "non-finite mu: {}", mu),
            Self::BTooLow { b } => {
                write!(f, "b ({}) must be greater than zero", b)
            }
            Self::BNotFinite { b } => write!(f, "non-finite b: {}", b),
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

    test_basic_impls!(f64, Laplace);

    #[test]
    fn new() {
        let laplace = Laplace::new(0.0, 1.0).unwrap();
        assert::close(laplace.mu, 0.0, TOL);
        assert::close(laplace.b, 1.0, TOL);
    }

    #[test]
    fn new_should_reject_non_finite_mu() {
        assert!(Laplace::new(f64::NEG_INFINITY, 1.0).is_err());
        assert!(Laplace::new(f64::INFINITY, 1.0).is_err());
        assert!(Laplace::new(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn new_should_reject_negative_b() {
        assert!(Laplace::new(0.0, 0.0).is_err());
        assert!(Laplace::new(0.0, -1e-12).is_err());
        assert!(Laplace::new(0.0, -1e12).is_err());
    }

    #[test]
    fn new_should_reject_non_finite_b() {
        assert!(Laplace::new(0.0, f64::NAN).is_err());
        assert!(Laplace::new(0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn mean() {
        let m: f64 = Laplace::new(1.2, 3.4).unwrap().mean().unwrap();
        assert::close(m, 1.2, TOL);
    }

    #[test]
    fn median() {
        let m: f64 = Laplace::new(1.2, 3.4).unwrap().median().unwrap();
        assert::close(m, 1.2, TOL);
    }

    #[test]
    fn mode() {
        let m: f64 = Laplace::new(1.2, 3.4).unwrap().mode().unwrap();
        assert::close(m, 1.2, TOL);
    }

    #[test]
    fn variance() {
        let v: f64 = Laplace::new(1.2, 3.4).unwrap().variance().unwrap();
        assert::close(v, 23.119_999_999_999_997, TOL);
    }

    #[test]
    fn entropy() {
        let h: f64 = Laplace::new(1.2, 3.4).unwrap().entropy();
        assert::close(h, 2.916_922_612_182_061, TOL);
    }

    #[test]
    fn skewness() {
        let s: f64 = Laplace::new(1.2, 3.4).unwrap().skewness().unwrap();
        assert::close(s, 0.0, TOL);
    }

    #[test]
    fn kurtosis() {
        let k: f64 = Laplace::new(1.2, 3.4).unwrap().kurtosis().unwrap();
        assert::close(k, 3.0, TOL);
    }

    #[test]
    fn cdf_at_mu() {
        let laplace = Laplace::new(1.2, 3.4).unwrap();
        let cdf = laplace.cdf(&1.2_f64);
        assert::close(cdf, 0.5, TOL);
    }

    #[test]
    fn cdf_below_mu() {
        let laplace = Laplace::new(1.2, 3.4).unwrap();
        let cdf = laplace.cdf(&0.0_f64);
        assert::close(cdf, 0.351_309_261_331_497_76, TOL);
    }

    #[test]
    fn cdf_above_mu() {
        let laplace = Laplace::new(1.2, 3.4).unwrap();
        let cdf = laplace.cdf(&3.0_f64);
        assert::close(cdf, 0.705_524_345_124_723_3, TOL);
    }

    #[test]
    fn ln_pdf() {
        let laplace = Laplace::new(1.2, 3.4).unwrap();
        assert::close(laplace.ln_pdf(&1.2), -1.916_922_612_182_061, TOL);
        assert::close(laplace.ln_pdf(&0.2), -2.211_040_259_240_884_4, TOL);
    }

    #[test]
    fn draw_test() {
        // Since we've had to implement the laplace draw ourselves, we have to
        // make sure the thing works, so we use the Kolmogorov-Smirnov test.
        let mut rng = rand::thread_rng();
        let laplace = Laplace::new(1.2, 3.4).unwrap();
        let cdf = |x: f64| laplace.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = laplace.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });
        assert!(passes > 0);
    }
    use crate::test_shiftable_cdf;
    use crate::test_shiftable_density;
    use crate::test_shiftable_entropy;
    use crate::test_shiftable_method;

    test_shiftable_method!(Laplace::new(2.0, 1.0).unwrap(), mean);
    test_shiftable_method!(Laplace::new(2.0, 1.0).unwrap(), median);
    test_shiftable_method!(Laplace::new(2.0, 1.0).unwrap(), mode);
    test_shiftable_method!(Laplace::new(2.0, 1.0).unwrap(), variance);
    test_shiftable_method!(Laplace::new(2.0, 1.0).unwrap(), skewness);
    test_shiftable_method!(Laplace::new(2.0, 1.0).unwrap(), kurtosis);
    test_shiftable_density!(Laplace::new(2.0, 1.0).unwrap());
    test_shiftable_entropy!(Laplace::new(2.0, 1.0).unwrap());
    test_shiftable_cdf!(Laplace::new(2.0, 1.0).unwrap());

    use crate::test_scalable_cdf;
    use crate::test_scalable_density;
    use crate::test_scalable_entropy;
    use crate::test_scalable_method;

    test_scalable_method!(Laplace::new(2.0, 1.0).unwrap(), mean);
    test_scalable_method!(Laplace::new(2.0, 1.0).unwrap(), median);
    test_scalable_method!(Laplace::new(2.0, 1.0).unwrap(), mode);
    test_scalable_method!(Laplace::new(2.0, 1.0).unwrap(), variance);
    test_scalable_method!(Laplace::new(2.0, 1.0).unwrap(), skewness);
    test_scalable_method!(Laplace::new(2.0, 1.0).unwrap(), kurtosis);
    test_scalable_density!(Laplace::new(2.0, 1.0).unwrap());
    test_scalable_entropy!(Laplace::new(2.0, 1.0).unwrap());
    test_scalable_cdf!(Laplace::new(2.0, 1.0).unwrap());
}
