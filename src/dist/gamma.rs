//! Gamma distribution over x in (0, ∞)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::traits::*;
use rand::Rng;
use special::Gamma as _;
use std::cell::OnceCell;
use std::fmt;

mod poisson_prior;

/// [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) G(α, β)
/// over x in (0, ∞).
///
/// **NOTE**: The gamma distribution is parameterized in terms of shape, α, and
/// rate, β.
///
/// ```math
///             β^α
/// f(x|α, β) = ----  x^(α-1) e^(-βx)
///             Γ(α)
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Gamma {
    shape: f64,
    rate: f64,
    // ln(gamma(shape))
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_gamma_shape: OnceCell<f64>,
    // ln(rate)
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_rate: OnceCell<f64>,
}

impl PartialEq for Gamma {
    fn eq(&self, other: &Gamma) -> bool {
        self.shape == other.shape && self.rate == other.rate
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum GammaError {
    /// Shape parameter is less than or equal to zero
    ShapeTooLow { shape: f64 },
    /// Shape parameter is infinite or NaN
    ShapeNotFinite { shape: f64 },
    /// Rate parameter is less than or equal to zero
    RateTooLow { rate: f64 },
    /// Rate parameter is infinite or NaN
    RateNotFinite { rate: f64 },
}

impl Gamma {
    /// Create a new `Gamma` distribution with shape (α) and rate (β).
    pub fn new(shape: f64, rate: f64) -> Result<Self, GammaError> {
        if shape <= 0.0 {
            Err(GammaError::ShapeTooLow { shape })
        } else if rate <= 0.0 {
            Err(GammaError::RateTooLow { rate })
        } else if !shape.is_finite() {
            Err(GammaError::ShapeNotFinite { shape })
        } else if !rate.is_finite() {
            Err(GammaError::RateNotFinite { rate })
        } else {
            Ok(Gamma::new_unchecked(shape, rate))
        }
    }

    /// Creates a new Gamma without checking whether the parameters are valid.
    #[inline]
    pub fn new_unchecked(shape: f64, rate: f64) -> Self {
        Gamma {
            shape,
            rate,
            ln_gamma_shape: OnceCell::new(),
            ln_rate: OnceCell::new(),
        }
    }

    /// Get ln(rate)
    #[inline]
    fn ln_rate(&self) -> f64 {
        *self.ln_rate.get_or_init(|| self.rate.ln())
    }

    /// Get ln(gamma(rate))
    #[inline]
    fn ln_gamma_shape(&self) -> f64 {
        *self.ln_gamma_shape.get_or_init(|| self.shape.ln_gamma().0)
    }

    /// Get the shape parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gamma;
    /// let gam = Gamma::new(2.0, 1.0).unwrap();
    /// assert_eq!(gam.shape(), 2.0);
    /// ```
    #[inline]
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Set the shape parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gamma;
    /// let mut gam = Gamma::new(2.0, 1.0).unwrap();
    /// assert_eq!(gam.shape(), 2.0);
    ///
    /// gam.set_shape(1.1).unwrap();
    /// assert_eq!(gam.shape(), 1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Gamma;
    /// # let mut gam = Gamma::new(2.0, 1.0).unwrap();
    /// assert!(gam.set_shape(1.1).is_ok());
    /// assert!(gam.set_shape(0.0).is_err());
    /// assert!(gam.set_shape(-1.0).is_err());
    /// assert!(gam.set_shape(std::f64::INFINITY).is_err());
    /// assert!(gam.set_shape(std::f64::NEG_INFINITY).is_err());
    /// assert!(gam.set_shape(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_shape(&mut self, shape: f64) -> Result<(), GammaError> {
        if shape <= 0.0 {
            Err(GammaError::ShapeTooLow { shape })
        } else if !shape.is_finite() {
            Err(GammaError::ShapeNotFinite { shape })
        } else {
            self.set_shape_unchecked(shape);
            Ok(())
        }
    }

    /// Set the shape parameter without input validation
    #[inline]
    pub fn set_shape_unchecked(&mut self, shape: f64) {
        self.shape = shape;
        self.ln_gamma_shape = OnceCell::new();
    }

    /// Get the rate parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gamma;
    /// let gam = Gamma::new(2.0, 1.0).unwrap();
    /// assert_eq!(gam.rate(), 1.0);
    /// ```
    #[inline]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Set the rate parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gamma;
    /// let mut gam = Gamma::new(2.0, 1.0).unwrap();
    /// assert_eq!(gam.rate(), 1.0);
    ///
    /// gam.set_rate(1.1).unwrap();
    /// assert_eq!(gam.rate(), 1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Gamma;
    /// # let mut gam = Gamma::new(2.0, 1.0).unwrap();
    /// assert!(gam.set_rate(1.1).is_ok());
    /// assert!(gam.set_rate(0.0).is_err());
    /// assert!(gam.set_rate(-1.0).is_err());
    /// assert!(gam.set_rate(std::f64::INFINITY).is_err());
    /// assert!(gam.set_rate(std::f64::NEG_INFINITY).is_err());
    /// assert!(gam.set_rate(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_rate(&mut self, rate: f64) -> Result<(), GammaError> {
        if rate <= 0.0 {
            Err(GammaError::RateTooLow { rate })
        } else if !rate.is_finite() {
            Err(GammaError::RateNotFinite { rate })
        } else {
            self.set_rate_unchecked(rate);
            Ok(())
        }
    }

    /// Set the rate parameter without input validation
    #[inline]
    pub fn set_rate_unchecked(&mut self, rate: f64) {
        self.rate = rate;
        self.ln_rate = OnceCell::new();
    }
}

impl Default for Gamma {
    fn default() -> Self {
        Gamma::new_unchecked(1.0, 1.0)
    }
}

impl From<&Gamma> for String {
    fn from(gam: &Gamma) -> String {
        format!("G(α: {}, β: {})", gam.shape, gam.rate)
    }
}

impl_display!(Gamma);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Gamma {
            fn ln_f(&self, x: &$kind) -> f64 {
                self.shape.mul_add(self.ln_rate(), -self.ln_gamma_shape())
                    + (self.shape - 1.0).mul_add(
                        f64::from(*x).ln(),
                        -(self.rate * f64::from(*x)),
                    )
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let g = rand_distr::Gamma::new(self.shape, 1.0 / self.rate)
                    .unwrap();
                rng.sample(g) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let g = rand_distr::Gamma::new(self.shape, 1.0 / self.rate)
                    .unwrap();
                (0..n).map(|_| rng.sample(g) as $kind).collect()
            }
        }

        impl ContinuousDistr<$kind> for Gamma {}

        impl Support<$kind> for Gamma {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite() && *x > 0.0
            }
        }

        impl Cdf<$kind> for Gamma {
            fn cdf(&self, x: &$kind) -> f64 {
                (self.rate * f64::from(*x)).inc_gamma(self.shape)
            }
        }

        impl Mean<$kind> for Gamma {
            fn mean(&self) -> Option<$kind> {
                Some((self.shape / self.rate) as $kind)
            }
        }

        impl Mode<$kind> for Gamma {
            fn mode(&self) -> Option<$kind> {
                if self.shape >= 1.0 {
                    let m = (self.shape - 1.0) / self.rate;
                    Some(m as $kind)
                } else {
                    None
                }
            }
        }
    };
}

impl Variance<f64> for Gamma {
    fn variance(&self) -> Option<f64> {
        Some(self.shape / (self.rate * self.rate))
    }
}

impl Entropy for Gamma {
    fn entropy(&self) -> f64 {
        self.shape - self.ln_rate()
            + (1.0 - self.shape)
                .mul_add(self.shape.digamma(), self.ln_gamma_shape())
    }
}

impl Skewness for Gamma {
    fn skewness(&self) -> Option<f64> {
        Some(2.0 / self.shape.sqrt())
    }
}

impl Kurtosis for Gamma {
    fn kurtosis(&self) -> Option<f64> {
        Some(6.0 / self.shape)
    }
}

impl_traits!(f32);
impl_traits!(f64);

impl std::error::Error for GammaError {}

impl fmt::Display for GammaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeTooLow { shape } => {
                write!(f, "rate ({}) must be greater than zero", shape)
            }
            Self::ShapeNotFinite { shape } => {
                write!(f, "non-finite rate: {}", shape)
            }
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
    use std::f64;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!([continuous] Gamma::default());

    #[test]
    fn new() {
        let gam = Gamma::new(1.0, 2.0).unwrap();
        assert::close(gam.shape, 1.0, TOL);
        assert::close(gam.rate, 2.0, TOL);
    }

    #[test]
    fn ln_pdf_low_value() {
        let gam = Gamma::new(1.2, 3.4).unwrap();
        assert::close(gam.ln_pdf(&0.1_f64), 0.753_387_589_351_045_6, TOL);
    }

    #[test]
    fn ln_pdf_at_mean() {
        let gam = Gamma::new(1.2, 3.4).unwrap();
        assert::close(gam.ln_pdf(&100.0_f64), -337.525_061_354_852_54, TOL);
    }

    #[test]
    fn cdf() {
        let gam = Gamma::new(1.2, 3.4).unwrap();
        assert::close(gam.cdf(&0.5_f32), 0.759_436_544_318_054_6, TOL);
        assert::close(
            gam.cdf(&0.352_941_176_470_588_26_f64),
            0.620_918_065_523_85,
            TOL,
        );
        assert::close(gam.cdf(&100.0_f64), 1.0, TOL);
    }

    #[test]
    fn ln_pdf_hight_value() {
        let gam = Gamma::new(1.2, 3.4).unwrap();
        assert::close(
            gam.ln_pdf(&0.352_941_176_470_588_26_f64),
            0.145_613_832_984_222_48,
            TOL,
        );
    }

    #[test]
    fn mean_should_be_ratio_of_params() {
        let m1: f64 = Gamma::new(1.0, 2.0).unwrap().mean().unwrap();
        let m2: f64 = Gamma::new(1.0, 1.0).unwrap().mean().unwrap();
        let m3: f64 = Gamma::new(3.0, 1.0).unwrap().mean().unwrap();
        let m4: f64 = Gamma::new(0.3, 0.1).unwrap().mean().unwrap();
        assert::close(m1, 0.5, TOL);
        assert::close(m2, 1.0, TOL);
        assert::close(m3, 3.0, TOL);
        assert::close(m4, 3.0, TOL);
    }

    #[test]
    fn mode_undefined_for_shape_less_than_one() {
        let m1_opt: Option<f64> = Gamma::new(1.0, 2.0).unwrap().mode();
        let m2_opt: Option<f64> = Gamma::new(0.999, 2.0).unwrap().mode();
        let m3_opt: Option<f64> = Gamma::new(0.5, 2.0).unwrap().mode();
        let m4_opt: Option<f64> = Gamma::new(0.1, 2.0).unwrap().mode();
        assert!(m1_opt.is_some());
        assert!(m2_opt.is_none());
        assert!(m3_opt.is_none());
        assert!(m4_opt.is_none());
    }

    #[test]
    fn mode() {
        let m1: f64 = Gamma::new(2.0, 2.0).unwrap().mode().unwrap();
        let m2: f64 = Gamma::new(1.0, 2.0).unwrap().mode().unwrap();
        let m3: f64 = Gamma::new(2.0, 1.0).unwrap().mode().unwrap();
        assert::close(m1, 0.5, TOL);
        assert::close(m2, 0.0, TOL);
        assert::close(m3, 1.0, TOL);
    }

    #[test]
    fn variance() {
        assert::close(
            Gamma::new(2.0, 2.0).unwrap().variance().unwrap(),
            0.5,
            TOL,
        );
        assert::close(
            Gamma::new(0.5, 2.0).unwrap().variance().unwrap(),
            1.0 / 8.0,
            TOL,
        );
    }

    #[test]
    fn skewness() {
        assert::close(
            Gamma::new(4.0, 3.0).unwrap().skewness().unwrap(),
            1.0,
            TOL,
        );
        assert::close(
            Gamma::new(16.0, 4.0).unwrap().skewness().unwrap(),
            0.5,
            TOL,
        );
        assert::close(
            Gamma::new(16.0, 1.0).unwrap().skewness().unwrap(),
            0.5,
            TOL,
        );
    }

    #[test]
    fn kurtosis() {
        assert::close(
            Gamma::new(6.0, 3.0).unwrap().kurtosis().unwrap(),
            1.0,
            TOL,
        );
        assert::close(
            Gamma::new(6.0, 1.0).unwrap().kurtosis().unwrap(),
            1.0,
            TOL,
        );
        assert::close(
            Gamma::new(12.0, 1.0).unwrap().kurtosis().unwrap(),
            0.5,
            TOL,
        );
    }

    #[test]
    fn entropy() {
        let gam1 = Gamma::new(2.0, 1.0).unwrap();
        let gam2 = Gamma::new(1.2, 3.4).unwrap();
        assert::close(gam1.entropy(), 1.577_215_664_901_532_8, TOL);
        assert::close(gam2.entropy(), -0.051_341_542_306_993_84, TOL);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let gam = Gamma::new(1.2, 3.4).unwrap();
        let cdf = |x: f64| gam.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = gam.sample(1000, &mut rng);
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
