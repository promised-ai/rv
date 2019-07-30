//! Gamma distribution over x in (0, ∞)
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::impl_display;
use crate::result;
use crate::traits::*;
use getset::Setters;
use rand::distributions;
use rand::Rng;
use special::Gamma as _;

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
#[derive(Debug, Clone, PartialEq, PartialOrd, Setters)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Gamma {
    #[set = "pub"]
    shape: f64,
    #[set = "pub"]
    rate: f64,
}

impl Gamma {
    /// Create a new `Gamma` distribution with shape (α) and rate (β).
    pub fn new(shape: f64, rate: f64) -> result::Result<Self> {
        let shape_ok = shape > 0.0 && shape.is_finite();
        let rate_ok = rate > 0.0 && rate.is_finite();

        if shape_ok && rate_ok {
            Ok(Gamma { shape, rate })
        } else {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "shape and rate must be finite and greater than 0";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        }
    }

    /// Creates a new Gamma without checking whether the parameters are valid.
    pub fn new_unchecked(shape: f64, rate: f64) -> Self {
        Gamma { shape, rate }
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
    pub fn shape(&self) -> f64 {
        self.shape
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
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl Default for Gamma {
    fn default() -> Self {
        Gamma {
            shape: 1.0,
            rate: 1.0,
        }
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
                self.shape * self.rate.ln() - self.shape.ln_gamma().0
                    + (self.shape - 1.0) * f64::from(*x).ln()
                    - (self.rate * f64::from(*x))
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let g = distributions::Gamma::new(self.shape, 1.0 / self.rate);
                rng.sample(g) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let g = distributions::Gamma::new(self.shape, 1.0 / self.rate);
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
        Some(self.shape / self.rate.powi(2))
    }
}

impl Entropy for Gamma {
    fn entropy(&self) -> f64 {
        self.shape - self.rate.ln()
            + self.shape.ln_gamma().0
            + (1.0 - self.shape) * self.shape.digamma()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;
    use std::f64;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    #[test]
    fn new() {
        let gam = Gamma::new(1.0, 2.0).unwrap();
        assert::close(gam.shape, 1.0, TOL);
        assert::close(gam.rate, 2.0, TOL);
    }

    #[test]
    fn ln_pdf_low_value() {
        let gam = Gamma::new(1.2, 3.4).unwrap();
        assert::close(gam.ln_pdf(&0.1_f64), 0.75338758935104555, TOL);
    }

    #[test]
    fn ln_pdf_at_mean() {
        let gam = Gamma::new(1.2, 3.4).unwrap();
        assert::close(gam.ln_pdf(&100.0_f64), -337.52506135485254, TOL);
    }

    #[test]
    fn cdf() {
        let gam = Gamma::new(1.2, 3.4).unwrap();
        assert::close(gam.cdf(&0.5_f32), 0.75943654431805463, TOL);
        assert::close(
            gam.cdf(&0.35294117647058826_f64),
            0.62091806552384998,
            TOL,
        );
        assert::close(gam.cdf(&100.0_f64), 1.0, TOL);
    }

    #[test]
    fn ln_pdf_hight_value() {
        let gam = Gamma::new(1.2, 3.4).unwrap();
        assert::close(
            gam.ln_pdf(&0.35294117647058826_f64),
            0.14561383298422248,
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
        assert::close(gam1.entropy(), 1.5772156649015328, TOL);
        assert::close(gam2.entropy(), -0.05134154230699384, TOL);
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
