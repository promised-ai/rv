//! Inverse Gamma distribution over x in (0, ∞)
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::impl_display;
use crate::result;
use crate::traits::*;
use getset::Setters;
use rand::distributions;
use rand::Rng;
use special::Gamma as _;

/// [Inverse gamma distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
/// IG(α, β) over x in (0, ∞).
///
/// ```math
///             β^α
/// f(x|α, β) = ----  x^(-α-1) e^(-β/x)
///             Γ(α)
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Setters)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct InvGamma {
    // shape parameter, α
    #[set = "pub"]
    shape: f64,
    // scale parameter, β
    #[set = "pub"]
    scale: f64,
}

impl InvGamma {
    /// Create a new `Gamma` distribution with shape (α) and rate (β).
    ///
    /// # Arguments
    /// - shape: shape parameter, α
    /// - scale scale parameter, β
    pub fn new(shape: f64, scale: f64) -> result::Result<Self> {
        let shape_ok = shape > 0.0 && shape.is_finite();
        let scale_ok = scale > 0.0 && scale.is_finite();
        if shape_ok && scale_ok {
            Ok(InvGamma { shape, scale })
        } else {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "shape and scale must be finite and greater than 0";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        }
    }

    /// Get the shape paramter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvGamma;
    /// let ig = InvGamma::new(1.0, 2.0).unwrap();
    /// assert_eq!(ig.shape(), 1.0);
    /// ```
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Get the scale paramter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvGamma;
    /// let ig = InvGamma::new(1.0, 2.0).unwrap();
    /// assert_eq!(ig.scale(), 2.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Default for InvGamma {
    fn default() -> Self {
        InvGamma {
            shape: 1.0,
            scale: 1.0,
        }
    }
}

impl From<&InvGamma> for String {
    fn from(invgam: &InvGamma) -> String {
        format!("IG(α: {}, β: {})", invgam.shape, invgam.scale)
    }
}

impl_display!(InvGamma);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for InvGamma {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xf = f64::from(*x);
                self.shape * self.scale.ln()
                    - self.shape.ln_gamma().0
                    - (self.shape + 1.0) * xf.ln()
                    - (self.scale / xf)
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let g = distributions::Gamma::new(self.shape, 1.0 / self.scale);
                (1.0 / rng.sample(g)) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let g = distributions::Gamma::new(self.shape, 1.0 / self.scale);
                (0..n).map(|_| (1.0 / rng.sample(g)) as $kind).collect()
            }
        }

        impl ContinuousDistr<$kind> for InvGamma {}

        impl Support<$kind> for InvGamma {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite() && *x > 0.0
            }
        }

        impl Cdf<$kind> for InvGamma {
            fn cdf(&self, x: &$kind) -> f64 {
                1.0 - (self.scale / f64::from(*x)).inc_gamma(self.shape)
            }
        }

        impl Mean<$kind> for InvGamma {
            fn mean(&self) -> Option<$kind> {
                if self.shape > 1.0 {
                    Some((self.scale / (self.shape - 1.0)) as $kind)
                } else {
                    None
                }
            }
        }

        impl Mode<$kind> for InvGamma {
            fn mode(&self) -> Option<$kind> {
                Some((self.scale / (self.shape + 1.0)) as $kind)
            }
        }
    };
}

impl Variance<f64> for InvGamma {
    fn variance(&self) -> Option<f64> {
        if self.shape > 2.0 {
            let numer = self.scale.powi(2);
            let denom = (self.shape - 1.0).powi(2) * (self.shape - 2.0);
            Some(numer / denom)
        } else {
            None
        }
    }
}

impl Entropy for InvGamma {
    fn entropy(&self) -> f64 {
        self.shape + self.scale.ln() + self.shape.ln_gamma().0
            - (1.0 + self.shape) * self.shape.digamma()
    }
}

impl Skewness for InvGamma {
    fn skewness(&self) -> Option<f64> {
        if self.shape > 3.0 {
            Some(4.0 * (self.shape - 2.0).sqrt() / (self.shape - 3.0))
        } else {
            None
        }
    }
}

impl Kurtosis for InvGamma {
    fn kurtosis(&self) -> Option<f64> {
        if self.shape > 4.0 {
            let krt = (30.0 * self.shape - 66.0)
                / ((self.shape - 3.0) * (self.shape - 4.0));
            Some(krt)
        } else {
            None
        }
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
        let ig = InvGamma::new(1.0, 2.0).unwrap();
        assert::close(ig.shape, 1.0, TOL);
        assert::close(ig.scale, 2.0, TOL);
    }

    #[test]
    fn mean() {
        let mean: f64 = InvGamma::new(1.2, 3.4).unwrap().mean().unwrap();
        assert::close(mean, 17.000000000000004, TOL);
    }

    #[test]
    fn mean_undefined_for_shape_leq_1() {
        let m1_opt: Option<f64> = InvGamma::new(1.001, 3.4).unwrap().mean();
        let m2_opt: Option<f64> = InvGamma::new(1.0, 3.4).unwrap().mean();
        let m3_opt: Option<f64> = InvGamma::new(0.1, 3.4).unwrap().mean();
        assert!(m1_opt.is_some());
        assert!(m2_opt.is_none());
        assert!(m3_opt.is_none());
    }

    #[test]
    fn mode() {
        let m1: f64 = InvGamma::new(2.0, 3.0).unwrap().mode().unwrap();
        let m2: f64 = InvGamma::new(3.0, 2.0).unwrap().mode().unwrap();
        assert::close(m1, 1.0, TOL);
        assert::close(m2, 0.5, TOL);
    }

    #[test]
    fn variance() {
        let ig = InvGamma::new(2.3, 4.5).unwrap();
        assert::close(ig.variance().unwrap(), 39.940828402366897, TOL);
    }

    #[test]
    fn variance_undefined_for_shape_leq_2() {
        assert!(InvGamma::new(2.001, 3.4).unwrap().variance().is_some());
        assert!(InvGamma::new(2.0, 3.4).unwrap().variance().is_none());
        assert!(InvGamma::new(0.1, 3.4).unwrap().variance().is_none());
    }

    #[test]
    fn ln_pdf_low_value() {
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        assert::close(ig.ln_pdf(&0.1_f64), -9.4033652669039274, TOL);
    }

    #[test]
    fn ln_pdf_at_mean() {
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        assert::close(ig.ln_pdf(&1.0_f64), -0.61370563888010954, TOL);
    }

    #[test]
    fn ln_pdf_at_mode() {
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        assert::close(ig.ln_pdf(&0.5_f64), 0.15888308335967161, TOL);
    }

    #[test]
    fn ln_pdf_at_mode_should_be_higest() {
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        let x: f64 = ig.mode().unwrap();
        let delta = 1E-6;

        let fx = ig.ln_pdf(&x);
        let fa = ig.ln_pdf(&(x - delta));
        let fb = ig.ln_pdf(&(x + delta));

        assert!(fx > fa);
        assert!(fx > fb);
    }

    #[test]
    fn does_not_contain_negative_values() {
        assert!(!InvGamma::new(1.0, 1.0).unwrap().supports(&-0.000001_f64));
        assert!(!InvGamma::new(1.0, 1.0).unwrap().supports(&-1.0_f64));
    }

    #[test]
    fn does_not_contain_zero() {
        assert!(!InvGamma::new(1.0, 1.0).unwrap().supports(&0.0_f32));
    }

    #[test]
    fn contains_positive_values() {
        assert!(InvGamma::new(1.0, 1.0).unwrap().supports(&0.000001_f64));
        assert!(InvGamma::new(1.0, 1.0).unwrap().supports(&1.0_f64));
    }

    #[test]
    fn does_not_contain_infinity() {
        assert!(!InvGamma::new(1.0, 1.0).unwrap().supports(&f64::INFINITY));
    }

    #[test]
    fn sample_return_correct_number_of_draws() {
        let mut rng = rand::thread_rng();
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        let xs: Vec<f64> = ig.sample(103, &mut rng);
        assert_eq!(xs.len(), 103);
    }

    #[test]
    fn draw_always_returns_results_in_support() {
        let mut rng = rand::thread_rng();
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        for _ in 0..100 {
            let x: f64 = ig.draw(&mut rng);
            assert!(x > 0.0 && x.is_finite());
        }
    }

    #[test]
    fn cdf_at_1() {
        let ig = InvGamma::new(1.2, 3.4).unwrap();
        assert::close(ig.cdf(&1.0), 0.048714368540659622, TOL);
    }

    #[test]
    fn cdf_at_mean() {
        let ig = InvGamma::new(1.2, 3.4).unwrap();
        assert::close(ig.cdf(&17.0), 0.88185118032427523, TOL);
    }

    #[test]
    fn skewness() {
        let ig = InvGamma::new(5.2, 2.4).unwrap();
        assert::close(ig.skewness().unwrap(), 3.2524625127269666, TOL);
    }

    #[test]
    fn skewness_undefined_for_alpha_leq_3() {
        assert!(InvGamma::new(3.001, 3.4).unwrap().skewness().is_some());
        assert!(InvGamma::new(3.0, 3.4).unwrap().skewness().is_none());
        assert!(InvGamma::new(0.1, 3.4).unwrap().skewness().is_none());
    }

    #[test]
    fn kurtosis() {
        let ig = InvGamma::new(5.2, 2.4).unwrap();
        assert::close(ig.kurtosis().unwrap(), 34.090909090909086, TOL);
    }

    #[test]
    fn kurtosis_undefined_for_alpha_leq_4() {
        assert!(InvGamma::new(4.001, 3.4).unwrap().kurtosis().is_some());
        assert!(InvGamma::new(4.0, 3.4).unwrap().kurtosis().is_none());
        assert!(InvGamma::new(0.1, 3.4).unwrap().kurtosis().is_none());
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let ig = InvGamma::new(1.2, 3.4).unwrap();
        let cdf = |x: f64| ig.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = ig.sample(1000, &mut rng);
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
