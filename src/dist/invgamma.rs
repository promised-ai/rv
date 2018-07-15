//! Inverse Gamma distribution over x in (0, ∞)
extern crate rand;
extern crate special;

use self::rand::distributions;
use self::rand::Rng;
use self::special::Gamma as SGamma;

use traits::*;

/// Inverse gamma distribution IG(α, β)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InvGamma {
    pub shape: f64,
    pub scale: f64,
}

impl InvGamma {
    /// Create a new `Gamma` distribution with shape (α) and rate (β).
    pub fn new(shape: f64, scale: f64) -> Self {
        InvGamma {
            shape: shape,
            scale: scale,
        }
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for InvGamma {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xf = *x as f64;
                self.shape * self.scale.ln()
                    - self.shape.ln_gamma().0
                    - (self.shape + 1.0) * xf.ln()
                    - (self.scale / xf)
            }

            #[inline]
            fn ln_normalizer(&self) -> f64 {
                0.0
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
            fn contains(&self, x: &$kind) -> bool {
                if x.is_finite() && *x > 0.0 {
                    true
                } else {
                    false
                }
            }
        }

        impl Cdf<$kind> for InvGamma {
            fn cdf(&self, x: &$kind) -> f64 {
                1.0 - (self.scale / *x as f64).inc_gamma(self.shape)
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
    extern crate assert;
    use super::*;
    use std::f64;

    const TOL: f64 = 1E-12;

    #[test]
    fn new() {
        let ig = InvGamma::new(1.0, 2.0);
        assert::close(ig.shape, 1.0, TOL);
        assert::close(ig.scale, 2.0, TOL);
    }

    #[test]
    fn mean() {
        let mean: f64 = InvGamma::new(1.2, 3.4).mean().unwrap();
        assert::close(mean, 17.000000000000004, TOL);
    }

    #[test]
    fn mean_undefined_for_shape_leq_1() {
        let m1_opt: Option<f64> = InvGamma::new(1.001, 3.4).mean();
        let m2_opt: Option<f64> = InvGamma::new(1.0, 3.4).mean();
        let m3_opt: Option<f64> = InvGamma::new(0.1, 3.4).mean();
        assert!(m1_opt.is_some());
        assert!(m2_opt.is_none());
        assert!(m3_opt.is_none());
    }

    #[test]
    fn mode() {
        let m1: f64 = InvGamma::new(2.0, 3.0).mode().unwrap();
        let m2: f64 = InvGamma::new(3.0, 2.0).mode().unwrap();
        assert::close(m1, 1.0, TOL);
        assert::close(m2, 0.5, TOL);
    }

    #[test]
    fn variance() {
        let ig = InvGamma::new(2.3, 4.5);
        assert::close(ig.variance().unwrap(), 39.940828402366897, TOL);
    }

    #[test]
    fn variance_undefined_for_shape_leq_2() {
        assert!(InvGamma::new(2.001, 3.4).variance().is_some());
        assert!(InvGamma::new(2.0, 3.4).variance().is_none());
        assert!(InvGamma::new(0.1, 3.4).variance().is_none());
    }

    #[test]
    fn ln_pdf_low_value() {
        let ig = InvGamma::new(3.0, 2.0);
        assert::close(ig.ln_pdf(&0.1_f64), -9.4033652669039274, TOL);
    }

    #[test]
    fn ln_pdf_at_mean() {
        let ig = InvGamma::new(3.0, 2.0);
        assert::close(ig.ln_pdf(&1.0_f64), -0.61370563888010954, TOL);
    }

    #[test]
    fn ln_pdf_at_mode() {
        let ig = InvGamma::new(3.0, 2.0);
        assert::close(ig.ln_pdf(&0.5_f64), 0.15888308335967161, TOL);
    }

    #[test]
    fn ln_pdf_at_mode_should_be_higest() {
        let ig = InvGamma::new(3.0, 2.0);
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
        assert!(!InvGamma::new(1.0, 1.0).contains(&-0.000001_f64));
        assert!(!InvGamma::new(1.0, 1.0).contains(&-1.0_f64));
    }

    #[test]
    fn does_not_contain_zero() {
        assert!(!InvGamma::new(1.0, 1.0).contains(&0.0_f32));
    }

    #[test]
    fn contains_positive_values() {
        assert!(InvGamma::new(1.0, 1.0).contains(&0.000001_f64));
        assert!(InvGamma::new(1.0, 1.0).contains(&1.0_f64));
    }

    #[test]
    fn does_not_contain_infinity() {
        assert!(!InvGamma::new(1.0, 1.0).contains(&f64::INFINITY));
    }

    #[test]
    fn sample_return_correct_number_of_draws() {
        let mut rng = rand::thread_rng();
        let ig = InvGamma::new(3.0, 2.0);
        let xs: Vec<f64> = ig.sample(103, &mut rng);
        assert_eq!(xs.len(), 103);
    }

    #[test]
    fn draw_always_returns_results_in_support() {
        let mut rng = rand::thread_rng();
        let ig = InvGamma::new(3.0, 2.0);
        for _ in 0..100 {
            let x: f64 = ig.draw(&mut rng);
            assert!(x > 0.0 and x.is_finite());
        }
    }

    #[test]
    fn cdf_at_1() {
        let ig = InvGamma::new(1.2, 3.4);
        assert::close(ig.cdf(&1.0), 0.048714368540659622, TOL);
    }

    #[test]
    fn cdf_at_mean() {
        let ig = InvGamma::new(1.2, 3.4);
        assert::close(ig.cdf(&17.0), 0.88185118032427523, TOL);
    }

    #[test]
    fn skewness() {
        let ig = InvGamma::new(5.2, 2.4);
        assert::close(ig.skewness().unwrap(), 3.2524625127269666, TOL);
    }

    #[test]
    fn skewness_undefined_for_alpha_leq_3() {
        assert!(InvGamma::new(3.001, 3.4).skewness().is_some());
        assert!(InvGamma::new(3.0, 3.4).skewness().is_none());
        assert!(InvGamma::new(0.1, 3.4).skewness().is_none());
    }

    #[test]
    fn kurtosis() {
        let ig = InvGamma::new(5.2, 2.4);
        assert::close(ig.kurtosis().unwrap(), 34.090909090909086, TOL);
    }

    #[test]
    fn kurtosis_undefined_for_alpha_leq_4() {
        assert!(InvGamma::new(4.001, 3.4).kurtosis().is_some());
        assert!(InvGamma::new(4.0, 3.4).kurtosis().is_none());
        assert!(InvGamma::new(0.1, 3.4).kurtosis().is_none());
    }
}
