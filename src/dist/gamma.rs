//! Gamma distribution over x in (0, ∞)
extern crate rand;
extern crate special;

use self::rand::distributions;
use self::rand::Rng;
use self::special::Gamma as SGamma;

use traits::*;

/// Gamma distribution G(α, β)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Gamma {
    pub shape: f64,
    pub rate: f64,
}

impl Gamma {
    /// Create a new `Gamma` distribution with shape (α) and rate (β).
    pub fn new(shape: f64, rate: f64) -> Self {
        Gamma { shape, rate }
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Gamma {
            fn ln_f(&self, x: &$kind) -> f64 {
                self.shape * self.rate.ln() - self.shape.ln_gamma().0
                    + (self.shape - 1.0) * f64::from(*x).ln()
                    - (self.rate * f64::from(*x))
            }

            #[inline]
            fn ln_normalizer(&self) -> f64 {
                0.0
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
            fn contains(&self, x: &$kind) -> bool {
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
    extern crate assert;
    use super::*;
    use std::f64;

    const TOL: f64 = 1E-12;

    #[test]
    fn new() {
        let gam = Gamma::new(1.0, 2.0);
        assert::close(gam.shape, 1.0, TOL);
        assert::close(gam.rate, 2.0, TOL);
    }

    #[test]
    fn ln_pdf_low_value() {
        let gam = Gamma::new(1.2, 3.4);
        assert::close(gam.ln_pdf(&0.1_f64), 0.75338758935104555, TOL);
    }

    #[test]
    fn ln_pdf_at_mean() {
        let gam = Gamma::new(1.2, 3.4);
        assert::close(gam.ln_pdf(&100.0_f64), -337.52506135485254, TOL);
    }

    #[test]
    fn cdf() {
        let gam = Gamma::new(1.2, 3.4);
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
        let gam = Gamma::new(1.2, 3.4);
        assert::close(
            gam.ln_pdf(&0.35294117647058826_f64),
            0.14561383298422248,
            TOL,
        );
    }

    #[test]
    fn mean_should_be_ratio_of_params() {
        let m1: f64 = Gamma::new(1.0, 2.0).mean().unwrap();
        let m2: f64 = Gamma::new(1.0, 1.0).mean().unwrap();
        let m3: f64 = Gamma::new(3.0, 1.0).mean().unwrap();
        let m4: f64 = Gamma::new(0.3, 0.1).mean().unwrap();
        assert::close(m1, 0.5, TOL);
        assert::close(m2, 1.0, TOL);
        assert::close(m3, 3.0, TOL);
        assert::close(m4, 3.0, TOL);
    }

    #[test]
    fn mode_undefined_for_shape_less_than_one() {
        let m1_opt: Option<f64> = Gamma::new(1.0, 2.0).mode();
        let m2_opt: Option<f64> = Gamma::new(0.999, 2.0).mode();
        let m3_opt: Option<f64> = Gamma::new(0.5, 2.0).mode();
        let m4_opt: Option<f64> = Gamma::new(0.1, 2.0).mode();
        assert!(m1_opt.is_some());
        assert!(m2_opt.is_none());
        assert!(m3_opt.is_none());
        assert!(m4_opt.is_none());
    }

    #[test]
    fn mode() {
        let m1: f64 = Gamma::new(2.0, 2.0).mode().unwrap();
        let m2: f64 = Gamma::new(1.0, 2.0).mode().unwrap();
        let m3: f64 = Gamma::new(2.0, 1.0).mode().unwrap();
        assert::close(m1, 0.5, TOL);
        assert::close(m2, 0.0, TOL);
        assert::close(m3, 1.0, TOL);
    }

    #[test]
    fn variance() {
        assert::close(Gamma::new(2.0, 2.0).variance().unwrap(), 0.5, TOL);
        assert::close(Gamma::new(0.5, 2.0).variance().unwrap(), 1.0 / 8.0, TOL);
    }

    #[test]
    fn skewness() {
        assert::close(Gamma::new(4.0, 3.0).skewness().unwrap(), 1.0, TOL);
        assert::close(Gamma::new(16.0, 4.0).skewness().unwrap(), 0.5, TOL);
        assert::close(Gamma::new(16.0, 1.0).skewness().unwrap(), 0.5, TOL);
    }

    #[test]
    fn kurtosis() {
        assert::close(Gamma::new(6.0, 3.0).kurtosis().unwrap(), 1.0, TOL);
        assert::close(Gamma::new(6.0, 1.0).kurtosis().unwrap(), 1.0, TOL);
        assert::close(Gamma::new(12.0, 1.0).kurtosis().unwrap(), 0.5, TOL);
    }

    #[test]
    fn entropy() {
        let gam1 = Gamma::new(2.0, 1.0);
        let gam2 = Gamma::new(1.2, 3.4);
        assert::close(gam1.entropy(), 1.5772156649015328, TOL);
        assert::close(gam2.entropy(), -0.05134154230699384, TOL);
    }
}
