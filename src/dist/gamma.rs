extern crate rand;
extern crate special;

use std::marker::PhantomData;

use self::rand::distributions;
use self::rand::Rng;
use self::special::Gamma as SGamma;

use traits::*;

/// Gamma distribution G(α, β)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Gamma<T> {
    shape: f64,
    rate: f64,
    _phantom: PhantomData<T>,
}

impl<T> Gamma<T> {
    /// Create a new `Gamma` distribution with shape (α) and rate (β).
    pub fn new(shape: f64, rate: f64) -> Self {
        Gamma {
            shape: shape,
            rate: rate,
            _phantom: PhantomData,
        }
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv for Gamma<$kind> {
            type DatumType = $kind;
            fn ln_f(&self, x: &$kind) -> f64 {
                self.shape * self.rate.ln() - self.shape.ln_gamma().0
                    + (self.shape - 1.0) * (*x as f64).ln() - (self.rate * *x as f64)
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

        impl ContinuousDistr for Gamma<$kind> {}

        impl Support for Gamma<$kind> {
            fn contains(&self, x: &$kind) -> bool {
                if x.is_finite() && *x > 0.0 {
                    true
                } else {
                    false
                }
            }
        }

        impl Cdf for Gamma<$kind> {
            fn cdf(&self, x: &$kind) -> f64 {
                (self.rate * (*x as f64)).inc_gamma(self.shape)
            }
        }

        impl Mean<$kind> for Gamma<$kind> {
            fn mean(&self) -> Option<$kind> {
                Some((self.shape / self.rate) as $kind)
            }
        }

        impl Mode for Gamma<$kind> {
            fn mode(&self) -> Option<$kind> {
                if self.shape >= 1.0 {
                    let m = (self.shape - 1.0) / self.rate;
                    Some(m as $kind)
                } else {
                    None
                }
            }
        }

        impl Variance<$kind> for Gamma<$kind> {
            fn variance(&self) -> Option<$kind> {
                Some((self.shape / self.rate.powi(2)) as $kind)
            }
        }

        impl Entropy for Gamma<$kind> {
            fn entropy(&self) -> f64 {
                self.shape - self.rate.ln()
                    + self.shape.ln_gamma().0
                    + (1.0 - self.shape) * self.shape.digamma()
            }
        }

        impl Skewness for Gamma<$kind> {
            fn skewness(&self) -> Option<f64> {
                Some(2.0 / self.shape.sqrt())
            }
        }

        impl Kurtosis for Gamma<$kind> {
            fn kurtosis(&self) -> Option<f64> {
                Some(6.0 / self.shape)
            }
        }
    };
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
        let gam = Gamma::<f64>::new(1.0, 2.0);
        assert::close(gam.shape, 1.0, TOL);
        assert::close(gam.rate, 2.0, TOL);
    }

    #[test]
    fn ln_pdf_low_value() {
        let gam = Gamma::<f64>::new(1.2, 3.4);
        assert::close(gam.ln_pdf(&0.1), 0.75338758935104555, TOL);
    }

    #[test]
    fn ln_pdf_at_mean() {
        let gam = Gamma::<f64>::new(1.2, 3.4);
        assert::close(gam.ln_pdf(&100.0), -337.52506135485254, TOL);
    }

    #[test]
    fn cdf() {
        let gam = Gamma::<f64>::new(1.2, 3.4);
        assert::close(gam.cdf(&0.5), 0.75943654431805463, TOL);
        assert::close(gam.cdf(&0.35294117647058826), 0.62091806552384998, TOL);
        assert::close(gam.cdf(&100.0), 1.0, TOL);
    }

    #[test]
    fn ln_pdf_hight_value() {
        let gam = Gamma::<f64>::new(1.2, 3.4);
        assert::close(gam.ln_pdf(&0.35294117647058826), 0.14561383298422248, TOL);
    }

    #[test]
    fn mean_should_be_ratio_of_params() {
        assert::close(Gamma::<f64>::new(1.0, 2.0).mean().unwrap(), 0.5, TOL);
        assert::close(Gamma::<f64>::new(1.0, 1.0).mean().unwrap(), 1.0, TOL);
        assert::close(Gamma::<f64>::new(3.0, 1.0).mean().unwrap(), 3.0, TOL);
        assert::close(Gamma::<f64>::new(0.3, 0.1).mean().unwrap(), 3.0, TOL);
    }

    #[test]
    fn mode_undefined_for_shape_less_than_one() {
        assert!(Gamma::<f64>::new(1.0, 2.0).mode().is_some());
        assert!(Gamma::<f64>::new(0.9999, 2.0).mode().is_none());
        assert!(Gamma::<f64>::new(0.5, 2.0).mode().is_none());
        assert!(Gamma::<f64>::new(0.1, 2.0).mode().is_none());
    }

    #[test]
    fn mode() {
        assert::close(Gamma::<f64>::new(2.0, 2.0).mode().unwrap(), 0.5, TOL);
        assert::close(Gamma::<f64>::new(1.0, 2.0).mode().unwrap(), 0.0, TOL);
        assert::close(Gamma::<f64>::new(2.0, 1.0).mode().unwrap(), 1.0, TOL);
    }

    #[test]
    fn variance() {
        assert::close(Gamma::<f64>::new(2.0, 2.0).variance().unwrap(), 0.5, TOL);
        assert::close(
            Gamma::<f64>::new(0.5, 2.0).variance().unwrap(),
            1.0 / 8.0,
            TOL,
        );
    }

    #[test]
    fn skewness() {
        assert::close(Gamma::<f64>::new(4.0, 3.0).skewness().unwrap(), 1.0, TOL);
        assert::close(Gamma::<f64>::new(16.0, 4.0).skewness().unwrap(), 0.5, TOL);
        assert::close(Gamma::<f64>::new(16.0, 1.0).skewness().unwrap(), 0.5, TOL);
    }

    #[test]
    fn kurtosis() {
        assert::close(Gamma::<f64>::new(6.0, 3.0).kurtosis().unwrap(), 1.0, TOL);
        assert::close(Gamma::<f64>::new(6.0, 1.0).kurtosis().unwrap(), 1.0, TOL);
        assert::close(Gamma::<f64>::new(12.0, 1.0).kurtosis().unwrap(), 0.5, TOL);
    }

    #[test]
    fn entropy() {
        let gam1 = Gamma::<f64>::new(2.0, 1.0);
        let gam2 = Gamma::<f64>::new(1.2, 3.4);
        assert::close(gam1.entropy(), 1.5772156649015328, TOL);
        assert::close(gam2.entropy(), -0.05134154230699384, TOL);
    }
}
