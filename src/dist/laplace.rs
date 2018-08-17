//! Laplace (double exponential) distribution
extern crate rand;

use self::rand::Rng;
use std::f64::consts::{E, FRAC_1_SQRT_2, LN_2};
use std::io;
use traits::*;

/// [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution), or double
/// exponential, distribution over x in (-∞, ∞).
///
/// # Example
///
/// ```
/// # extern crate rv;
/// extern crate rand;
///
/// use rv::prelude::*;
///
/// let laplace = Laplace::new(0.0, 1.0).expect("Invalid params");
///
/// // 100 draws from Laplace
/// let mut rng = rand::thread_rng();
/// let xs: Vec<f64> = laplace.sample(100, &mut rng);
/// assert_eq!(xs.len(), 100);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Laplace {
    /// Location in (-∞, ∞)
    mu: f64,
    /// Scale in (0, ∞)
    b: f64,
}

impl Laplace {
    pub fn new(mu: f64, b: f64) -> io::Result<Self> {
        if !mu.is_finite() {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "mu must be finite");
            Err(err)
        } else if b <= 0.0 || !b.is_finite() {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "b must be in (0, ∞)");
            Err(err)
        } else {
            Ok(Laplace { mu, b })
        }
    }
}

/// Laplace with mean 0 and variance 1
impl Default for Laplace {
    fn default() -> Self {
        Laplace::new(0.0, FRAC_1_SQRT_2).unwrap()
    }
}

#[inline]
fn laplace_partial_draw(u: f64) -> f64 {
    let r = u - 0.5;
    r.signum() * (1.0 - 2.0 * r.abs()).ln()
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Laplace {
            fn ln_f(&self, x: &$kind) -> f64 {
                -(f64::from(*x) - self.mu).abs() / self.b - self.b.ln() - LN_2
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let u = rng.sample(rand::distributions::OpenClosed01);
                (self.mu - self.b * laplace_partial_draw(u)) as $kind
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
                    1.0 - 0.5 * (-(xf - self.mu) / self.b).exp()
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
                Some((2.0 * self.b.powi(2)) as $kind)
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

#[cfg(test)]
mod tests {
    extern crate assert;
    use super::*;
    use misc::ks_test;
    use std::f64;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

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
        assert::close(v, 23.119999999999997, TOL);
    }

    #[test]
    fn entropy() {
        let h: f64 = Laplace::new(1.2, 3.4).unwrap().entropy();
        assert::close(h, 2.916922612182061, TOL);
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
        assert::close(cdf, 0.35130926133149776, TOL);
    }

    #[test]
    fn cdf_above_mu() {
        let laplace = Laplace::new(1.2, 3.4).unwrap();
        let cdf = laplace.cdf(&3.0_f64);
        assert::close(cdf, 0.70552434512472328, TOL);
    }

    #[test]
    fn ln_pdf() {
        let laplace = Laplace::new(1.2, 3.4).unwrap();
        assert::close(laplace.ln_pdf(&1.2), -1.9169226121820611, TOL);
        assert::close(laplace.ln_pdf(&0.2), -2.2110402592408844, TOL);
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
}
