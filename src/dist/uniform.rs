//! Continuous uniform distribution, U(a, b) on the interval x in [a, b]
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::impl_display;
use crate::result;
use crate::traits::*;
use rand::Rng;
use std::f64;

/// [Continuous uniform distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)),
/// U(a, b) on the interval x in [a, b]
///
/// # Example
///
/// The Uniform CDF is a line
///
/// ```
/// # extern crate rv;
/// use rv::prelude::*;
///
/// let u = Uniform::new(2.0, 4.0).unwrap();
///
/// // A line representing the CDF
/// let y = |x: f64| { 0.5 * x - 1.0 };
///
/// assert!((u.cdf(&3.0_f64) - y(3.0)).abs() < 1E-12);
/// assert!((u.cdf(&3.2_f64) - y(3.2)).abs() < 1E-12);
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Uniform {
    pub a: f64,
    pub b: f64,
}

impl Uniform {
    pub fn new(a: f64, b: f64) -> result::Result<Self> {
        let a_ok = a.is_finite();
        let b_ok = b.is_finite() && b > a;
        if !a_ok {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "a must be finite";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        } else if !b_ok {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "b must be finite and greater than a";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        } else {
            Ok(Uniform { a, b })
        }
    }
}

impl Default for Uniform {
    fn default() -> Self {
        Uniform::new(0.0, 1.0).unwrap()
    }
}

impl From<&Uniform> for String {
    fn from(u: &Uniform) -> String {
        format!("U({}, {})", u.a, u.b)
    }
}

impl_display!(Uniform);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Uniform {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xf = f64::from(*x);
                if self.a <= xf && xf <= self.b {
                    -(self.b - self.a).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let u = rand::distributions::Uniform::new(self.a, self.b);
                rng.sample(u) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let u = rand::distributions::Uniform::new(self.a, self.b);
                (0..n).map(|_| rng.sample(u) as $kind).collect()
            }
        }

        impl Support<$kind> for Uniform {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite()
                    && self.a <= f64::from(*x)
                    && f64::from(*x) <= self.b
            }
        }

        impl ContinuousDistr<$kind> for Uniform {}

        impl Mean<$kind> for Uniform {
            fn mean(&self) -> Option<$kind> {
                let m = (self.b + self.a) / 2.0;
                Some(m as $kind)
            }
        }

        impl Median<$kind> for Uniform {
            fn median(&self) -> Option<$kind> {
                let m = (self.b + self.a) / 2.0;
                Some(m as $kind)
            }
        }

        impl Variance<$kind> for Uniform {
            fn variance(&self) -> Option<$kind> {
                let v = (self.b - self.a).powi(2) / 12.0;
                Some(v as $kind)
            }
        }

        impl Cdf<$kind> for Uniform {
            fn cdf(&self, x: &$kind) -> f64 {
                let xf = f64::from(*x);
                if xf < self.a {
                    0.0
                } else if xf >= self.b {
                    1.0
                } else {
                    (xf - self.a) / (self.b - self.a)
                }
            }
        }

        impl InverseCdf<$kind> for Uniform {
            fn invcdf(&self, p: f64) -> $kind {
                let x = p * (self.b - self.a) + self.a;
                x as $kind
            }
        }
    };
}

impl Skewness for Uniform {
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl Kurtosis for Uniform {
    fn kurtosis(&self) -> Option<f64> {
        Some(-1.2)
    }
}

impl Entropy for Uniform {
    fn entropy(&self) -> f64 {
        (self.b - self.a).ln()
    }
}

impl_traits!(f64);
impl_traits!(f32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    #[test]
    fn new() {
        let u = Uniform::new(0.0, 1.0).unwrap();
        assert::close(u.a, 0.0, TOL);
        assert::close(u.b, 1.0, TOL);
    }

    #[test]
    fn new_rejects_a_equal_to_b() {
        assert!(Uniform::new(1.0, 1.0).is_err());
    }

    #[test]
    fn new_rejects_a_gt_b() {
        assert!(Uniform::new(2.0, 1.0).is_err());
    }

    #[test]
    fn new_rejects_non_finite_a_or_b() {
        assert!(Uniform::new(f64::NEG_INFINITY, 1.0).is_err());
        assert!(Uniform::new(f64::NAN, 1.0).is_err());
        assert!(Uniform::new(0.0, f64::INFINITY).is_err());
        assert!(Uniform::new(0.0, f64::NAN).is_err());
    }

    #[test]
    fn mean() {
        let m: f64 = Uniform::new(2.0, 4.0).unwrap().mean().unwrap();
        assert::close(m, 3.0, TOL);
    }

    #[test]
    fn median() {
        let m: f64 = Uniform::new(2.0, 4.0).unwrap().median().unwrap();
        assert::close(m, 3.0, TOL);
    }

    #[test]
    fn variance() {
        let v: f64 = Uniform::new(2.0, 4.0).unwrap().variance().unwrap();
        assert::close(v, 2.0 / 6.0, TOL);
    }

    #[test]
    fn entropy() {
        let h: f64 = Uniform::new(2.0, 4.0).unwrap().entropy();
        assert::close(h, 0.69314718055994529, TOL);
    }

    #[test]
    fn ln_pdf() {
        let u = Uniform::new(2.0, 4.0).unwrap();
        assert::close(u.ln_pdf(&2.0_f64), -0.69314718055994529, TOL);
        assert::close(u.ln_pdf(&2.3_f64), -0.69314718055994529, TOL);
        assert::close(u.ln_pdf(&3.3_f64), -0.69314718055994529, TOL);
        assert::close(u.ln_pdf(&4.0_f64), -0.69314718055994529, TOL);
    }

    #[test]
    fn cdf() {
        let u = Uniform::new(2.0, 4.0).unwrap();
        assert::close(u.cdf(&2.0_f64), 0.0, TOL);
        assert::close(u.cdf(&2.3_f64), 0.14999999999999991, TOL);
        assert::close(u.cdf(&3.3_f64), 0.64999999999999991, TOL);
        assert::close(u.cdf(&4.0_f64), 1.0, TOL);
    }

    #[test]
    fn cdf_inv_cdf_ident() {
        let mut rng = rand::thread_rng();
        let ru = rand::distributions::Uniform::new(1.2, 3.4);
        let u = Uniform::new(1.2, 3.4).unwrap();
        for _ in 0..100 {
            let x: f64 = rng.sample(ru);
            let cdf = u.cdf(&x);
            let y: f64 = u.invcdf(cdf);
            assert::close(x, y, 1E-8);
        }
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let u = Uniform::new(1.2, 3.4).unwrap();
        let cdf = |x: f64| u.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = u.sample(1000, &mut rng);
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
