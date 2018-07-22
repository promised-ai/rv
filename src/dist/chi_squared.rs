//! Χ</sup>2</sup> over x in (0, ∞)
extern crate rand;
extern crate special;

use self::rand::distributions;
use self::rand::Rng;
use self::special::Gamma as SGamma;
use std::f64::consts::LN_2;
use std::io;

use traits::*;

/// Χ</sup>2</sup> distribution G(α, β)
///
/// # Example
///
/// ```
/// # extern crate rv;
/// use rv::prelude::*;
///
/// let x2 = ChiSquared::new(2.0).unwrap();
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChiSquared {
    /// Degrees of freedom in (0, ∞)
    pub k: f64,
}

impl ChiSquared {
    pub fn new(k: f64) -> io::Result<Self> {
        if k > 0.0 && k.is_finite() {
            Ok(ChiSquared { k })
        } else {
            let err_kind = io::ErrorKind::InvalidInput;
            let msg = "k must be finite and greater than 0";
            let err = io::Error::new(err_kind, msg);
            Err(err)
        }
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for ChiSquared {
            fn ln_f(&self, x: &$kind) -> f64 {
                let k2 = self.k / 2.0;
                let xf = f64::from(*x);
                (k2 - 1.0) * xf.ln() - xf / 2.0 - k2 * LN_2 - k2.ln_gamma().0
            }

            #[inline]
            fn ln_normalizer(&self) -> f64 {
                0.0
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let x2 = distributions::ChiSquared::new(self.k);
                rng.sample(x2) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let x2 = distributions::ChiSquared::new(self.k);
                (0..n).map(|_| rng.sample(x2) as $kind).collect()
            }
        }

        impl Support<$kind> for ChiSquared {
            fn contains(&self, x: &$kind) -> bool {
                *x > 0.0 && x.is_finite()
            }
        }

        impl ContinuousDistr<$kind> for ChiSquared {}

        impl Mean<$kind> for ChiSquared {
            fn mean(&self) -> Option<$kind> {
                Some(self.k as $kind)
            }
        }

        impl Mode<$kind> for ChiSquared {
            fn mode(&self) -> Option<$kind> {
                Some(0.0f64.max(self.k - 2.0) as $kind)
            }
        }

        impl Variance<$kind> for ChiSquared {
            fn variance(&self) -> Option<$kind> {
                Some((self.k * 2.0) as $kind)
            }
        }

        impl Cdf<$kind> for ChiSquared {
            fn cdf(&self, x: &$kind) -> f64 {
                (f64::from(*x) / 2.0).inc_gamma(self.k / 2.0)
            }
        }
    };
}

impl Skewness for ChiSquared {
    fn skewness(&self) -> Option<f64> {
        Some((8.0 / self.k).sqrt())
    }
}

impl Kurtosis for ChiSquared {
    fn kurtosis(&self) -> Option<f64> {
        Some(12.0 / self.k)
    }
}

impl_traits!(f64);
impl_traits!(f32);

#[cfg(test)]
mod tests {
    extern crate assert;
    use super::*;
    use std::f64;

    const TOL: f64 = 1E-12;

    #[test]
    fn new() {
        let x2 = ChiSquared::new(3.2).unwrap();
        assert::close(x2.k, 3.2, TOL);
    }

    #[test]
    fn new_should_reject_k_leq_zero() {
        assert!(ChiSquared::new(f64::MIN_POSITIVE).is_ok());
        assert!(ChiSquared::new(0.0).is_err());
        assert!(ChiSquared::new(-f64::MIN_POSITIVE).is_err());
        assert!(ChiSquared::new(-1.0).is_err());
    }

    #[test]
    fn new_should_reject_non_finite_k() {
        assert!(ChiSquared::new(f64::INFINITY).is_err());
        assert!(ChiSquared::new(-f64::NAN).is_err());
        assert!(ChiSquared::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn ln_pdf() {
        let x2 = ChiSquared::new(2.5).unwrap();
        assert::close(x2.ln_pdf(&1.2_f64), -1.32258175007963, TOL);
        assert::close(x2.ln_pdf(&3.4_f64), -2.1622182813725894, TOL);
    }

    #[test]
    fn cdf() {
        let x2 = ChiSquared::new(2.5).unwrap();
        assert::close(x2.cdf(&1.2_f64), 0.33859384379982849, TOL);
        assert::close(x2.cdf(&3.4_f64), 0.74430510487063328, TOL);
    }

    #[test]
    fn mean() {
        let m: f64 = ChiSquared::new(2.5).unwrap().mean().unwrap();
        assert::close(m, 2.5, TOL);
    }

    #[test]
    fn variance() {
        let v: f64 = ChiSquared::new(2.5).unwrap().variance().unwrap();
        assert::close(v, 5.0, TOL);
    }

    #[test]
    fn kurtosis() {
        let k = ChiSquared::new(2.5).unwrap().kurtosis().unwrap();
        assert::close(k, 4.8, TOL);
    }

    #[test]
    fn skewness() {
        let s = ChiSquared::new(2.5).unwrap().skewness().unwrap();
        assert::close(s, 1.7888543819998317, TOL);
    }
}
