//! Exponential distribution over x in [0, ∞)
extern crate rand;

use std::f64::consts::LN_2;
use std::io;

use self::rand::distributions::Exp;
use self::rand::Rng;

use traits::*;

/// Exponential distribution, Exp(λ) over x in [0, ∞)
///
/// # Examples
///
/// Compute 50% confidence interval
///
/// ```rust
/// # extern crate rv;
/// use rv::prelude::*;
///
/// let expon = Exponential::new(1.5).unwrap();
/// let interval: (f64, f64) = expon.interval(0.5);  // (0.19, 0.92)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exponential {
    /// λ > 0, rate or inverse scale
    pub rate: f64,
}

impl Exponential {
    pub fn new(rate: f64) -> io::Result<Self> {
        if rate > 0.0 && rate.is_finite() {
            Ok(Exponential { rate })
        } else {
            let err_kind = io::ErrorKind::InvalidInput;
            let msg = "rate must be finite and greater than zero";
            let err = io::Error::new(err_kind, msg);
            Err(err)
        }
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Exponential {
            fn ln_f(&self, x: &$kind) -> f64 {
                self.rate.ln() - self.rate * f64::from(*x)
            }

            #[inline]
            fn ln_normalizer(&self) -> f64 {
                0.0
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let expdist = Exp::new(self.rate);
                rng.sample(expdist) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let expdist = Exp::new(self.rate);
                (0..n).map(|_| rng.sample(expdist) as $kind).collect()
            }
        }

        impl Support<$kind> for Exponential {
            fn contains(&self, x: &$kind) -> bool {
                *x >= 0.0 && x.is_finite()
            }
        }

        impl ContinuousDistr<$kind> for Exponential {}

        impl Cdf<$kind> for Exponential {
            fn cdf(&self, x: &$kind) -> f64 {
                1.0 - (-self.rate * f64::from(*x)).exp()
            }
        }

        impl InverseCdf<$kind> for Exponential {
            fn invcdf(&self, p: f64) -> $kind {
                let x = -(1.0 - p).ln() / self.rate;
                x as $kind
            }
        }

        impl Mean<$kind> for Exponential {
            fn mean(&self) -> Option<$kind> {
                Some(self.rate.recip() as $kind)
            }
        }

        impl Median<$kind> for Exponential {
            fn median(&self) -> Option<$kind> {
                Some((LN_2 / self.rate) as $kind)
            }
        }

        impl Mode<$kind> for Exponential {
            fn mode(&self) -> Option<$kind> {
                Some(0.0)
            }
        }

        impl Variance<$kind> for Exponential {
            fn variance(&self) -> Option<$kind> {
                Some(self.rate.recip().powi(2) as $kind)
            }
        }
    };
}

impl Skewness for Exponential {
    fn skewness(&self) -> Option<f64> {
        Some(2.0)
    }
}

impl Kurtosis for Exponential {
    fn kurtosis(&self) -> Option<f64> {
        Some(6.0)
    }
}

impl Entropy for Exponential {
    fn entropy(&self) -> f64 {
        1.0 - self.rate.ln()
    }
}

impl KlDivergence for Exponential {
    fn kl(&self, other: &Self) -> f64 {
        self.rate.ln() - other.rate.ln() + self.rate / other.rate - 1.0
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
        let expon = Exponential::new(1.5).unwrap();
        assert::close(expon.rate, 1.5, TOL);
    }

    #[test]
    fn new_should_reject_non_finite_rate() {
        assert!(Exponential::new(1.5).is_ok());
        assert!(Exponential::new(f64::NAN).is_err());
        assert!(Exponential::new(f64::INFINITY).is_err());
    }

    #[test]
    fn new_should_reject_leq_0_rate() {
        assert!(Exponential::new(f64::MIN_POSITIVE).is_ok());
        assert!(Exponential::new(0.0).is_err());
        assert!(Exponential::new(-f64::MIN_POSITIVE).is_err());
    }

    #[test]
    fn ln_pdf() {
        let expon = Exponential::new(1.5).unwrap();
        assert::close(expon.ln_pdf(&1.2_f64), -1.3945348918918357, TOL);
        assert::close(expon.ln_pdf(&0.2_f64), 0.1054651081081644, TOL);
        assert::close(expon.ln_pdf(&4.4_f64), -6.1945348918918359, TOL);
    }

    #[test]
    fn cdf() {
        let expon = Exponential::new(1.5).unwrap();
        assert::close(expon.cdf(&1.2_f64), 0.83470111177841344, TOL);
        assert::close(expon.cdf(&0.2_f64), 0.25918177931828218, TOL);
        assert::close(expon.cdf(&4.4_f64), 0.99863963196245209, TOL);
    }

    #[test]
    fn mean() {
        let m: f64 = Exponential::new(1.5).unwrap().mean().unwrap();
        assert::close(m, 0.66666666666666663, TOL);
    }

    #[test]
    fn median() {
        let m: f64 = Exponential::new(1.5).unwrap().median().unwrap();
        assert::close(m, 0.46209812037329684, TOL);
    }

    #[test]
    fn mode() {
        let m: f64 = Exponential::new(1.5).unwrap().mode().unwrap();
        assert::close(m, 0.0, TOL);
    }

    #[test]
    fn variance() {
        let v: f64 = Exponential::new(1.5).unwrap().variance().unwrap();
        assert::close(v, 0.44444444444444442, TOL);
    }

    #[test]
    fn skewness() {
        let s = Exponential::new(1.5).unwrap().skewness().unwrap();
        assert::close(s, 2.0, TOL);
    }

    #[test]
    fn kurtosis() {
        let k = Exponential::new(1.5).unwrap().kurtosis().unwrap();
        assert::close(k, 6.0, TOL);
    }

    #[test]
    fn entropy() {
        let h = Exponential::new(1.5).unwrap().entropy();
        assert::close(h, 0.5945348918918356, TOL);
    }

    #[test]
    fn quantile() {
        let expon = Exponential::new(1.5).unwrap();
        let q25: f64 = expon.quantile(0.25);
        let q75: f64 = expon.quantile(0.75);
        assert::close(q25, 0.19178804830118726, TOL);
        assert::close(q75, 0.92419624074659368, TOL);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let expon = Exponential::new(1.5).unwrap();
        let cdf = |x: f64| expon.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = expon.sample(1000, &mut rng);
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