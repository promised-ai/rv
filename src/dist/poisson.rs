//! Possion distribution on unisgned integers
extern crate rand;
extern crate special;

use self::rand::distributions::Poisson as RPossion;
use self::rand::Rng;
use self::special::Gamma as SGamma;
use std::io;
use traits::*;

/// Possion distribution on unisgned integers
///
/// # Example
///
/// ```
/// extern crate rv;
/// extern crate rand;
///
/// use rv::prelude::*;
///
/// // Create Possion(λ=5.3)
/// let pois = Poisson::new(5.3).unwrap();
///
/// // CDF at 5
/// assert!((pois.cdf(&5_u16) - 0.56347339228807169).abs() < 1E-12);
///
/// // Draw 100 samples
/// let mut rng = rand::thread_rng();
/// let xs: Vec<u32> = pois.sample(100, &mut rng);
/// assert_eq!(xs.len(), 100)
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Poisson {
    pub rate: f64,
}

impl Poisson {
    pub fn new(rate: f64) -> io::Result<Self> {
        if rate > 0.0 && rate.is_finite() {
            Ok(Poisson { rate })
        } else {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "rate must be greater than 0");
            Err(err)
        }
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Poisson {
            fn ln_f(&self, x: &$kind) -> f64 {
                let kf = f64::from(*x);
                kf * self.rate.ln() - self.rate - (kf + 1.0).ln_gamma().0
            }

            #[inline]
            fn ln_normalizer(&self) -> f64 {
                0.0
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let pois = RPossion::new(self.rate);
                rng.sample(pois) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let pois = RPossion::new(self.rate);
                (0..n).map(|_| rng.sample(pois) as $kind).collect()
            }
        }

        impl Support<$kind> for Poisson {
            #[allow(unused_comparisons)]
            fn contains(&self, x: &$kind) -> bool {
                *x >= 0
            }
        }

        impl DiscreteDistr<$kind> for Poisson {}

        impl Cdf<$kind> for Poisson {
            fn cdf(&self, x: &$kind) -> f64 {
                let kf = f64::from(*x);
                1.0 - (self.rate).inc_gamma(kf + 1.0)
            }
        }
    };
}

impl Mean<f64> for Poisson {
    fn mean(&self) -> Option<f64> {
        Some(self.rate)
    }
}

impl Variance<f64> for Poisson {
    fn variance(&self) -> Option<f64> {
        Some(self.rate)
    }
}

impl Skewness for Poisson {
    fn skewness(&self) -> Option<f64> {
        Some(self.rate.sqrt().recip())
    }
}

impl Kurtosis for Poisson {
    fn kurtosis(&self) -> Option<f64> {
        Some(self.rate.recip())
    }
}

impl_traits!(u16);
impl_traits!(u32);

#[cfg(test)]
mod tests {
    extern crate assert;
    use super::*;
    use misc::x2_test;
    use std::f64;

    const TOL: f64 = 1E-12;
    const N_TRIES: usize = 5;
    const X2_PVAL: f64 = 0.2;

    #[test]
    fn new() {
        assert::close(Poisson::new(0.001).unwrap().rate, 0.001, TOL);
        assert::close(Poisson::new(1.234).unwrap().rate, 1.234, TOL);
    }

    #[test]
    fn new_should_reject_non_finite_rate() {
        assert!(Poisson::new(f64::INFINITY).is_err());
        assert!(Poisson::new(f64::NAN).is_err());
    }

    #[test]
    fn new_should_reject_rate_lteq_zero() {
        assert!(Poisson::new(0.0).is_err());
        assert!(Poisson::new(-1E-12).is_err());
        assert!(Poisson::new(-1E12).is_err());
        assert!(Poisson::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn ln_pdf() {
        let pois = Poisson::new(5.3).unwrap();
        assert::close(pois.ln_pmf(&1_u32), -3.6322931794419238, TOL);
        assert::close(pois.ln_pmf(&5_u32), -1.7489576399916658, TOL);
        assert::close(pois.ln_pmf(&11_u32), -4.4575328197350492, TOL);
    }

    #[test]
    fn cdf_low() {
        let pois = Poisson::new(5.3).unwrap();
        assert::close(pois.cdf(&1_u32), 0.031447041613534364, TOL);
    }

    #[test]
    fn cdf_mid() {
        let pois = Poisson::new(5.3).unwrap();
        // at floor of rate
        assert::close(pois.cdf(&5_u32), 0.56347339228807169, TOL);
    }

    #[test]
    fn cdf_high() {
        let pois = Poisson::new(5.3).unwrap();
        assert::close(pois.cdf(&15_u32), 0.99986699950835034, TOL);
    }

    #[test]
    fn mean() {
        let m1 = Poisson::new(1.5).unwrap().mean().unwrap();
        assert::close(m1, 1.5, TOL);

        let m2 = Poisson::new(33.2).unwrap().mean().unwrap();
        assert::close(m2, 33.2, TOL);
    }

    #[test]
    fn variance() {
        let v1 = Poisson::new(1.5).unwrap().variance().unwrap();
        assert::close(v1, 1.5, TOL);

        let v2 = Poisson::new(33.2).unwrap().variance().unwrap();
        assert::close(v2, 33.2, TOL);
    }

    #[test]
    fn skewness() {
        let s = Poisson::new(5.3).unwrap().skewness().unwrap();
        assert::close(s, 0.4343722427630694, TOL);
    }

    #[test]
    fn kurtosis() {
        let k = Poisson::new(5.3).unwrap().kurtosis().unwrap();
        assert::close(k, 0.18867924528301888, TOL);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let pois = Poisson::new(2.0).unwrap();

        // How many bins do we need?
        let k: usize = (0..100)
            .position(|x| pois.pmf(&(x as u32)) < f64::EPSILON)
            .unwrap_or(99) + 1;

        let ps: Vec<f64> = (0..k).map(|x| pois.pmf(&(x as u32))).collect();

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; k];
            let xs: Vec<u32> = pois.sample(1000, &mut rng);
            xs.iter().for_each(|&x| f_obs[x as usize] += 1);
            let (_, p) = x2_test(&f_obs, &ps);
            if p > X2_PVAL {
                acc + 1
            } else {
                acc
            }
        });
        assert!(passes > 0);
    }
}
