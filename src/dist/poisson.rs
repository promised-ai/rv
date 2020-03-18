#[cfg(feature = "serde1")]
use serde_derive::{Deserialize, Serialize};

use crate::consts::LN_2PI_E;
use crate::data::PoissonSuffStat;
use crate::misc::ln_fact;
use crate::traits::*;
use crate::{clone_cache_f64, impl_display};
use once_cell::sync::OnceCell;
use rand::Rng;
use rand_distr::Poisson as RPossion;
use special::Gamma as _;
use std::fmt;

/// [Possion distribution](https://en.wikipedia.org/wiki/Poisson_distribution)
/// over x in {0, 1, ... }.
///
/// # Example
///
/// ```
/// use rv::prelude::*;
///
/// // Create Poisson(λ=5.3)
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
#[derive(Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Poisson {
    rate: f64,
    /// Cached ln(rate)
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_rate: OnceCell<f64>,
}

impl Clone for Poisson {
    fn clone(&self) -> Self {
        Poisson {
            rate: self.rate,
            ln_rate: clone_cache_f64!(self, ln_rate),
        }
    }
}

impl PartialEq for Poisson {
    fn eq(&self, other: &Poisson) -> bool {
        self.rate == other.rate
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum PoissonError {
    /// The rate parameter is less than or equal to zero
    RateTooLow { rate: f64 },
    /// The rate parameter is infinite or NaN
    RateNotFinite { rate: f64 },
}

impl Poisson {
    /// Create a new Poisson distribution with given rate
    #[inline]
    pub fn new(rate: f64) -> Result<Self, PoissonError> {
        if rate <= 0.0 {
            Err(PoissonError::RateTooLow { rate })
        } else if !rate.is_finite() {
            Err(PoissonError::RateNotFinite { rate })
        } else {
            Ok(Self::new_unchecked(rate))
        }
    }

    /// Creates a new Poisson without checking whether the parameter is valid.
    #[inline]
    pub fn new_unchecked(rate: f64) -> Self {
        Poisson {
            rate,
            ln_rate: OnceCell::new(),
        }
    }

    #[inline]
    fn ln_rate(&self) -> f64 {
        *self.ln_rate.get_or_init(|| self.rate.ln())
    }

    /// Get the rate parameter
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::Poisson;
    /// let pois = Poisson::new(2.0).unwrap();
    /// assert_eq!(pois.rate(), 2.0);
    /// ```
    #[inline]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Set the rate parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::Poisson;
    /// let mut pois = Poisson::new(1.0).unwrap();
    /// assert_eq!(pois.rate(), 1.0);
    ///
    /// pois.set_rate(1.1).unwrap();
    /// assert_eq!(pois.rate(), 1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Poisson;
    /// # let mut pois = Poisson::new(1.0).unwrap();
    /// assert!(pois.set_rate(1.1).is_ok());
    /// assert!(pois.set_rate(0.0).is_err());
    /// assert!(pois.set_rate(-1.0).is_err());
    /// assert!(pois.set_rate(std::f64::INFINITY).is_err());
    /// assert!(pois.set_rate(std::f64::NEG_INFINITY).is_err());
    /// assert!(pois.set_rate(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_rate(&mut self, rate: f64) -> Result<(), PoissonError> {
        if rate <= 0.0 {
            Err(PoissonError::RateTooLow { rate })
        } else if !rate.is_finite() {
            Err(PoissonError::RateNotFinite { rate })
        } else {
            self.set_rate_unchecked(rate);
            Ok(())
        }
    }

    /// Set the rate parameter without input validation
    #[inline]
    pub fn set_rate_unchecked(&mut self, rate: f64) {
        self.rate = rate;
        self.ln_rate = OnceCell::new();
    }
}

impl From<&Poisson> for String {
    fn from(pois: &Poisson) -> String {
        format!("Poisson(λ: {})", pois.rate)
    }
}

impl_display!(Poisson);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Poisson {
            fn ln_f(&self, x: &$kind) -> f64 {
                let kf = f64::from(*x);
                kf * self.ln_rate() - self.rate - ln_fact(*x as usize)
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let pois = RPossion::new(self.rate).unwrap();
                let x: u64 = rng.sample(pois);
                x as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let pois = RPossion::new(self.rate).unwrap();
                (0..n)
                    .map(|_| {
                        let x: u64 = rng.sample(pois);
                        x as $kind
                    })
                    .collect()
            }
        }

        impl Support<$kind> for Poisson {
            #[allow(unused_comparisons)]
            fn supports(&self, x: &$kind) -> bool {
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

        impl HasSuffStat<$kind> for Poisson {
            type Stat = PoissonSuffStat;
            fn empty_suffstat(&self) -> Self::Stat {
                PoissonSuffStat::new()
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

impl Entropy for Poisson {
    fn entropy(&self) -> f64 {
        // TODO: optimize this
        // - if we could compute the inverse cdf, we could find where the
        //   PMF > 1E-16 so we wouldn't have to compute f(x) when the mass
        //   hasn't stated yet
        // - should be some better approximations out there
        if self.rate() < 150.0 {
            // compute expectation until f(x) is close to zero
            let mean = self.rate().round() as u32;
            crate::misc::count_entropy(&self, 0, mean)
        } else {
            // Approximation for large rate. Error is O(1/rate^3)
            // https://en.wikipedia.org/wiki/Poisson_distribution
            (0.5) * (LN_2PI_E + self.ln_rate())
                - (12.0 * self.rate()).recip()
                - (24.0 * self.rate().powi(2)).recip()
                - 19.0 * (360.0 * self.rate().powi(3)).recip()
        }
    }
}

impl_traits!(u8);
impl_traits!(u16);
impl_traits!(u32);

impl std::error::Error for PoissonError {}

impl fmt::Display for PoissonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RateTooLow { rate } => {
                write!(f, "rate ({}) must be greater than zero", rate)
            }
            Self::RateNotFinite { rate } => {
                write!(f, "non-finite rate: {}", rate)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::x2_test;
    use crate::test_basic_impls;
    use std::f64;

    const TOL: f64 = 1E-12;
    const N_TRIES: usize = 5;
    const X2_PVAL: f64 = 0.2;

    test_basic_impls!([count] Poisson::new(0.5).unwrap());

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
    fn ln_pmf() {
        let pois = Poisson::new(5.3).unwrap();
        assert::close(pois.ln_pmf(&1_u32), -3.6322931794419238, TOL);
        assert::close(pois.ln_pmf(&5_u32), -1.7489576399916658, TOL);
        assert::close(pois.ln_pmf(&11_u32), -4.4575328197350492, TOL);
    }

    #[test]
    fn pmf_preserved_after_rate_set_reset() {
        let x: u32 = 3;
        let mut pois = Poisson::new(5.3).unwrap();

        let pmf_1 = pois.pmf(&x);
        let ln_pmf_1 = pois.ln_pmf(&x);

        pois.set_rate(1.2).unwrap();

        assert!((pmf_1 - pois.pmf(&x)).abs() > 1e-4);
        assert!((ln_pmf_1 - pois.ln_pmf(&x)).abs() > 1e-4);

        pois.set_rate(5.3).unwrap();

        assert_eq!(pmf_1, pois.pmf(&x));
        assert_eq!(ln_pmf_1, pois.ln_pmf(&x));
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
            .unwrap_or(99)
            + 1;

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
