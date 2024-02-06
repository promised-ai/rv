#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::consts::LN_2PI_E;
use crate::data::PoissonSuffStat;
use crate::impl_display;
use crate::misc::ln_fact;
use crate::traits::*;
use rand::Rng;
use rand_distr::Poisson as RPossion;
use special::Gamma as _;
use std::fmt;
use std::sync::OnceLock;

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
///
/// The Poisson can have two modes. The modes are distinct only if the rate is
/// an integer.
///
/// ```
/// # use rv::prelude::*;
/// {
///     let pois = Poisson::new(2.0).unwrap();
///     let modes: (u32, u32) = pois.mode().unwrap();
///
///     assert_eq!(modes, (1, 2))
/// }
///
/// {
///     let pois = Poisson::new(2.1).unwrap();
///     let modes: (u32, u32) = pois.mode().unwrap();
///
///     assert_eq!(modes, (2, 2))
/// }
/// ```
///
/// If we know that the rate is not an integer, or we only care about one of
/// the modes, we can call mode for an unsigned type, which will return the
/// leftmost (lowest) mode.
///
/// ```
/// # use rv::prelude::*;
/// let pois = Poisson::new(2.1).unwrap();
/// let mode: u32 = pois.mode().unwrap();
///
/// assert_eq!(mode, 2)
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Poisson {
    rate: f64,
    /// Cached ln(rate)
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_rate: OnceLock<f64>,
}

impl PartialEq for Poisson {
    fn eq(&self, other: &Poisson) -> bool {
        self.rate == other.rate
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
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
            ln_rate: OnceLock::new(),
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
        self.ln_rate = OnceLock::new();
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
                let kf = *x as f64;
                kf.mul_add(self.ln_rate(), -self.rate) - ln_fact(*x as usize)
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let pois = RPossion::new(self.rate).unwrap();
                let x: u64 = rng.sample(pois) as u64;
                x as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let pois = RPossion::new(self.rate).unwrap();
                (0..n)
                    .map(|_| {
                        let x: u64 = rng.sample(pois) as u64;
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
                let kf = *x as f64;
                1.0 - (self.rate).inc_gamma(kf + 1.0)
            }
        }

        impl HasSuffStat<$kind> for Poisson {
            type Stat = PoissonSuffStat;

            fn empty_suffstat(&self) -> Self::Stat {
                PoissonSuffStat::new()
            }

            fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
                let n = stat.n() as f64;
                let t1 =
                    self.ln_rate().mul_add(stat.sum(), -stat.sum_ln_fact());
                n.mul_add(-self.rate, t1)
            }
        }

        impl Mode<($kind, $kind)> for Poisson {
            fn mode(&self) -> Option<($kind, $kind)> {
                let left = self.rate.ceil() as $kind - 1;
                let right = self.rate.floor() as $kind;
                Some((left, right))
            }
        }

        impl Mode<$kind> for Poisson {
            fn mode(&self) -> Option<$kind> {
                Some(self.rate.ceil() as $kind - 1)
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

impl KlDivergence for Poisson {
    fn kl(&self, other: &Poisson) -> f64 {
        self.rate()
            .mul_add(self.ln_rate() - other.ln_rate(), other.rate())
            - self.rate()
    }
}

impl Entropy for Poisson {
    fn entropy(&self) -> f64 {
        // TODO: optimize this. Should be some better approximations out there
        if self.rate() < 200.0 {
            // compute expectation until f(x) is close to zero
            let mid = self.rate().floor() as u32;
            crate::misc::entropy::count_entropy(self, mid)
        } else {
            // Approximation for large rate. Error is O(1/rate^3)
            // https://en.wikipedia.org/wiki/Poisson_distribution
            19.0_f64.mul_add(
                -(360.0 * self.rate().powi(3)).recip(),
                0.5_f64.mul_add(
                    LN_2PI_E + self.ln_rate(),
                    -(12.0 * self.rate()).recip(),
                ) - (24.0 * self.rate() * self.rate()).recip(),
            )
        }
    }
}

impl_traits!(u8);
impl_traits!(u16);
impl_traits!(u32);
impl_traits!(usize);

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

    fn brute_force_kl(fx: &Poisson, fy: &Poisson, x_max: u32) -> f64 {
        (0..=x_max)
            .map(|x| {
                let lnfx = fx.ln_f(&x);
                let lnfy = fy.ln_f(&x);
                lnfx.exp() * (lnfx - lnfy)
            })
            .sum()
    }

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
        assert::close(pois.ln_pmf(&1_u32), -3.632_293_179_441_923_8, TOL);
        assert::close(pois.ln_pmf(&5_u32), -1.748_957_639_991_665_8, TOL);
        assert::close(pois.ln_pmf(&11_u32), -4.457_532_819_735_049, TOL);
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
        assert::close(pois.cdf(&1_u32), 0.031_447_041_613_534_364, TOL);
    }

    #[test]
    fn cdf_mid() {
        let pois = Poisson::new(5.3).unwrap();
        // at floor of rate
        assert::close(pois.cdf(&5_u32), 0.563_473_392_288_071_7, TOL);
    }

    #[test]
    fn cdf_high() {
        let pois = Poisson::new(5.3).unwrap();
        assert::close(pois.cdf(&15_u32), 0.999_866_999_508_350_3, TOL);
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
        assert::close(s, 0.434_372_242_763_069_4, TOL);
    }

    #[test]
    fn kurtosis() {
        let k = Poisson::new(5.3).unwrap().kurtosis().unwrap();
        assert::close(k, 0.188_679_245_283_018_88, TOL);
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

    #[test]
    fn kl_divergence_vs_brute() {
        let prior = crate::dist::Gamma::new(1.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let pois_x: Poisson = prior.draw(&mut rng);
            let pois_y: Poisson = prior.draw(&mut rng);

            let kl_true = pois_x.kl(&pois_y);
            let kl_est = brute_force_kl(&pois_x, &pois_y, 1_000);
            assert::close(kl_true, kl_est, TOL);
        }
    }

    #[test]
    fn entropy_value_checks() {
        let rates = [0.1, 0.5, 1.0, 2.2, 3.4, 10.2, 131.4];
        // from scipy, which I think uses an approximation via simulation. Not
        // 100% sure.
        let hs = [
            0.333_676_996_501_232_7,
            0.927_637_467_495_797_5,
            1.304_842_242_256_251_6,
            1.758_957_749_331_246,
            1.999_531_514_109_100_8,
            2.571_495_552_115_918,
            3.857_424_953_514_813,
        ];
        rates.iter().zip(hs.iter()).for_each(|(rate, h)| {
            let pois = Poisson::new(*rate).unwrap();
            assert::close(*h, pois.entropy(), TOL);
        })
    }

    #[test]
    fn mode_value_checks() {
        {
            let pois = Poisson::new(2.0).unwrap();
            let mode: (u32, u32) = pois.mode().unwrap();
            assert_eq!(mode, (1, 2));
        }

        {
            let pois = Poisson::new(2.1).unwrap();
            let mode: (u32, u32) = pois.mode().unwrap();
            assert_eq!(mode, (2, 2));
        }

        {
            let pois = Poisson::new(2.1).unwrap();
            let mode: u32 = pois.mode().unwrap();
            assert_eq!(mode, 2);
        }

        {
            let pois = Poisson::new(2.0).unwrap();
            let mode: u32 = pois.mode().unwrap();
            assert_eq!(mode, 1);
        }
    }

    #[test]
    fn ln_f_stat() {
        let data: Vec<u32> = vec![1, 2, 2, 8, 10, 3];
        let mut stat = PoissonSuffStat::new();
        stat.observe_many(&data);

        let pois = Poisson::new(0.53).unwrap();

        let ln_f_base: f64 = data.iter().map(|x| pois.ln_f(x)).sum();
        let ln_f_stat: f64 =
            <Poisson as HasSuffStat<u32>>::ln_f_stat(&pois, &stat);

        assert::close(ln_f_base, ln_f_stat, TOL);
    }
}
