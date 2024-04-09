//! Binomial distribution
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::misc::ln_binom;
use crate::traits::*;
use rand::Rng;
use std::f64;
use std::fmt;

/// [Binomial distribution](https://en.wikipedia.org/wiki/Beta-binomial_distribution)
/// with success probability *p*
///
/// # Examples
///
/// ```
/// use rv::prelude::*;
///
/// let binom = Binomial::new(4, 0.5).unwrap();
/// let cdf = binom.cdf(&2_u8);
///
/// assert_eq!(cdf, binom.pmf(&0_u8) + binom.pmf(&1_u8) + binom.pmf(&2_u8))
/// ```
///
/// Values outside the support of [0, n] can cause panics in certain functions
///
/// ```
/// # use rv::prelude::*;
/// let n = 4;
/// let binom = Binomial::new(n, 0.5).unwrap();
/// assert!(!binom.supports(&5_u8))
/// ```
///
/// The maximum allowed value is 4, so the PMF of 5 will be 0.0
///
/// ```
/// # use rv::prelude::*;
/// # let n = 4;
/// # let binom = Binomial::new(n, 0.5).unwrap();
/// let f = binom.pmf(&5_u8);
/// assert_eq!(f, 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Binomial {
    /// Total number of trials
    n: u64,
    /// Probability of a success
    p: f64,
}

pub struct BinomialParameters {
    pub n: u64,
    pub p: f64,
}

impl Parameterized for Binomial {
    type Parameters = BinomialParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            n: self.n(),
            p: self.p(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.n, params.p)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum BinomialError {
    /// The number of trials is zero
    NIsZero,
    /// Bernoulli p is less than zero
    PLessThanZero { p: f64 },
    /// Bernoulli p is greater than one
    PGreaterThanOne { p: f64 },
    /// Bernoulli p is infinite or NaN
    PNotFinite { p: f64 },
}

impl Binomial {
    /// Create a new Binomial distribution
    ///
    /// # Arguments
    ///
    /// - n: the total number of trials
    /// - p: the pobability of success
    pub fn new(n: u64, p: f64) -> Result<Self, BinomialError> {
        if n == 0 {
            Err(BinomialError::NIsZero)
        } else if p < 0.0 {
            Err(BinomialError::PLessThanZero { p })
        } else if p > 1.0 {
            Err(BinomialError::PGreaterThanOne { p })
        } else if !p.is_finite() {
            Err(BinomialError::PNotFinite { p })
        } else {
            Ok(Binomial { n, p })
        }
    }

    /// Creates a new Binomial without checking whether the parameters are
    /// valid.
    #[inline]
    pub fn new_unchecked(n: u64, p: f64) -> Self {
        Binomial { n, p }
    }

    /// A Binomial distribution with a 50% chance of success
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Binomial;
    /// let binom = Binomial::uniform(11);
    /// assert_eq!(binom.p(), 0.5);
    /// ```
    #[inline]
    pub fn uniform(n: u64) -> Self {
        Binomial::new_unchecked(n, 0.5)
    }

    /// Get the number of trials
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Binomial;
    /// let binom = Binomial::uniform(11);
    /// assert_eq!(binom, Binomial::new(11, 0.5).unwrap());
    /// ```
    #[inline]
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Set the value of the n parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::Binomial;
    ///
    /// let mut binom = Binomial::new(10, 0.5).unwrap();
    ///
    /// binom.set_n(11).unwrap();
    ///
    /// assert_eq!(binom.n(), 11);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Binomial;
    /// # let mut binom = Binomial::new(10, 0.5).unwrap();
    /// assert!(binom.set_n(11).is_ok());
    /// assert!(binom.set_n(1).is_ok());
    /// assert!(binom.set_n(0).is_err());
    /// ```
    #[inline]
    pub fn set_n(&mut self, n: u64) -> Result<(), BinomialError> {
        if n == 0 {
            Err(BinomialError::NIsZero)
        } else {
            self.set_n_unchecked(n);
            Ok(())
        }
    }

    /// Set the value of n without input validation
    #[inline]
    pub fn set_n_unchecked(&mut self, n: u64) {
        self.n = n
    }

    /// Get the probability of success
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Binomial;
    /// let binom = Binomial::new(10, 0.2).unwrap();
    /// assert_eq!(binom.p(), 0.2);
    /// ```
    #[inline]
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Set p, the probability of success.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Binomial;
    /// let mut binom = Binomial::new(10, 0.2).unwrap();
    /// binom.set_p(0.5).unwrap();
    ///
    /// assert_eq!(binom.p(), 0.5);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Binomial;
    /// # let mut binom = Binomial::new(10, 0.2).unwrap();
    /// assert!(binom.set_p(0.0).is_ok());
    /// assert!(binom.set_p(1.0).is_ok());
    /// assert!(binom.set_p(-1.0).is_err());
    /// assert!(binom.set_p(1.1).is_err());
    /// assert!(binom.set_p(f64::INFINITY).is_err());
    /// assert!(binom.set_p(f64::NEG_INFINITY).is_err());
    /// assert!(binom.set_p(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_p(&mut self, p: f64) -> Result<(), BinomialError> {
        if !p.is_finite() {
            Err(BinomialError::PNotFinite { p })
        } else if p > 1.0 {
            Err(BinomialError::PGreaterThanOne { p })
        } else if p < 0.0 {
            Err(BinomialError::PLessThanZero { p })
        } else {
            self.set_p_unchecked(p);
            Ok(())
        }
    }

    /// Set p without input validation
    #[inline]
    pub fn set_p_unchecked(&mut self, p: f64) {
        self.p = p;
    }

    /// The complement of `p`, i.e. `(1 - p)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Binomial;
    /// let binom = Binomial::new(10, 0.2).unwrap();
    /// assert_eq!(binom.q(), 0.8);
    /// ```
    #[inline]
    pub fn q(&self) -> f64 {
        1.0 - self.p
    }
}

impl From<&Binomial> for String {
    fn from(b: &Binomial) -> String {
        format!("Binomial({}; p: {})", b.n, b.p)
    }
}

impl_display!(Binomial);

macro_rules! impl_int_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for Binomial {
            fn ln_f(&self, k: &$kind) -> f64 {
                let nf = self.n as f64;
                let kf = *k as f64;
                // TODO: could cache ln(p) and ln(q)
                self.q().ln().mul_add(
                    (nf - kf),
                    self.p.ln().mul_add(kf, ln_binom(nf, kf)),
                )
            }
        }

        impl Sampleable<$kind> for Binomial {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let b = rand_distr::Binomial::new(self.n, self.p).unwrap();
                rng.sample(b) as $kind
            }
        }

        impl Support<$kind> for Binomial {
            #[allow(unused_comparisons)]
            fn supports(&self, k: &$kind) -> bool {
                *k >= 0 && *k <= self.n as $kind
            }
        }

        impl DiscreteDistr<$kind> for Binomial {}

        impl Cdf<$kind> for Binomial {
            fn cdf(&self, k: &$kind) -> f64 {
                (0..=*k).fold(0.0, |acc, x| acc + self.pmf(&x))
            }
        }
    };
}

impl Skewness for Binomial {
    fn skewness(&self) -> Option<f64> {
        let nf = self.n as f64;
        Some(2.0_f64.mul_add(-self.p, 1.0) / (nf * self.p * self.q()).sqrt())
    }
}

impl Kurtosis for Binomial {
    fn kurtosis(&self) -> Option<f64> {
        let q = self.q();
        let nf = self.n as f64;
        Some((6.0 * self.p).mul_add(-q, 1.0) / (nf * self.p * q))
    }
}

impl Mean<f64> for Binomial {
    fn mean(&self) -> Option<f64> {
        Some(self.n as f64 * self.p)
    }
}

impl Variance<f64> for Binomial {
    fn variance(&self) -> Option<f64> {
        Some(self.n as f64 * self.p * (1.0 - self.p))
    }
}

impl_int_traits!(u8);
impl_int_traits!(u16);
impl_int_traits!(u32);
impl_int_traits!(u64);
impl_int_traits!(usize);

impl_int_traits!(i8);
impl_int_traits!(i16);
impl_int_traits!(i32);
impl_int_traits!(i64);

impl std::error::Error for BinomialError {}

impl fmt::Display for BinomialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PLessThanZero { p } => {
                write!(f, "p ({}) was less than zero", p)
            }
            Self::PGreaterThanOne { p } => {
                write!(f, "p ({}) was greater than zero", p)
            }
            Self::PNotFinite { p } => write!(f, "p ({}) was non-finite", p),
            Self::NIsZero => write!(f, "n was zero"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::x2_test;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;
    const N_TRIES: usize = 5;
    const X2_PVAL: f64 = 0.2;

    test_basic_impls!(u32, Binomial, Binomial::uniform(10));

    #[test]
    fn new() {
        let binom = Binomial::new(10, 0.6).unwrap();
        assert_eq!(binom.n, 10);
        assert::close(binom.p, 0.6, TOL);
    }

    #[test]
    fn new_should_reject_n_zero() {
        assert!(Binomial::new(0, 0.6).is_err());
    }

    #[test]
    fn new_should_reject_low_p() {
        assert!(Binomial::new(10, -0.1).is_err());
        assert!(Binomial::new(10, -1.0).is_err());
    }

    #[test]
    fn new_should_reject_high_p() {
        assert!(Binomial::new(10, 1.1).is_err());
        assert!(Binomial::new(10, 200.0).is_err());
    }

    #[test]
    fn ln_pmf() {
        let binom = Binomial::new(10, 0.6).unwrap();
        let xs: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let known_values = vec![
            -9.162_907_318_741_55,
            -6.454_857_117_639_339,
            -4.545_314_612_754_902,
            -3.159_020_251_635_010_5,
            -2.193_939_355_591_423_3,
            -1.606_152_690_689_303_8,
            -1.383_009_139_375_095,
            -1.537_159_819_202_353_5,
            -2.112_523_964_105_916_4,
            -3.211_136_252_774_024_6,
            -5.108_256_237_659_907,
        ];
        let generated_values: Vec<f64> =
            xs.iter().map(|x| binom.ln_pmf(x)).collect();
        assert::close(known_values, generated_values, TOL);
    }

    #[test]
    fn cdf() {
        let binom = Binomial::new(10, 0.6).unwrap();
        let xs: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let known_values = vec![
            0.000_104_857_600_000_000_06,
            0.001_677_721_600_000_000_5,
            0.012_294_553_600_000_008,
            0.054_761_881_600_000_02,
            0.166_238_617_6,
            0.366_896_742_400_000_03,
            0.617_719_398_399_999_9,
            0.832_710_246_4,
            0.953_642_598_4,
            0.993_953_382_4,
            1.0,
        ];
        let generated_values: Vec<f64> =
            xs.iter().map(|x| binom.cdf(x)).collect();
        assert::close(known_values, generated_values, TOL);
    }

    #[test]
    fn support() {
        let binom = Binomial::new(10, 0.6).unwrap();
        assert!((0..=10).all(|x| binom.supports(&x)));
        assert!(!binom.supports(&-1_i32));
        assert!(!binom.supports(&11_u32));
    }

    #[test]
    fn kurtosis() {
        let k1 = Binomial::new(10, 0.6).unwrap().kurtosis().unwrap();
        let k2 = Binomial::new(21, 0.21).unwrap().kurtosis().unwrap();
        assert::close(k1, -0.183_333_333_333_333_3, TOL);
        assert::close(k2, 0.001_320_359_367_375_624_2, TOL);
    }

    #[test]
    fn skewness() {
        let s1 = Binomial::new(10, 0.6).unwrap().skewness().unwrap();
        let s2 = Binomial::new(21, 0.21).unwrap().skewness().unwrap();
        assert::close(s1, -0.129_099_444_873_580_52, TOL);
        assert::close(s2, 0.310_738_563_112_901_9, TOL);
    }

    #[test]
    fn mean() {
        let m1 = Binomial::new(10, 0.6).unwrap().mean().unwrap();
        let m2 = Binomial::new(21, 0.21).unwrap().mean().unwrap();
        assert::close(m1, 6.0, TOL);
        assert::close(m2, 4.41, TOL);
    }

    #[test]
    fn variance() {
        let v1 = Binomial::new(10, 0.6).unwrap().variance().unwrap();
        let v2 = Binomial::new(21, 0.21).unwrap().variance().unwrap();
        assert::close(v1, 2.400_000_000_000_000_4, TOL);
        assert::close(v2, 3.4839, TOL);
    }

    #[test]
    fn sample_test() {
        let mut rng = rand::thread_rng();
        let binom = Binomial::new(5, 0.6).unwrap();
        let ps: Vec<f64> = vec![
            0.010_240_000_000_000_008,
            0.076_800_000_000_000_01,
            0.2304,
            0.345_599_999_999_999_9,
            0.259_199_999_999_999_9,
            0.07776,
        ];

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; 6];
            let xs: Vec<usize> = binom.sample(1000, &mut rng);
            xs.iter().for_each(|&x| f_obs[x] += 1);
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
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let binom = Binomial::new(5, 0.6).unwrap();
        let ps: Vec<f64> = vec![
            0.010_240_000_000_000_008,
            0.076_800_000_000_000_01,
            0.2304,
            0.345_599_999_999_999_9,
            0.259_199_999_999_999_9,
            0.07776,
        ];

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; 6];
            let xs: Vec<usize> =
                (0..1000).map(|_| binom.draw(&mut rng)).collect();
            xs.iter().for_each(|&x| f_obs[x] += 1);
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
