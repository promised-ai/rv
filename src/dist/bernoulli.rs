//! Bernoulli distribution of x in {0, 1}
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::{BernoulliSuffStat, Booleable};
use crate::impl_display;
use crate::traits::{Cdf, DiscreteDistr, Entropy, HasDensity, HasSuffStat, KlDivergence, Kurtosis, Mean, Median, Mode, Parameterized, Sampleable, Skewness, SuffStat, Support, Variance};
use rand::Rng;
use std::f64;
use std::fmt;

/// [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)
/// with success probability *p*
///
/// # Example
///
/// ```
/// # use rv::prelude::*;
/// let b = Bernoulli::new(0.75).unwrap();
/// assert::close(b.pmf(&true), 0.75, 1E-12);
/// ```
///
/// The following example panics because 2 is out of outside the Bernoulli
/// support
///
/// ```should_panic
/// # use rv::prelude::*;
/// let b = Bernoulli::new(0.75).unwrap();
/// assert!(!b.supports(&2_u8));
///
/// b.pmf(&2_u8); // panics
/// ```
///
/// Parameters struct for Bernoulli distribution
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct BernoulliParameters {
    /// Probability of a success (x=1)
    pub p: f64,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Bernoulli {
    /// Probability of a success (x=1)
    p: f64,
}

impl Parameterized for Bernoulli {
    type Parameters = BernoulliParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters { p: self.p() }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.p)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum BernoulliError {
    /// Bernoulli p is less than zero
    PLessThanZero { p: f64 },
    /// Bernoulli p is greater than one
    PGreaterThanOne { p: f64 },
    /// Bernoulli p is infinite or NaN
    PNotFinite { p: f64 },
}

impl Bernoulli {
    /// Create a new Bernoulli distribution.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rv::dist::Bernoulli;
    /// # use rv::traits::*;
    /// # let mut rng = rand::thread_rng();
    /// let b = Bernoulli::new(0.5).unwrap();
    ///
    /// let coin_flips: Vec<bool> = b.sample(5, &mut rng);
    ///
    /// assert_eq!(coin_flips.len(), 5);
    /// ```
    ///
    /// `Bernoulli::new` will return an `Error` type if given an invalid
    /// parameter.
    ///
    /// ```rust
    /// # use rv::dist::Bernoulli;
    /// assert!(Bernoulli::new(-1.0).is_err());
    /// assert!(Bernoulli::new(1.1).is_err());
    /// ```
    pub fn new(p: f64) -> Result<Self, BernoulliError> {
        if !p.is_finite() {
            Err(BernoulliError::PNotFinite { p })
        } else if p > 1.0 {
            Err(BernoulliError::PGreaterThanOne { p })
        } else if p < 0.0 {
            Err(BernoulliError::PLessThanZero { p })
        } else {
            Ok(Bernoulli { p })
        }
    }

    /// Creates a new Bernoulli without checking whether parameter value is
    /// valid.
    #[inline]
    #[must_use] pub fn new_unchecked(p: f64) -> Self {
        Bernoulli { p }
    }

    /// A Bernoulli distribution with a 50% chance of success
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Bernoulli;
    /// let b = Bernoulli::uniform();
    ///
    /// assert_eq!(b.p(), 0.5);
    /// assert_eq!(b.q(), 0.5);
    /// ```
    #[inline]
    #[must_use] pub fn uniform() -> Self {
        Bernoulli { p: 0.5 }
    }

    /// Get p, the probability of success.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Bernoulli;
    /// let b = Bernoulli::new(0.2).unwrap();
    ///
    /// assert_eq!(b.p(), 0.2);
    /// ```
    #[inline]
    #[must_use] pub fn p(&self) -> f64 {
        self.p
    }

    /// Set p, the probability of success.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Bernoulli;
    /// let mut b = Bernoulli::new(0.2).unwrap();
    /// b.set_p(0.5).unwrap();
    ///
    /// assert_eq!(b.p(), 0.5);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Bernoulli;
    /// # let mut b = Bernoulli::new(0.2).unwrap();
    /// assert!(b.set_p(0.0).is_ok());
    /// assert!(b.set_p(1.0).is_ok());
    /// assert!(b.set_p(-1.0).is_err());
    /// assert!(b.set_p(1.1).is_err());
    /// assert!(b.set_p(f64::INFINITY).is_err());
    /// assert!(b.set_p(std::f64::NEG_INFINITY).is_err());
    /// assert!(b.set_p(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_p(&mut self, p: f64) -> Result<(), BernoulliError> {
        if !p.is_finite() {
            Err(BernoulliError::PNotFinite { p })
        } else if p > 1.0 {
            Err(BernoulliError::PGreaterThanOne { p })
        } else if p < 0.0 {
            Err(BernoulliError::PLessThanZero { p })
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
    /// # use rv::dist::Bernoulli;
    /// let b = Bernoulli::new(0.2).unwrap();
    ///
    /// assert_eq!(b.q(), 0.8);
    /// ```
    #[inline]
    #[must_use] pub fn q(&self) -> f64 {
        1.0 - self.p
    }
}

impl Default for Bernoulli {
    fn default() -> Self {
        Bernoulli::uniform()
    }
}

impl From<&Bernoulli> for String {
    fn from(b: &Bernoulli) -> String {
        format!("Bernoulli(p: {})", b.p)
    }
}

impl_display!(Bernoulli);

impl<X: Booleable> HasDensity<X> for Bernoulli {
    fn f(&self, x: &X) -> f64 {
        let val: bool = x.into_bool();
        if val {
            self.p
        } else {
            1.0_f64 - self.p
        }
    }

    fn ln_f(&self, x: &X) -> f64 {
        // TODO: this is really slow, we should cache ln(p) and ln(q)
        self.f(x).ln()
    }
}

impl<X: Booleable> Sampleable<X> for Bernoulli {
    fn draw<R: Rng>(&self, rng: &mut R) -> X {
        let u = rand_distr::Open01;
        let x: f64 = rng.sample(u);
        X::from_bool(x < self.p)
    }

    fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<X> {
        let u = rand_distr::Open01;
        (0..n)
            .map(|_| {
                let x: f64 = rng.sample(u);
                X::from_bool(x < self.p)
            })
            .collect()
    }
}

impl<X: Booleable> Support<X> for Bernoulli {
    fn supports(&self, x: &X) -> bool {
        x.try_into_bool().is_some()
    }
}

impl<X: Booleable> DiscreteDistr<X> for Bernoulli {
    fn pmf(&self, x: &X) -> f64 {
        let val: bool = x.into_bool();
        self.f(&val)
    }

    fn ln_pmf(&self, x: &X) -> f64 {
        let val: bool = x.into_bool();
        self.ln_f(&val)
    }
}

impl<X: Booleable> Cdf<X> for Bernoulli {
    fn cdf(&self, x: &X) -> f64 {
        let val: bool = x.into_bool();
        if val {
            1.0
        } else {
            self.q()
        }
    }
}

impl<X: Booleable> Mode<X> for Bernoulli {
    fn mode(&self) -> Option<X> {
        let q = self.q();
        if self.p < q {
            Some(X::from_bool(false))
        } else if (self.p - q).abs() < f64::EPSILON {
            None
        } else {
            Some(X::from_bool(true))
        }
    }
}

impl<X: Booleable> HasSuffStat<X> for Bernoulli {
    type Stat = BernoulliSuffStat;
    fn empty_suffstat(&self) -> Self::Stat {
        BernoulliSuffStat::new()
    }

    fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
        let n = stat.n() as f64;
        let k = stat.k() as f64;

        let ln_p = self.p().ln();
        let ln_q = self.q().ln();
        k.mul_add(ln_p, (n - k) * ln_q)
    }
}

impl KlDivergence for Bernoulli {
    fn kl(&self, other: &Self) -> f64 {
        self.p.mul_add(
            other.p.ln() - self.p.ln(),
            self.q() * (other.q().ln() - self.q().ln()),
        )
    }
}

impl Entropy for Bernoulli {
    fn entropy(&self) -> f64 {
        let q = self.q();
        (-q).mul_add(q.ln(), -self.p * self.p.ln())
    }
}

impl Skewness for Bernoulli {
    fn skewness(&self) -> Option<f64> {
        Some(2.0_f64.mul_add(-self.p, 1.0) / (self.p * self.q()).sqrt())
    }
}

impl Kurtosis for Bernoulli {
    fn kurtosis(&self) -> Option<f64> {
        let q = self.q();
        Some((6.0 * self.p).mul_add(-q, 1.0) / (self.p * q))
    }
}

impl Mean<f64> for Bernoulli {
    fn mean(&self) -> Option<f64> {
        Some(self.p)
    }
}

impl Median<f64> for Bernoulli {
    fn median(&self) -> Option<f64> {
        let q = self.q();
        if self.p < q {
            Some(0.0)
        } else if (self.p - q) < f64::EPSILON {
            Some(0.5)
        } else {
            Some(1.0)
        }
    }
}

impl Variance<f64> for Bernoulli {
    fn variance(&self) -> Option<f64> {
        Some(self.p * (1.0 - self.p))
    }
}

impl std::error::Error for BernoulliError {}

impl fmt::Display for BernoulliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PLessThanZero { p } => {
                write!(f, "p was less than zero: {p}")
            }
            Self::PGreaterThanOne { p } => {
                write!(f, "p was less greater than one: {p}")
            }
            Self::PNotFinite { p } => write!(f, "p was non-finite: {p}"),
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

    test_basic_impls!(bool, Bernoulli, Bernoulli::default());

    #[test]
    fn new() {
        let b: Bernoulli = Bernoulli::new(0.1).unwrap();
        assert::close(b.p, 0.1, TOL);
    }

    #[test]
    fn new_should_reject_oob_p() {
        assert!(Bernoulli::new(0.0).is_ok());
        assert!(Bernoulli::new(1.0).is_ok());
        assert_eq!(
            Bernoulli::new(-0.001),
            Err(BernoulliError::PLessThanZero { p: -0.001 })
        );
        assert_eq!(
            Bernoulli::new(1.001),
            Err(BernoulliError::PGreaterThanOne { p: 1.001 })
        );
    }

    #[test]
    fn new_should_reject_non_finite_p() {
        match Bernoulli::new(f64::NAN) {
            Err(BernoulliError::PNotFinite { .. }) => (),
            Err(_) => panic!("wrong error"),
            Ok(_) => panic!("should've errored"),
        }
        match Bernoulli::new(f64::INFINITY) {
            Err(BernoulliError::PNotFinite { .. }) => (),
            Err(_) => panic!("wrong error"),
            Ok(_) => panic!("should've errored"),
        }
    }

    #[test]
    fn uniform_p_should_be_one_half() {
        let b: Bernoulli = Bernoulli::uniform();
        assert::close(b.p, 0.5, TOL);
    }

    #[test]
    fn q_should_be_the_compliment_of_p() {
        let b: Bernoulli = Bernoulli::new(0.1).unwrap();
        assert::close(b.q(), 0.9, TOL);
    }

    #[test]
    fn pmf_of_true_should_be_p() {
        let b1: Bernoulli = Bernoulli::new(0.1).unwrap();
        assert::close(b1.pmf(&true), 0.1, TOL);

        let b2: Bernoulli = Bernoulli::new(0.85).unwrap();
        assert::close(b2.pmf(&true), 0.85, TOL);
    }

    #[test]
    fn pmf_of_1_should_be_p() {
        let b1: Bernoulli = Bernoulli::new(0.1).unwrap();
        assert::close(b1.pmf(&1_u8), 0.1, TOL);

        let b2: Bernoulli = Bernoulli::new(0.85).unwrap();
        assert::close(b2.pmf(&1_i16), 0.85, TOL);
    }

    #[test]
    fn ln_pmf_of_true_should_be_ln_p() {
        let b1 = Bernoulli::new(0.1).unwrap();
        assert::close(b1.ln_pmf(&true), 0.1_f64.ln(), TOL);

        let b2 = Bernoulli::new(0.85).unwrap();
        assert::close(b2.ln_pmf(&true), 0.85_f64.ln(), TOL);
    }

    #[test]
    fn ln_pmf_of_1_should_be_ln_p() {
        let b1 = Bernoulli::new(0.1).unwrap();
        assert::close(b1.ln_pmf(&1_usize), 0.1_f64.ln(), TOL);

        let b2 = Bernoulli::new(0.85).unwrap();
        assert::close(b2.ln_pmf(&1_i32), 0.85_f64.ln(), TOL);
    }

    #[test]
    fn pmf_of_false_should_be_q() {
        let b1 = Bernoulli::new(0.1).unwrap();
        assert::close(b1.pmf(&false), 0.9, TOL);

        let b2 = Bernoulli::new(0.85).unwrap();
        assert::close(b2.pmf(&false), 0.15, TOL);
    }

    #[test]
    fn pmf_of_0_should_be_q() {
        let b1 = Bernoulli::new(0.1).unwrap();
        assert::close(b1.pmf(&0_u8), 0.9, TOL);

        let b2 = Bernoulli::new(0.85).unwrap();
        assert::close(b2.pmf(&0_u32), 0.15, TOL);
    }

    #[test]
    fn ln_pmf_of_false_should_be_ln_q() {
        let b1 = Bernoulli::new(0.1).unwrap();
        assert::close(b1.ln_pmf(&false), 0.9_f64.ln(), TOL);

        let b2 = Bernoulli::new(0.85).unwrap();
        assert::close(b2.ln_pmf(&false), 0.15_f64.ln(), TOL);
    }

    #[test]
    fn ln_pmf_of_zero_should_be_ln_q() {
        let b1 = Bernoulli::new(0.1).unwrap();
        assert::close(b1.ln_pmf(&0_u8), 0.9_f64.ln(), TOL);

        let b2 = Bernoulli::new(0.85).unwrap();
        assert::close(b2.ln_pmf(&0_i16), 0.15_f64.ln(), TOL);
    }

    #[test]
    fn sample_bools_should_draw_the_correct_number_of_samples() {
        let mut rng = rand::thread_rng();
        let n = 103;
        let xs: Vec<bool> = Bernoulli::uniform().sample(n, &mut rng);
        assert_eq!(xs.len(), n);
    }

    #[test]
    fn sample_ints_should_draw_the_correct_number_of_samples() {
        let mut rng = rand::thread_rng();
        let n = 103;
        let xs: Vec<i16> = Bernoulli::uniform().sample(n, &mut rng);
        assert_eq!(xs.len(), n);
        // and they should all be 0 or 1
        assert!(xs.iter().all(|&x| x == 0 || x == 1));
    }

    #[test]
    fn contains_both_true_and_false() {
        let b = Bernoulli::uniform();
        assert!(b.supports(&true));
        assert!(b.supports(&false));
    }

    #[test]
    fn contains_both_zero_and_one() {
        let b = Bernoulli::uniform();
        assert!(b.supports(&0));
        assert!(b.supports(&1));
        assert!(!b.supports(&-1));
        assert!(!b.supports(&2));
    }

    #[test]
    fn cmf_of_false_is_q() {
        let b = Bernoulli::new(0.1).unwrap();
        assert::close(b.cdf(&false), 0.9, TOL);
    }

    #[test]
    fn cmf_of_zero_is_q() {
        let b = Bernoulli::new(0.1).unwrap();
        assert::close(b.cdf(&0_i16), 0.9, TOL);
    }

    #[test]
    fn cmf_of_true_is_one() {
        let b = Bernoulli::new(0.1).unwrap();
        assert::close(b.cdf(&true), 1.0, TOL);
    }

    #[test]
    fn cmf_of_one_is_one() {
        let b = Bernoulli::new(0.1).unwrap();
        assert::close(b.cdf(&1_u8), 1.0, TOL);
    }

    #[test]
    #[should_panic]
    fn cmf_less_than_zero_fails() {
        let b = Bernoulli::new(0.1).unwrap();
        let _p = b.cdf(&-1_i16);
    }

    #[test]
    fn mean_is_p() {
        assert::close(Bernoulli::new(0.1).unwrap().mean().unwrap(), 0.1, TOL);
        assert::close(Bernoulli::new(0.7).unwrap().mean().unwrap(), 0.7, TOL);
    }

    #[test]
    fn median_for_low_p_is_zero() {
        assert::close(Bernoulli::new(0.1).unwrap().median().unwrap(), 0.0, TOL);
        assert::close(
            Bernoulli::new(0.499).unwrap().median().unwrap(),
            0.0,
            TOL,
        );
    }

    #[test]
    fn median_for_high_p_is_one() {
        assert::close(Bernoulli::new(0.9).unwrap().median().unwrap(), 1.0, TOL);
        assert::close(
            Bernoulli::new(0.5001).unwrap().median().unwrap(),
            1.0,
            TOL,
        );
    }

    #[test]
    fn median_for_p_one_half_is_one_half() {
        assert::close(Bernoulli::new(0.5).unwrap().median().unwrap(), 0.5, TOL);
        assert::close(Bernoulli::uniform().median().unwrap(), 0.5, TOL);
    }

    #[test]
    fn mode_for_high_p_is_true() {
        let m1: bool = Bernoulli::new(0.5001).unwrap().mode().unwrap();
        let m2: bool = Bernoulli::new(0.8).unwrap().mode().unwrap();
        assert!(m1);
        assert!(m2);
    }

    #[test]
    fn mode_for_low_p_is_false() {
        let m1: bool = Bernoulli::new(0.4999).unwrap().mode().unwrap();
        let m2: bool = Bernoulli::new(0.2).unwrap().mode().unwrap();
        assert!(!m1);
        assert!(!m2);
    }

    #[test]
    fn mode_for_high_p_is_one() {
        let m1: u8 = Bernoulli::new(0.5001).unwrap().mode().unwrap();
        let m2: u16 = Bernoulli::new(0.8).unwrap().mode().unwrap();
        assert_eq!(m1, 1);
        assert_eq!(m2, 1);
    }

    #[test]
    fn mode_for_low_p_is_zero() {
        let m1: u8 = Bernoulli::new(0.4999).unwrap().mode().unwrap();
        let m2: u8 = Bernoulli::new(0.2).unwrap().mode().unwrap();
        assert_eq!(m1, 0);
        assert_eq!(m2, 0);
    }

    #[test]
    fn mode_for_even_p_is_none() {
        let m1: Option<bool> = Bernoulli::new(0.5).unwrap().mode();
        let m2: Option<u8> = Bernoulli::uniform().mode();
        assert!(m1.is_none());
        assert!(m2.is_none());
    }

    #[test]
    fn variance_for_uniform() {
        assert::close(Bernoulli::uniform().variance().unwrap(), 0.25, TOL);
    }

    #[test]
    fn variance() {
        assert::close(
            Bernoulli::new(0.1).unwrap().variance().unwrap(),
            0.09,
            TOL,
        );
        assert::close(
            Bernoulli::new(0.9).unwrap().variance().unwrap(),
            0.09,
            TOL,
        );
    }

    #[test]
    fn entropy() {
        let b1 = Bernoulli::new(0.1).unwrap();
        let b2 = Bernoulli::new(0.9).unwrap();
        assert::close(b1.entropy(), 0.325_082_973_391_448_2, TOL);
        assert::close(b2.entropy(), 0.325_082_973_391_448_2, TOL);
    }

    #[test]
    fn uniform_entropy() {
        let b = Bernoulli::uniform();
        assert::close(b.entropy(), f64::consts::LN_2, TOL);
    }

    #[test]
    fn uniform_skewness_should_be_zero() {
        let b = Bernoulli::uniform();
        assert::close(b.skewness().unwrap(), 0.0, TOL);
    }

    #[test]
    fn skewness() {
        let b = Bernoulli::new(0.3).unwrap();
        assert::close(b.skewness().unwrap(), 0.872_871_560_943_969_6, TOL);
    }

    #[test]
    fn uniform_kurtosis() {
        let b = Bernoulli::uniform();
        assert::close(b.kurtosis().unwrap(), -2.0, TOL);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let b = Bernoulli::new(0.7).unwrap();
        let ps: Vec<f64> = vec![0.3, 0.7];

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0, 0];
            let xs: Vec<usize> = b.sample(1000, &mut rng);
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
    fn set_p() {
        let mut bern = Bernoulli::new(0.6).unwrap();
        assert::close(bern.pmf(&true), 0.6, 1E-10);

        bern.set_p(0.5).unwrap();

        assert::close(bern.pmf(&true), 0.5, 1E-10);
    }

    #[test]
    fn ln_f_stat() {
        let data: Vec<bool> = vec![true, false, false, false, true];
        let mut stat = BernoulliSuffStat::new();
        stat.observe_many(&data);

        let bern = Bernoulli::new(0.3).unwrap();

        let ln_f_base: f64 = data.iter().map(|x| bern.ln_f(x)).sum();
        let ln_f_stat: f64 =
            <Bernoulli as HasSuffStat<bool>>::ln_f_stat(&bern, &stat);

        assert::close(ln_f_base, ln_f_stat, TOL);
    }
}
