//! Continuous uniform distribution, U(a, b) on the interval x in [a, b]
use crate::traits::*;
use num::{FromPrimitive, Integer, ToPrimitive};
use rand::Rng;
use rand_distr::uniform::SampleUniform;
use std::f64;
use std::fmt;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

pub trait DuParam: Integer + Copy {}
impl<T> DuParam for T where T: Integer + Copy {}

/// [Discrete uniform distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution),
/// U(a, b) on the interval x in [a, b]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct DiscreteUniform<T: DuParam> {
    a: T,
    b: T,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum DiscreteUniformError {
    /// a is greater than or equal to b
    InvalidInterval,
}

impl<T: DuParam> DiscreteUniform<T> {
    /// Create a new discreet uniform distribution
    ///
    /// # Arguments
    /// - a: lower bound
    /// - b : upper bound
    #[inline]
    pub fn new(a: T, b: T) -> Result<Self, DiscreteUniformError> {
        if a < b {
            Ok(Self { a, b })
        } else {
            Err(DiscreteUniformError::InvalidInterval)
        }
    }

    /// Creates a new DiscreteUniform without checking whether the parameters
    /// are valid.
    #[inline]
    pub fn new_unchecked(a: T, b: T) -> Self {
        Self { a, b }
    }

    /// Get lower bound parameter, a
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::DiscreteUniform;
    /// let du = DiscreteUniform::new(1_u8, 22_u8).unwrap();
    /// assert_eq!(du.a(), 1);
    /// ```
    #[inline]
    pub fn a(&self) -> T {
        self.a
    }

    /// Get upper bound parameter, a
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::DiscreteUniform;
    /// let du = DiscreteUniform::new(1_u8, 22_u8).unwrap();
    /// assert_eq!(du.b(), 22);
    /// ```
    #[inline]
    pub fn b(&self) -> T {
        self.b
    }
}

impl<T> From<&DiscreteUniform<T>> for String
where
    T: DuParam + fmt::Display,
{
    fn from(u: &DiscreteUniform<T>) -> String {
        format!("DiscreteUniform({}, {})", u.a, u.b)
    }
}

impl<T> fmt::Display for DiscreteUniform<T>
where
    T: DuParam + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", String::from(self))
    }
}

impl<X, T> Rv<X> for DiscreteUniform<T>
where
    T: DuParam + SampleUniform + Copy,
    X: Integer + From<T>,
{
    fn ln_f(&self, x: &X) -> f64 {
        if *x >= X::from(self.a) && *x <= X::from(self.b) {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> X {
        let d = rand::distributions::Uniform::new_inclusive(self.a, self.b);
        X::from(rng.sample(d))
    }

    fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<X> {
        let d = rand::distributions::Uniform::new_inclusive(self.a, self.b);
        rng.sample_iter(&d).take(n).map(X::from).collect()
    }
}

impl<X, T> Support<X> for DiscreteUniform<T>
where
    X: Integer + From<T>,
    T: DuParam,
{
    fn supports(&self, x: &X) -> bool {
        X::from(self.a) <= *x && X::from(self.b) >= *x
    }
}

impl<X, T> DiscreteDistr<X> for DiscreteUniform<T>
where
    X: Integer + From<T>,
    T: DuParam + SampleUniform + Into<f64>,
{
}

impl<T> Entropy for DiscreteUniform<T>
where
    T: DuParam + Into<f64>,
{
    fn entropy(&self) -> f64 {
        let diff: f64 = (self.b - self.a).into();
        diff.ln()
    }
}

impl<T> Mean<f64> for DiscreteUniform<T>
where
    T: DuParam + SampleUniform + Into<f64>,
{
    fn mean(&self) -> Option<f64> {
        let m = ((self.b + self.a).into()) / 2.0;
        Some(m)
    }
}

impl<T> Median<f64> for DiscreteUniform<T>
where
    T: DuParam + SampleUniform + Into<f64>,
{
    fn median(&self) -> Option<f64> {
        let m: f64 = (self.b + self.a).into() / 2.0;
        Some(m)
    }
}

impl<T> Variance<f64> for DiscreteUniform<T>
where
    T: DuParam + SampleUniform + Into<f64>,
{
    fn variance(&self) -> Option<f64> {
        let v = (self.b - self.a + T::one()).into()
            * (self.b - self.a + T::one()).into()
            / 12.0;
        Some(v)
    }
}

impl<X, T> Cdf<X> for DiscreteUniform<T>
where
    X: Integer + From<T> + ToPrimitive + Copy,
    T: DuParam + SampleUniform + ToPrimitive,
{
    fn cdf(&self, x: &X) -> f64 {
        if *x < X::from(self.a) {
            0.0
        } else if *x >= X::from(self.b) {
            1.0
        } else {
            let xf: f64 = (*x).to_f64().unwrap();
            let a: f64 = self.a.to_f64().unwrap();
            let b: f64 = self.b.to_f64().unwrap();

            (xf - a + 1.0) / (b - a + 1.0)
        }
    }
}

impl<X, T> InverseCdf<X> for DiscreteUniform<T>
where
    X: Integer + From<T> + FromPrimitive,
    T: DuParam + SampleUniform + ToPrimitive,
{
    fn invcdf(&self, p: f64) -> X {
        let diff: f64 = (self.b - self.a).to_f64().unwrap();
        X::from_f64(p * diff).unwrap() + X::from(self.a)
    }
}

impl<T: DuParam> Skewness for DiscreteUniform<T> {
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl<T: DuParam> Kurtosis for DiscreteUniform<T> {
    fn kurtosis(&self) -> Option<f64> {
        Some(-1.2)
    }
}

impl std::error::Error for DiscreteUniformError {}

impl fmt::Display for DiscreteUniformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInterval => {
                write!(f, "a (lower) is greater than or equal to b (upper)")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!([count] DiscreteUniform::new(0_u32, 10_u32).unwrap());

    #[test]
    fn new() {
        let u = DiscreteUniform::new(0, 10).unwrap();
        assert!(u.a == 0);
        assert!(u.b == 10);
    }

    #[test]
    fn new_rejects_a_equal_to_b() {
        assert!(DiscreteUniform::new(5, 5).is_err());
    }

    #[test]
    fn new_rejects_a_gt_b() {
        assert!(DiscreteUniform::new(5, 1).is_err());
    }

    #[test]
    fn mean() {
        let m: f64 = DiscreteUniform::new(0, 10).unwrap().mean().unwrap();
        assert::close(m, 5.0, TOL);
    }

    #[test]
    fn median() {
        let m: f64 = DiscreteUniform::new(0, 10).unwrap().median().unwrap();
        assert::close(m, 5.0, TOL);
    }

    #[test]
    fn variance() {
        let v: f64 = DiscreteUniform::new(0, 10).unwrap().variance().unwrap();
        assert::close(v, (11.0 * 11.0) / 12.0, TOL);
    }

    #[test]
    fn entropy() {
        let h: f64 = DiscreteUniform::new(2, 4).unwrap().entropy();
        assert::close(h, 0.693_147_180_559_9, TOL);
    }

    #[test]
    fn ln_pmf() {
        let u = DiscreteUniform::new(0, 10).unwrap();
        assert::close(u.ln_pmf(&2_u8), 0.0, TOL);
    }
    #[test]
    fn cdf() {
        let u = DiscreteUniform::new(0_u32, 10_u32).unwrap();
        assert::close(u.cdf(&0_u32), 1.0 / 11.0, TOL);
        assert::close(u.cdf(&5_u32), 6.0 / 11.0, TOL);
        assert::close(u.cdf(&10_u32), 1.0, TOL);
    }

    #[test]
    fn cdf_inv_cdf_ident() {
        let mut rng = rand::thread_rng();
        let ru = rand::distributions::Uniform::new_inclusive(0_u32, 100_u32);
        let u = DiscreteUniform::new(0_u32, 100_u32).unwrap();
        for _ in 0..100 {
            let x: u32 = rng.sample(ru);
            let cdf = u.cdf(&x);
            let y: u32 = u.invcdf(cdf);
            assert!(x == y);
        }
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let u = DiscreteUniform::new(0_u32, 100_u32).unwrap();
        let cdf = |x: u64| u.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<u64> = u.sample(1000, &mut rng);
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
