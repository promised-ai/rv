//! Continuous uniform distribution, U(a, b) on the interval x in [a, b]
extern crate num;
extern crate rand;

use self::num::{FromPrimitive, Integer, ToPrimitive};
use self::rand::distributions::uniform::SampleUniform;
use self::rand::Rng;
use std::f64;

use crate::result::*;
use crate::traits::*;

#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

/// [Discrete uniform distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution),
/// U(a, b) on the interval x in [a, b]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct DiscreteUniform<T: Integer> {
    pub a: T,
    pub b: T,
}

impl<T: Integer> DiscreteUniform<T> {
    pub fn new(a: T, b: T) -> Result<Self> {
        if a < b {
            Ok(Self { a, b })
        } else {
            Err(Error::new(
                ErrorKind::InvalidParameterError,
                "a must be less than b",
            ))
        }
    }
}

impl<X, T> Rv<X> for DiscreteUniform<T>
where
    T: Integer + SampleUniform + Copy,
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
        rng.sample_iter(&d).take(n).map(|t| X::from(t)).collect()
    }
}

impl<X, T> Support<X> for DiscreteUniform<T>
where
    X: Integer + From<T>,
    T: Integer + Copy,
{
    fn supports(&self, x: &X) -> bool {
        X::from(self.a) <= *x && X::from(self.b) >= *x
    }
}

impl<X, T> DiscreteDistr<X> for DiscreteUniform<T>
where
    X: Integer + From<T>,
    T: Integer + SampleUniform + Into<f64> + Copy,
{
}

impl<T> Entropy for DiscreteUniform<T>
where
    T: Integer + Into<f64> + Copy,
{
    fn entropy(&self) -> f64 {
        let diff: f64 = (self.b - self.a).into();
        diff.ln()
    }
}

impl<T> Mean<f64> for DiscreteUniform<T>
where
    T: Integer + SampleUniform + Into<f64> + Copy,
{
    fn mean(&self) -> Option<f64> {
        let m = ((self.b + self.a).into()) / 2.0;
        Some(m)
    }
}

impl<T> Median<f64> for DiscreteUniform<T>
where
    T: Integer + SampleUniform + Into<f64> + Copy,
{
    fn median(&self) -> Option<f64> {
        let m: f64 = (self.b + self.a).into() / 2.0;
        Some(m)
    }
}

impl<T> Variance<f64> for DiscreteUniform<T>
where
    T: Integer + SampleUniform + Into<f64> + Copy,
{
    fn variance(&self) -> Option<f64> {
        let v = (self.b - self.a + T::one()).into().powi(2) / 12.0;
        Some(v)
    }
}

impl<X, T> Cdf<X> for DiscreteUniform<T>
where
    X: Integer + From<T> + ToPrimitive + Copy,
    T: Integer + SampleUniform + ToPrimitive + Copy,
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
    T: Integer + SampleUniform + ToPrimitive + Copy,
{
    fn invcdf(&self, p: f64) -> X {
        let diff: f64 = (self.b - self.a).to_f64().unwrap();
        X::from_f64(p * diff).unwrap() + X::from(self.a)
    }
}

impl<T: Integer> Skewness for DiscreteUniform<T> {
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl<T: Integer> Kurtosis for DiscreteUniform<T> {
    fn kurtosis(&self) -> Option<f64> {
        Some(-1.2)
    }
}

#[cfg(test)]
mod tests {
    extern crate assert;
    use super::*;
    use crate::misc::ks_test;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

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
        assert::close(h, 0.6931471805599, TOL);
    }

    #[test]
    fn ln_pmf() {
        let u = DiscreteUniform::new(0, 10).unwrap();
        assert::close(u.ln_pmf(&2u8), 0.0, TOL);
    }
    #[test]
    fn cdf() {
        let u = DiscreteUniform::new(0u32, 10u32).unwrap();
        assert::close(u.cdf(&0u32), 1.0 / 11.0, TOL);
        assert::close(u.cdf(&5u32), 6.0 / 11.0, TOL);
        assert::close(u.cdf(&10u32), 1.0, TOL);
    }

    #[test]
    fn cdf_inv_cdf_ident() {
        let mut rng = rand::thread_rng();
        let ru = rand::distributions::Uniform::new_inclusive(0u32, 100u32);
        let u = DiscreteUniform::new(0u32, 100u32).unwrap();
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
