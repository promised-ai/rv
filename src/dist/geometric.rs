//! Possion distribution on unisgned integers
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::dist::Uniform;
use crate::impl_display;
use crate::result;
use crate::traits::*;
use num::{Bounded, FromPrimitive, Integer, Saturating, ToPrimitive, Unsigned};
use rand::Rng;

/// [Geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution)
/// over x in {0, 1, 2, 3, ... }.
///
/// # Example
///
/// ```
/// extern crate rv;
/// extern crate rand;
///
/// use rv::prelude::*;
///
/// // Create Geometric(p=0.5)
/// let geom = Geometric::new(0.5).unwrap();
///
/// // Draw Samples
/// let mut rng = rand::thread_rng();
/// let xs: Vec<u32> = geom.sample(100, &mut rng);
/// assert_eq!(xs.len(), 100)
/// ```
#[derive(Debug, Clone, PartialOrd, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Geometric {
    pub p: f64,
}

impl Geometric {
    pub fn new(p: f64) -> result::Result<Self> {
        if p > 0.0 && p <= 1.0 {
            Ok(Geometric { p })
        } else {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let err = result::Error::new(
                err_kind,
                "p must be between zero and one (right closed).",
            );
            Err(err)
        }
    }

    // Use the inversion method to select the corresponding integer.
    fn inversion_draw_method<X, R>(rng: &mut R, p: f64) -> X
    where
        X: Unsigned + Integer + FromPrimitive + Bounded,
        R: Rng,
    {
        let u: f64 = Uniform::new(0.0, 1.0).unwrap().draw(rng);
        X::from_f64(((1.0 - u).ln() / (1.0 - p).ln()).ceil() - 1.0)
            .unwrap_or(X::max_value())
    }

    // Increase the value until the cdf surpasses the given p value.
    fn search_draw_method<X, R>(rng: &mut R, p: f64) -> X
    where
        X: Unsigned + Integer + Saturating,
        R: Rng,
    {
        let u: f64 = rng.gen();
        let q = 1.0 - p;

        let mut t: X = X::zero();
        let mut sum = p;
        let mut prod = p;

        while u > sum {
            prod *= q;
            sum += prod;
            t = t.saturating_add(X::one());
        }
        t
    }
}

impl Default for Geometric {
    fn default() -> Self {
        Geometric { p: 0.5 }
    }
}

impl From<&Geometric> for String {
    fn from(geom: &Geometric) -> String {
        format!("G(p: {})", geom.p)
    }
}

impl_display!(Geometric);

impl<X> Rv<X> for Geometric
where
    X: Unsigned + Integer + FromPrimitive + ToPrimitive + Saturating + Bounded,
{
    fn ln_f(&self, k: &X) -> f64 {
        let kf = (*k).to_f64().unwrap();
        kf * (1.0 - self.p).ln() + self.p.ln()
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> X {
        // Follows the same pattern as
        // https://github.com/numpy/numpy/blob/7c41164f5340dc998ea1c04d2061f7d246894955/numpy/random/mtrand/distributions.c#L777
        if 3.0 * self.p > 1.0 {
            Geometric::search_draw_method(rng, self.p)
        } else {
            Geometric::inversion_draw_method(rng, self.p)
        }
    }
}

impl<X> Support<X> for Geometric
where
    X: Unsigned + Integer,
{
    #[allow(unused_comparisons)]
    fn supports(&self, k: &X) -> bool {
        *k >= X::zero()
    }
}

impl<X> DiscreteDistr<X> for Geometric where
    X: Unsigned + Integer + FromPrimitive + ToPrimitive + Saturating + Bounded
{
}

impl<X> Cdf<X> for Geometric
where
    X: Unsigned + Integer + FromPrimitive + ToPrimitive + Saturating + Bounded,
{
    fn cdf(&self, k: &X) -> f64 {
        let kf = (*k).to_f64().unwrap();
        1.0 - (1.0 - self.p).powf(kf + 1.0)
    }
}

impl Mean<f64> for Geometric {
    fn mean(&self) -> Option<f64> {
        Some((1.0 - self.p) / self.p)
    }
}

impl Variance<f64> for Geometric {
    fn variance(&self) -> Option<f64> {
        Some((1.0 - self.p) / (self.p * self.p))
    }
}

impl Skewness for Geometric {
    fn skewness(&self) -> Option<f64> {
        Some((2.0 - self.p) / (1.0 - self.p).sqrt())
    }
}

impl Kurtosis for Geometric {
    fn kurtosis(&self) -> Option<f64> {
        Some(6.0 + (self.p * self.p) / (1.0 - self.p))
    }
}

impl Entropy for Geometric {
    fn entropy(&self) -> f64 {
        (-(1.0 - self.p) * (1.0 - self.p).log2() - self.p * self.p.log2())
            / self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::x2_test;
    use std::f64;

    const TOL: f64 = 1E-12;
    const N_TRIES: usize = 5;
    const X2_PVAL: f64 = 0.2;

    #[test]
    fn new() {
        assert::close(Geometric::new(0.001).unwrap().p, 0.001, TOL);
        assert::close(Geometric::new(1.0).unwrap().p, 1.00, TOL);
    }

    #[test]
    fn new_should_reject_non_finite_rate() {
        assert!(Geometric::new(f64::INFINITY).is_err());
        assert!(Geometric::new(f64::NAN).is_err());
        assert!(Geometric::new(1.1).is_err());
    }

    #[test]
    fn new_should_reject_rate_lteq_zero() {
        assert!(Geometric::new(0.0).is_err());
        assert!(Geometric::new(-1E-12).is_err());
        assert!(Geometric::new(-1E12).is_err());
        assert!(Geometric::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn ln_pdf() {
        let geom = Geometric::new(0.5).unwrap();
        assert::close(geom.ln_pmf(&0_u32), -0.6931471805599453, TOL);
        assert::close(geom.ln_pmf(&1_u32), -1.3862943611198906, TOL);
        assert::close(geom.ln_pmf(&5_u32), -4.1588830833596715, TOL);
        assert::close(geom.ln_pmf(&11_u32), -8.317766166719343, TOL);
    }

    #[test]
    fn cdf() {
        let geom = Geometric::new(0.5).unwrap();
        assert::close(geom.cdf(&0_u32), 0.5, TOL);
        assert::close(geom.cdf(&1_u32), 0.75, TOL);
        assert::close(geom.cdf(&3_u32), 0.9375, TOL);
        assert::close(geom.cdf(&5_u32), 0.984375, TOL);
    }

    #[test]
    fn mean() {
        let m1 = Geometric::new(0.1).unwrap().mean().unwrap();
        assert::close(m1, 9.0, TOL);

        let m2 = Geometric::new(0.5).unwrap().mean().unwrap();
        assert::close(m2, 1.0, TOL);

        let m3 = Geometric::new(0.9).unwrap().mean().unwrap();
        assert::close(m3, 0.111111111111111, TOL);
    }

    #[test]
    fn variance() {
        let v1 = Geometric::new(0.1).unwrap().variance().unwrap();
        assert::close(v1, 90.0, TOL);

        let v2 = Geometric::new(0.5).unwrap().variance().unwrap();
        assert::close(v2, 2.0, TOL);

        let v3 = Geometric::new(0.9).unwrap().variance().unwrap();
        assert::close(v3, 0.12345679012345676, TOL);
    }

    #[test]
    fn skewness() {
        let s = Geometric::new(0.5).unwrap().skewness().unwrap();
        assert::close(s, 2.12132034355964257, TOL);
    }

    #[test]
    fn kurtosis() {
        let k = Geometric::new(0.5).unwrap().kurtosis().unwrap();
        assert::close(k, 6.5, TOL);
    }

    fn test_draw_generic(p: f64) {
        let mut rng = rand::thread_rng();
        let geom = Geometric::new(p).unwrap();

        // How many bins do we need?
        let k: usize = (0..100)
            .position(|x| geom.pmf(&(x as u32)) < f64::EPSILON)
            .unwrap_or(99)
            + 1;

        let ps: Vec<f64> = (0..k).map(|x| geom.pmf(&(x as u32))).collect();

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; k];
            let xs: Vec<u32> = geom.sample(1000, &mut rng);
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
    fn draw_test_05() {
        test_draw_generic(0.5)
    }

    #[test]
    fn draw_test_02() {
        test_draw_generic(0.2)
    }
}
