//! Bernoulli distribution of x in {0, 1}
extern crate rand;
extern crate special;

use self::rand::distributions::Uniform;
use self::rand::Rng;
use data::BernoulliSuffStat;
use std::f64;
use std::io;
use traits::*;

/// Bernoulli distribution with success probability *p*
///
/// # Examples
///
/// ```
/// # extern crate rv;
/// use rv::prelude::*;
///
/// let b = Bernoulli::new(0.75).unwrap();
/// assert!((b.pmf(&true) - 0.75).abs() < 1E-12);
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Bernoulli {
    /// Probability of a success (x=1)
    pub p: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> io::Result<Self> {
        if p.is_finite() && 0.0 < p && p < 1.0 {
            Ok(Bernoulli { p })
        } else {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "p must be in [0, 1]");
            Err(err)
        }
    }

    /// A Bernoulli distribution with a 50% chance of success
    pub fn uniform() -> Self {
        Bernoulli::new(0.5).unwrap()
    }

    /// The complement of `p`, i.e. `(1 - p)`.
    #[inline]
    pub fn q(&self) -> f64 {
        1.0 - self.p
    }
}

impl Default for Bernoulli {
    fn default() -> Self {
        Bernoulli::uniform()
    }
}

macro_rules! impl_int_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Bernoulli {
            fn f(&self, x: &$kind) -> f64 {
                if *x == 1 {
                    self.p
                } else {
                    1.0_f64 - self.p
                }
            }

            fn ln_f(&self, x: &$kind) -> f64 {
                self.f(x).ln()
            }

            fn ln_normalizer(&self) -> f64 {
                0.0
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let u = Uniform::new(0.0, 1.0);
                if rng.sample(u) < self.p {
                    1
                } else {
                    0
                }
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let u = Uniform::new(0.0, 1.0);
                (0..n)
                    .map(|_| if rng.sample(u) < self.p { 1 } else { 0 })
                    .collect()
            }
        }

        impl Support<$kind> for Bernoulli {
            fn contains(&self, x: &$kind) -> bool {
                *x == 0 || *x == 1
            }
        }

        impl DiscreteDistr<$kind> for Bernoulli {
            fn pmf(&self, x: &$kind) -> f64 {
                self.f(x)
            }

            fn ln_pmf(&self, x: &$kind) -> f64 {
                self.ln_f(x)
            }
        }

        impl Cdf<$kind> for Bernoulli {
            fn cdf(&self, x: &$kind) -> f64 {
                if *x == 0 {
                    self.q()
                } else if *x > 0 {
                    1.0
                } else {
                    0.0
                }
            }
        }

        impl Mode<$kind> for Bernoulli {
            fn mode(&self) -> Option<$kind> {
                let q = self.q();
                if self.p < q {
                    Some(0)
                } else if (self.p - q).abs() < f64::EPSILON {
                    None
                } else {
                    Some(1)
                }
            }
        }

        impl HasSuffStat<$kind> for Bernoulli {
            type Stat = BernoulliSuffStat;
            fn empty_suffstat(&self) -> Self::Stat {
                BernoulliSuffStat::new()
            }
        }
    };
}

impl KlDivergence for Bernoulli {
    fn kl(&self, other: &Self) -> f64 {
        self.p * (other.p.ln() - self.p.ln())
            + self.q() * (other.q().ln() - self.q().ln())
    }
}

impl Entropy for Bernoulli {
    fn entropy(&self) -> f64 {
        let q = self.q();
        -q * q.ln() - self.p * self.p.ln()
    }
}

impl Skewness for Bernoulli {
    fn skewness(&self) -> Option<f64> {
        Some((1.0 - 2.0 * self.p) / (self.p * self.q()).sqrt())
    }
}

impl Kurtosis for Bernoulli {
    fn kurtosis(&self) -> Option<f64> {
        let q = self.q();
        Some((1.0 - 6.0 * self.p * q) / (self.p * q))
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

impl Rv<bool> for Bernoulli {
    fn f(&self, x: &bool) -> f64 {
        if *x {
            self.p
        } else {
            1.0_f64 - self.p
        }
    }

    fn ln_f(&self, x: &bool) -> f64 {
        self.f(x).ln()
    }

    fn ln_normalizer(&self) -> f64 {
        0.0
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> bool {
        let u = Uniform::new(0.0, 1.0);
        rng.sample(u) < self.p
    }

    fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<bool> {
        let u = Uniform::new(0.0, 1.0);
        (0..n).map(|_| rng.sample(u) < self.p).collect()
    }
}

impl Support<bool> for Bernoulli {
    fn contains(&self, _x: &bool) -> bool {
        true
    }
}

impl DiscreteDistr<bool> for Bernoulli {
    fn pmf(&self, x: &bool) -> f64 {
        self.f(x)
    }

    fn ln_pmf(&self, x: &bool) -> f64 {
        self.ln_f(x)
    }
}

impl Cdf<bool> for Bernoulli {
    fn cdf(&self, x: &bool) -> f64 {
        if *x {
            1.0
        } else {
            self.q()
        }
    }
}

impl Mode<bool> for Bernoulli {
    fn mode(&self) -> Option<bool> {
        let q = self.q();
        if self.p < q {
            Some(false)
        } else if (self.p - q).abs() < f64::EPSILON {
            None
        } else {
            Some(true)
        }
    }
}

impl HasSuffStat<bool> for Bernoulli {
    type Stat = BernoulliSuffStat;
    fn empty_suffstat(&self) -> Self::Stat {
        BernoulliSuffStat::new()
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
impl_int_traits!(isize);

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
        let b: Bernoulli = Bernoulli::new(0.1).unwrap();
        assert::close(b.p, 0.1, TOL);
    }

    #[test]
    fn new_should_reject_oob_p() {
        assert!(Bernoulli::new(0.0).is_err());
        assert!(Bernoulli::new(1.0).is_err());
        assert!(Bernoulli::new(-0.001).is_err());
        assert!(Bernoulli::new(1.001).is_err());
    }

    #[test]
    fn new_should_reject_non_finite_p() {
        assert!(Bernoulli::new(f64::NAN).is_err());
        assert!(Bernoulli::new(f64::INFINITY).is_err());
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
        assert!(b.contains(&true));
        assert!(b.contains(&false));
    }

    #[test]
    fn contains_both_zero_and_one() {
        let b = Bernoulli::uniform();
        assert!(b.contains(&0));
        assert!(b.contains(&1));
        assert!(!b.contains(&-1));
        assert!(!b.contains(&2));
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
    fn cmf_less_than_zero_is_zero() {
        let b = Bernoulli::new(0.1).unwrap();
        assert::close(b.cdf(&-1_i16), 0.0, TOL);
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
        assert::close(b1.entropy(), 0.3250829733914482, TOL);
        assert::close(b2.entropy(), 0.3250829733914482, TOL);
    }

    #[test]
    fn unifrom_entropy() {
        let b = Bernoulli::uniform();
        assert::close(b.entropy(), 0.6931471805599453, TOL);
    }

    #[test]
    fn uniform_skewness_should_be_zero() {
        let b = Bernoulli::uniform();
        assert::close(b.skewness().unwrap(), 0.0, TOL);
    }

    #[test]
    fn skewness() {
        let b = Bernoulli::new(0.3).unwrap();
        assert::close(b.skewness().unwrap(), 0.8728715609439696, TOL);
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
}
