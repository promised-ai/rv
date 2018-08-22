//! Binomial distribution
extern crate rand;
extern crate special;

use self::rand::Rng;
use misc::ln_binom;
use std::f64;
use std::io;
use traits::*;

/// [Binomial distribution](https://en.wikipedia.org/wiki/Beta-binomial_distribution)
/// with success probability *p*
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Binomial {
    /// Total number of trials
    pub n: u64,
    /// Probability of a success
    pub p: f64,
}

impl Binomial {
    pub fn new(n: u64, p: f64) -> io::Result<Self> {
        let p_ok = p.is_finite() && 0.0 < p && p < 1.0;
        let n_ok = n > 0;
        if !p_ok {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "p must be in [0, 1]");
            Err(err)
        } else if !n_ok {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "n must be > 0");
            Err(err)
        } else {
            Ok(Binomial { n, p })
        }
    }

    /// A Binomial distribution with a 50% chance of success
    pub fn uniform(n: u64) -> Self {
        Binomial::new(n, 0.5).unwrap()
    }

    /// The complement of `p`, i.e. `(1 - p)`.
    #[inline]
    pub fn q(&self) -> f64 {
        1.0 - self.p
    }
}

macro_rules! impl_int_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Binomial {
            fn ln_f(&self, k: &$kind) -> f64 {
                let nf = self.n as f64;
                let kf = *k as f64;
                ln_binom(nf, kf) + self.p.ln() * kf + self.q().ln() * (nf - kf)
            }

            // XXX: Opportunity for optimization in `sample`. Sometime in the
            // future, we should do some criterion benchmarks to test when it
            // is faster to draw using alias tables or some other method.
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                // TODO: This is really awful.
                let b = rand::distributions::Binomial::new(self.n, self.p);
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
        Some((1.0 - 2.0 * self.p) / (nf * self.p * self.q()).sqrt())
    }
}

impl Kurtosis for Binomial {
    fn kurtosis(&self) -> Option<f64> {
        let q = self.q();
        let nf = self.n as f64;
        Some((1.0 - 6.0 * self.p * q) / (nf * self.p * q))
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
            -9.16290731874155,
            -6.454857117639339,
            -4.545314612754902,
            -3.1590202516350105,
            -2.1939393555914233,
            -1.6061526906893038,
            -1.383009139375095,
            -1.5371598192023535,
            -2.1125239641059164,
            -3.2111362527740246,
            -5.108256237659907,
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
            0.00010485760000000006,
            0.0016777216000000005,
            0.012294553600000008,
            0.05476188160000002,
            0.1662386176,
            0.36689674240000003,
            0.6177193983999999,
            0.8327102464,
            0.9536425984,
            0.9939533824,
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
        assert::close(k1, -0.1833333333333333, TOL);
        assert::close(k2, 0.0013203593673756242, TOL);
    }

    #[test]
    fn skewness() {
        let s1 = Binomial::new(10, 0.6).unwrap().skewness().unwrap();
        let s2 = Binomial::new(21, 0.21).unwrap().skewness().unwrap();
        assert::close(s1, -0.12909944487358052, TOL);
        assert::close(s2, 0.3107385631129019, TOL);
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
        assert::close(v1, 2.4000000000000004, TOL);
        assert::close(v2, 3.4839, TOL);
    }

    #[test]
    fn sample_test() {
        let mut rng = rand::thread_rng();
        let binom = Binomial::new(5, 0.6).unwrap();
        let ps: Vec<f64> = vec![
            0.010240000000000008,
            0.07680000000000001,
            0.2304,
            0.3455999999999999,
            0.2591999999999999,
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
            0.010240000000000008,
            0.07680000000000001,
            0.2304,
            0.3455999999999999,
            0.2591999999999999,
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