//! Binomial distribution of x in {0, 1}
extern crate rand;
extern crate special;
extern crate num;

use self::num::integer::binomial as choose;

use self::rand::distributions::{Binomial as RBinomial, Distribution};
use self::special::Beta;
use consts;
use self::rand::Rng;
use std::f64;
use std::io;
use traits::*;

/// [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
/// with success probability *p* and trials *n*
///
/// # Example
///
/// ```
/// # extern crate rv;
/// use rv::prelude::*;
///
/// let b = Binomial::new(0.5, 256).unwrap();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Binomial {
    /// Probability of a success for a single Bernoulli trial.
    pub p: f64,
    /// Number of trials
    pub n: u64,
}

impl Binomial {
    pub fn new(p: f64, n: u64) -> io::Result<Self> {
        if p.is_finite() && 0.0 < p && p < 1.0 {
            Ok(Binomial { p, n })
        } else {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "p must be in [0, 1]");
            Err(err)
        }
    }
    
    #[inline]
    pub fn q(&self) -> f64 {
        1.0 - self.p
    }
}

macro_rules! impl_int_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Binomial {
            fn ln_f(&self, x: &$kind) -> f64 {
                let nf = self.n as f64;
                let xf = *x as f64;
                let bin_coeff = if *x == 0 || xf == nf {
                    0.0
                } else if self.n <= 17 { // TODO make this dynamic
                    (choose(self.n, *x as u64) as f64).ln()
                } else {
                    // Use stirling's approximation for log(choose(n, k))
                    nf * nf.ln() - xf * xf.ln() - (nf - xf) * (nf - xf).ln() + 0.5 * (nf.ln() - xf.ln() - (nf - xf).ln() - consts::LN_2PI)
                };

                bin_coeff + self.p.ln() * xf + (self.q()).ln() * (nf - xf)
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let b = RBinomial::new(self.n, self.p);
                b.sample(rng) as $kind
            }

            fn sample<R: Rng>(&self, n:usize, rng: &mut R) -> Vec<$kind> {
                let b = RBinomial::new(self.n, self.p);
                (0..n).map(|_| b.sample(rng) as $kind).collect()
            }
        }

        impl Support<$kind> for Binomial {
            fn contains(&self, x: &$kind) -> bool {
                *x <= (self.n as $kind)
            }
        }

        impl DiscreteDistr<$kind> for Binomial {
            fn pmf(&self, x: &$kind) -> f64 {
                self.f(x)
            }

            fn ln_pmf(&self, x: &$kind) -> f64 {
                self.ln_f(x)
            }
        }

        impl Cdf<$kind> for Binomial {
            fn cdf(&self, k: &$kind) -> f64 {
                let kk = *k as f64;
                let nf = self.n as f64;
                let ln_beta = (nf - kk).ln_beta(kk + 1.0);
                (self.q()).inc_beta(nf - kk, kk + 1.0, ln_beta)
            }
        }

        impl Mode<$kind> for Binomial {
            fn mode(&self) -> Option<$kind> {
                Some((((self.n + 1) as f64) * self.p).floor() as $kind)
            }
        }
    };
}

impl Entropy for Binomial {
    fn entropy(&self) -> f64 {
        consts::HALF_LN_2PI_E + (self.p * (self.n as f64) * (self.q())).ln()
    }
}

impl Skewness for Binomial {
    fn skewness(&self) -> Option<f64> {
        Some((1.0 - 2.0 * self.p) / ((self.n as f64) * self.p * (self.q())).sqrt())
    }
}

impl Kurtosis for Binomial {
    fn kurtosis(&self) -> Option<f64> {
        Some(1.0 * 6.0 * self.p * self.q() / ((self.n as f64) * self.p * self.q()))
    }
}

impl Mean<f64> for Binomial {
    fn mean(&self) -> Option<f64> {
        Some(self.p * (self.n as f64))
    }
}

impl Median<f64> for Binomial {
    fn median(&self) -> Option<f64> {
        Some((self.p * (self.n as f64)).floor())
    }
}

impl Variance<f64> for Binomial {
    fn variance(&self) -> Option<f64> {
        let n = self.n as f64;
        Some(n * self.p * self.q())
    }
}

impl_int_traits!(u8);
impl_int_traits!(u16);
impl_int_traits!(u32);
impl_int_traits!(u64);

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
        let b: Binomial = Binomial::new(0.5, 256).unwrap();
        assert::close(b.p, 0.5, TOL);
        assert_eq!(b.n, 256);
        assert::close(b.q(), 0.5, TOL);
    }

    #[test]
    fn draw_test_low() {
        let k = 5;
        let mut rng = rand::thread_rng();
        let b: Binomial = Binomial::new(0.5, k).unwrap();
        let ps: Vec<f64> = (0..k+1).map(|x| b.pmf(&(x as u32))).collect();

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; (k + 1) as usize];
            let xs: Vec<u32> = b.sample(1000, &mut rng);
            xs.iter().for_each(|&x| f_obs[x as usize] += 1);
            println!("{:#?}", f_obs);
            println!("{:#?}", ps);

            let (_, p) =  x2_test(f_obs.as_slice(), ps.as_slice());

            if p > X2_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }

    #[test]
    fn draw_test_high() {
        let k = 200;
        let mut rng = rand::thread_rng();
        let b: Binomial = Binomial::new(0.5, k).unwrap();
        let ps: Vec<f64> = (0..k+1).map(|x| b.pmf(&(x as u32))).collect();

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; (k + 1) as usize];
            let xs: Vec<u32> = b.sample(1000, &mut rng);
            xs.iter().for_each(|&x| f_obs[x as usize] += 1);
            println!("{:#?}", f_obs);
            println!("{:#?}", ps);

            let (_, p) =  x2_test(f_obs.as_slice(), ps.as_slice());

            if p > X2_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }



}
