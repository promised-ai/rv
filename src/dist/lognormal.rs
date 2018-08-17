//! Log Normal Distribution over x in (0, ∞)

extern crate rand;
extern crate special;

use std::f64::consts::SQRT_2;
use std::io;

use self::rand::distributions::Normal;
use self::rand::Rng;
use self::special::Error;

use consts::*;
// use data::LogNormalSuffStat;
use traits::*;

/// [LogNormal Distribution](https://en.wikipedia.org/wiki/Log-normal_distribution)
/// If x ~ Normal(μ, σ), then e^x ~ LogNormal(μ, σ).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct LogNormal {
    /// log scale mean
    pub mu: f64,
    /// log scale standard deviation
    pub sigma: f64,
}

impl LogNormal {
    pub fn new(mu: f64, sigma: f64) -> io::Result<Self> {
        let mu_ok = mu.is_finite();
        let sigma_ok = sigma > 0.0 && sigma.is_finite();

        if mu_ok && sigma_ok {
            Ok(LogNormal { mu, sigma })
        } else {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "mu must be finite and sigma must be finite and greater than zero.");
            Err(err)
        }
    }

    pub fn standard() -> Self {
        LogNormal {
            mu: 0.0,
            sigma: 1.0,
        }
    }
}

impl Default for LogNormal {
    fn default() -> Self {
        LogNormal::standard()
    }
}

macro_rules! impl_traits {
    ($kind: ty) => {
        impl Rv<$kind> for LogNormal {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xk = f64::from(*x);
                let xk_ln = xk.ln();
                let d = (xk_ln - self.mu) / self.sigma;
                -xk_ln - self.sigma.ln() - HALF_LN_2PI - 0.5 * d * d
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let g = Normal::new(self.mu, self.sigma);
                rng.sample(g).exp() as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let g = Normal::new(self.mu, self.sigma);
                (0..n).map(|_| rng.sample(g).exp() as $kind).collect()
            }
        }

        impl ContinuousDistr<$kind> for LogNormal {}

        impl Support<$kind> for LogNormal {
            fn supports(&self, x: &$kind) -> bool {
                *x > 0.0 && x.is_finite()
            }
        }

        impl Cdf<$kind> for LogNormal {
            fn cdf(&self, x: &$kind) -> f64 {
                let xk = f64::from(*x);
                0.5 + 0.5 * ((xk.ln() - self.mu) / (SQRT_2 * self.sigma)).erf()
            }
        }

        impl InverseCdf<$kind> for LogNormal {
            fn invcdf(&self, p: f64) -> $kind {
                (self.mu + SQRT_2 * self.sigma * (2.0 * p - 1.0).inv_erf())
                    .exp() as $kind
            }
        }

        impl Mean<$kind> for LogNormal {
            fn mean(&self) -> Option<$kind> {
                Some((self.mu + self.sigma * self.sigma / 2.0).exp() as $kind)
            }
        }

        impl Median<$kind> for LogNormal {
            fn median(&self) -> Option<$kind> {
                Some(self.mu.exp() as $kind)
            }
        }

        impl Mode<$kind> for LogNormal {
            fn mode(&self) -> Option<$kind> {
                Some((self.mu - self.sigma * self.sigma) as $kind)
            }
        }
    };
}

impl Variance<f64> for LogNormal {
    fn variance(&self) -> Option<f64> {
        Some(
            ((self.sigma * self.sigma).exp() - 1.0)
                * (2.0 * self.mu + self.sigma * self.sigma).exp(),
        )
    }
}

impl Entropy for LogNormal {
    fn entropy(&self) -> f64 {
        (self.mu + 0.5) + self.sigma.ln() + HALF_LN_2PI
    }
}

impl Skewness for LogNormal {
    fn skewness(&self) -> Option<f64> {
        let e_sigma_2 = (self.sigma * self.sigma).exp();
        Some((e_sigma_2 + 2.0) * (e_sigma_2 - 1.0).sqrt())
    }
}

impl Kurtosis for LogNormal {
    fn kurtosis(&self) -> Option<f64> {
        let s2 = self.sigma * self.sigma;
        Some(
            (4.0 * s2).exp() + 2.0 * (3.0 * s2).exp() + 3.0 * (2.0 * s2).exp()
                - 6.0,
        )
    }
}

impl_traits!(f32);
impl_traits!(f64);

#[cfg(test)]
mod tests {
    extern crate assert;
    use super::*;
    use std::f64;

    const TOL: f64 = 1E-12;

    #[test]
    fn new() {
        let lognorm = LogNormal::new(1.2, 3.0).unwrap();
        assert::close(lognorm.mu, 1.2, TOL);
        assert::close(lognorm.sigma, 3.0, TOL);
    }

    #[test]
    fn mean() {
        let mu = 3.0;
        let sigma = 2.0;
        let mean: f64 = LogNormal::new(mu, sigma).unwrap().mean().unwrap();
        assert::close(mean, 5.0_f64.exp(), TOL);
    }

    #[test]
    fn median_should_be_exp_mu() {
        let mu = 3.4;
        let median: f64 = LogNormal::new(mu, 0.5).unwrap().median().unwrap();
        assert::close(median, mu.exp(), TOL);
    }

    #[test]
    fn mode() {
        let mode: f64 = LogNormal::new(4.0, 2.0).unwrap().mode().unwrap();
        assert::close(mode, 0.0, TOL);
    }

    #[test]
    fn variance() {
        let lognorm_1 = LogNormal::new(3.4, 1.0).unwrap();
        let lognorm_2 = LogNormal::new(1.0, 3.0).unwrap();
        assert::close(
            lognorm_1.variance().unwrap(),
            (1.0_f64.exp() - 1.0) * 7.8_f64.exp(),
            TOL,
        );
        assert::close(
            lognorm_2.variance().unwrap(),
            (9.0_f64.exp() - 1.0) * 11.0_f64.exp(),
            TOL,
        );
    }

    #[test]
    fn draws_should_be_finite() {
        let mut rng = rand::thread_rng();
        let lognorm = LogNormal::standard();
        for _ in 0..100 {
            let x: f64 = lognorm.draw(&mut rng);
            assert!(x.is_finite())
        }
    }

    #[test]
    fn sample_length() {
        let mut rng = rand::thread_rng();
        let lognorm = LogNormal::standard();
        let xs: Vec<f64> = lognorm.sample(10, &mut rng);
        assert_eq!(xs.len(), 10);
    }

    #[test]
    fn standard_ln_pdf_at_one() {
        let lognorm = LogNormal::standard();
        assert::close(lognorm.ln_pdf(&1.0_f64), -0.91893853320467267, TOL);
    }

    #[test]
    fn standard_ln_pdf_at_e() {
        let lognorm = LogNormal::standard();
        assert::close(
            lognorm.ln_pdf(&f64::consts::E),
            -2.4189385332046727,
            TOL,
        );
    }

    #[test]
    fn should_contain_positve_finite_values() {
        let lognorm = LogNormal::standard();
        assert!(lognorm.supports(&1E-8_f32));
        assert!(lognorm.supports(&10E8_f64));
    }

    #[test]
    fn should_not_contain_negative_or_zero() {
        let lognorm = LogNormal::standard();
        assert!(!lognorm.supports(&-1.0_f64));
        assert!(!lognorm.supports(&0.0_f64));
    }

    #[test]
    fn should_not_contain_nan() {
        let lognorm = LogNormal::standard();
        assert!(!lognorm.supports(&f64::NAN));
    }

    #[test]
    fn should_not_contain_positive_or_negative_infinity() {
        let lognorm = LogNormal::standard();
        assert!(!lognorm.supports(&f64::INFINITY));
        assert!(!lognorm.supports(&f64::NEG_INFINITY));
    }

    #[test]
    fn skewness() {
        let lognorm = LogNormal::new(-1.2, 3.4).unwrap();
        assert::close(lognorm.skewness().unwrap(), 33936928.306623809, TOL);
    }

    #[test]
    fn kurtosis() {
        let lognorm = LogNormal::new(-1.2, 1.0).unwrap();
        assert::close(lognorm.kurtosis().unwrap(), 110.93639217631153, TOL);
    }

    #[test]
    fn cdf_standard_at_one_should_be_one_half() {
        let lognorm1 = LogNormal::new(0.0, 1.0).unwrap();
        assert::close(lognorm1.cdf(&1.0_f64), 0.5, TOL);
    }

    #[test]
    fn cdf_standard_value_at_two() {
        let lognorm = LogNormal::standard();
        assert::close(lognorm.cdf(&2.0_f64), 0.75589140421441725, TOL);
    }

    #[test]
    fn quantile_agree_with_cdf() {
        let mut rng = rand::thread_rng();
        let lognorm = LogNormal::standard();
        let xs: Vec<f64> = lognorm.sample(100, &mut rng);

        xs.iter().for_each(|x| {
            let p = lognorm.cdf(x);
            let y: f64 = lognorm.quantile(p);
            assert::close(y, *x, TOL);
        })
    }

    #[test]
    fn entropy() {
        let lognorm = LogNormal::new(1.2, 3.4).unwrap();
        assert::close(lognorm.entropy(), 3.8427139648267885, TOL);
    }

    #[test]
    fn entropy_standard() {
        let lognorm = LogNormal::standard();
        assert::close(lognorm.entropy(), 1.4189385332046727, TOL);
    }

    /*
    #[test]
    fn kl_of_idential_dsitrbutions_should_be_zero() {
        let lognorm = LogNormal::new(1.2, 3.4).unwrap();
        assert::close(lognorm.kl(&lognorm), 0.0, TOL);
    }
    */
}
