//! Gaussian/Normal distribution over x in (-∞, ∞)
extern crate rand;
extern crate special;

use std::f64::consts::SQRT_2;
use std::io;

use self::rand::distributions::Normal;
use self::rand::Rng;
use self::special::Error;

use consts::*;
use dist::GaussianSuffStat;
use traits::*;

/// Gaussian / Normal distribution, N(μ, σ)
///
/// # Examples
///
/// Compute the [KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
/// between two Gaussians.
///
/// ```
/// # extern crate rv;
/// use rv::prelude::*;
///
/// let gauss_1 = Gaussian::new(0.1, 2.3).unwrap();
/// let gauss_2 = Gaussian::standard();
///
/// // KL is not symmetric
/// let kl_12 = gauss_1.kl(&gauss_2);
/// let kl_21 = gauss_2.kl(&gauss_1);
///
/// // ... but kl_sym is because it's the sum of KL(P|Q) and KL(Q|P)
/// let kl_sym = gauss_1.kl_sym(&gauss_2);
/// assert!((kl_sym - (kl_12 + kl_21)).abs() < 1E-12);
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Gaussian {
    /// Mean
    pub mu: f64,
    /// Standard deviation
    pub sigma: f64,
}

impl Gaussian {
    pub fn new(mu: f64, sigma: f64) -> io::Result<Self> {
        let mu_ok = mu.is_finite();
        let sigma_ok = sigma > 0.0 && sigma.is_finite();
        if !mu_ok {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "mu must be finite");
            Err(err)
        } else if !sigma_ok {
            let err_kind = io::ErrorKind::InvalidInput;
            let msg = "sigma must be finite and greater than zero";
            let err = io::Error::new(err_kind, msg);
            Err(err)
        } else {
            Ok(Gaussian { mu, sigma })
        }
    }

    /// Standard normal
    pub fn standard() -> Self {
        Gaussian::new(0.0, 1.0).unwrap()
    }
}

impl Default for Gaussian {
    fn default() -> Self {
        Gaussian::standard()
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Gaussian {
            fn ln_f(&self, x: &$kind) -> f64 {
                let k = (f64::from(*x) - self.mu) / self.sigma;
                -self.sigma.ln() - 0.5 * k * k
            }

            fn ln_normalizer(&self) -> f64 {
                HALF_LOG_2PI
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let g = Normal::new(self.mu, self.sigma);
                rng.sample(g) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let g = Normal::new(self.mu, self.sigma);
                (0..n).map(|_| rng.sample(g) as $kind).collect()
            }
        }

        impl ContinuousDistr<$kind> for Gaussian {}

        impl Support<$kind> for Gaussian {
            fn contains(&self, x: &$kind) -> bool {
                x.is_finite()
            }
        }

        impl Cdf<$kind> for Gaussian {
            fn cdf(&self, x: &$kind) -> f64 {
                let errf =
                    ((f64::from(*x) - self.mu) / (self.sigma * SQRT_2)).erf();
                0.5 * (1.0 + errf)
            }
        }

        impl InverseCdf<$kind> for Gaussian {
            fn invcdf(&self, p: f64) -> $kind {
                if (p <= 0.0) || (1.0 <= p) {
                    panic!("P out of range");
                }
                let x =
                    self.mu + self.sigma * SQRT_2 * (2.0 * p - 1.0).inv_erf();
                x as $kind
            }
        }

        impl Mean<$kind> for Gaussian {
            fn mean(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl Median<$kind> for Gaussian {
            fn median(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl Mode<$kind> for Gaussian {
            fn mode(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl HasSuffStat<$kind> for Gaussian {
            type Stat = GaussianSuffStat;
            fn empty_suffstat(&self) -> Self::Stat {
                GaussianSuffStat::new()
            }
        }
    };
}

impl Variance<f64> for Gaussian {
    fn variance(&self) -> Option<f64> {
        Some(self.sigma * self.sigma)
    }
}

impl Entropy for Gaussian {
    fn entropy(&self) -> f64 {
        HALF_LOG_2PI_E + self.sigma.ln()
    }
}

impl Skewness for Gaussian {
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl Kurtosis for Gaussian {
    fn kurtosis(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl KlDivergence for Gaussian {
    fn kl(&self, other: &Self) -> f64 {
        let m1 = self.mu;
        let m2 = other.mu;

        let s1 = self.sigma;
        let s2 = other.sigma;

        let term1 = s2.ln() - s1.ln();
        let term2 = (s1 * s1 + (m1 - m2) * (m1 - m2)) / (2.0 * s2 * s2);

        term1 + term2 - 0.5
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
        let gauss = Gaussian::new(1.2, 3.0).unwrap();
        assert::close(gauss.mu, 1.2, TOL);
        assert::close(gauss.sigma, 3.0, TOL);
    }

    #[test]
    fn standard() {
        let gauss = Gaussian::standard();
        assert::close(gauss.mu, 0.0, TOL);
        assert::close(gauss.sigma, 1.0, TOL);
    }

    #[test]
    fn standard_gaussian_mean_should_be_zero() {
        let mean: f64 = Gaussian::standard().mean().unwrap();
        assert::close(mean, 0.0, TOL);
    }

    #[test]
    fn standard_gaussian_variance_should_be_one() {
        assert::close(Gaussian::standard().variance().unwrap(), 1.0, TOL);
    }

    #[test]
    fn mean_should_be_mu() {
        let mu = 3.4;
        let mean: f64 = Gaussian::new(mu, 0.5).unwrap().mean().unwrap();
        assert::close(mean, mu, TOL);
    }

    #[test]
    fn median_should_be_mu() {
        let mu = 3.4;
        let median: f64 = Gaussian::new(mu, 0.5).unwrap().median().unwrap();
        assert::close(median, mu, TOL);
    }

    #[test]
    fn mode_should_be_mu() {
        let mu = 3.4;
        let mode: f64 = Gaussian::new(mu, 0.5).unwrap().mode().unwrap();
        assert::close(mode, mu, TOL);
    }

    #[test]
    fn variance_should_be_sigma_squared() {
        let sigma = 0.5;
        let gauss = Gaussian::new(3.4, sigma).unwrap();
        assert::close(gauss.variance().unwrap(), sigma * sigma, TOL);
    }

    #[test]
    fn draws_should_be_finite() {
        let mut rng = rand::thread_rng();
        let gauss = Gaussian::standard();
        for _ in 0..100 {
            let x: f64 = gauss.draw(&mut rng);
            assert!(x.is_finite())
        }
    }

    #[test]
    fn sample_length() {
        let mut rng = rand::thread_rng();
        let gauss = Gaussian::standard();
        let xs: Vec<f64> = gauss.sample(10, &mut rng);
        assert_eq!(xs.len(), 10);
    }

    #[test]
    fn standard_ln_pdf_at_zero() {
        let gauss = Gaussian::standard();
        assert::close(gauss.ln_pdf(&0.0_f64), -0.91893853320467267, TOL);
    }

    #[test]
    fn standard_ln_pdf_off_zero() {
        let gauss = Gaussian::standard();
        assert::close(gauss.ln_pdf(&2.1_f64), -3.1239385332046727, TOL);
    }

    #[test]
    fn nonstandard_ln_pdf_on_mean() {
        let gauss = Gaussian::new(-1.2, 0.33).unwrap();
        assert::close(gauss.ln_pdf(&-1.2_f64), 0.18972409131693846, TOL);
    }

    #[test]
    fn nonstandard_ln_pdf_off_mean() {
        let gauss = Gaussian::new(-1.2, 0.33).unwrap();
        assert::close(gauss.ln_pdf(&0.0_f32), -6.4218461566169447, TOL);
    }

    #[test]
    fn should_contain_finite_values() {
        let gauss = Gaussian::standard();
        assert!(gauss.contains(&0.0_f32));
        assert!(gauss.contains(&10E8_f64));
        assert!(gauss.contains(&-10E8_f64));
    }

    #[test]
    fn should_not_contain_nan() {
        let gauss = Gaussian::standard();
        assert!(!gauss.contains(&f64::NAN));
    }

    #[test]
    fn should_not_contain_positive_or_negative_infinity() {
        let gauss = Gaussian::standard();
        assert!(!gauss.contains(&f64::INFINITY));
        assert!(!gauss.contains(&f64::NEG_INFINITY));
    }

    #[test]
    fn skewness_should_be_zero() {
        let gauss = Gaussian::new(-12.3, 45.6).unwrap();
        assert::close(gauss.skewness().unwrap(), 0.0, TOL);
    }

    #[test]
    fn kurtosis_should_be_zero() {
        let gauss = Gaussian::new(-12.3, 45.6).unwrap();
        assert::close(gauss.skewness().unwrap(), 0.0, TOL);
    }

    #[test]
    fn cdf_at_mean_should_be_one_half() {
        let mu1: f64 = 2.3;
        let gauss1 = Gaussian::new(mu1, 0.2).unwrap();
        assert::close(gauss1.cdf(&mu1), 0.5, TOL);

        let mu2: f32 = -8.0;
        let gauss2 = Gaussian::new(mu2.into(), 100.0).unwrap();
        assert::close(gauss2.cdf(&mu2), 0.5, TOL);
    }

    #[test]
    fn cdf_value_at_one() {
        let gauss = Gaussian::standard();
        assert::close(gauss.cdf(&1.0_f64), 0.84134474606854293, TOL);
    }

    #[test]
    fn cdf_value_at_neg_two() {
        let gauss = Gaussian::standard();
        assert::close(gauss.cdf(&-2.0_f64), 0.022750131948179195, TOL);
    }

    #[test]
    fn quantile_at_one_half_should_be_mu() {
        let mu = 1.2315;
        let gauss = Gaussian::new(mu, 1.0).unwrap();
        let x: f64 = gauss.quantile(0.5);
        assert::close(x, mu, TOL);
    }

    #[test]
    fn quantile_agree_with_cdf() {
        let mut rng = rand::thread_rng();
        let gauss = Gaussian::standard();
        let xs: Vec<f64> = gauss.sample(100, &mut rng);

        xs.iter().for_each(|x| {
            let p = gauss.cdf(x);
            let y: f64 = gauss.quantile(p);
            assert::close(y, *x, TOL);
        })
    }

    #[test]
    fn standard_gaussian_entropy() {
        let gauss = Gaussian::standard();
        assert::close(gauss.entropy(), 1.4189385332046727, TOL);
    }

    #[test]
    fn entropy() {
        let gauss = Gaussian::new(3.0, 12.3).unwrap();
        assert::close(gauss.entropy(), 3.9285377955830447, TOL);
    }

    #[test]
    fn kl_of_idential_dsitrbutions_should_be_zero() {
        let gauss = Gaussian::new(1.2, 3.4).unwrap();
        assert::close(gauss.kl(&gauss), 0.0, TOL);
    }

    #[test]
    fn kl() {
        let g1 = Gaussian::new(1.0, 2.0).unwrap();
        let g2 = Gaussian::new(2.0, 1.0).unwrap();
        let kl = 0.5_f64.ln() + 5.0 / 2.0 - 0.5;
        assert::close(g1.kl(&g2), kl, TOL);
    }
}
