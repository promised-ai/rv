extern crate rand;
extern crate special;

use self::rand::distributions::Normal;
use self::rand::Rng;
use self::special::Error;
use consts::*;
use std::f64::consts::SQRT_2;
use traits::*;

/// Gaussian / Normal distribution, N(μ, σ)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Gaussian {
    /// mean of the distribution
    mu: f64,
    /// Standard deviation
    sigma: f64,
}

impl Gaussian {
    pub fn new(mu: f64, sigma: f64) -> Self {
        Gaussian {
            mu: mu,
            sigma: sigma,
        }
    }

    /// Standard normal
    pub fn standard() -> Self {
        Gaussian::new(0.0, 1.0)
    }
}

impl Default for Gaussian {
    fn default() -> Self {
        Gaussian::standard()
    }
}

impl Rv for Gaussian {
    type DatumType = f64;
    fn ln_f(&self, x: &f64) -> f64 {
        let k = (x - self.mu) / self.sigma;
        -self.sigma.ln() - 0.5 * k * k
    }

    fn ln_normalizer(&self) -> f64 {
        HALF_LOG_2PI
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> f64 {
        let g = Normal::new(self.mu, self.sigma);
        rng.sample(g)
    }

    fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<f64> {
        let g = Normal::new(self.mu, self.sigma);
        (0..n).map(|_| rng.sample(g)).collect()
    }
}

impl ContinuousDistr for Gaussian {}

impl Support for Gaussian {
    fn contains(&self, x: &f64) -> bool {
        if x.is_finite() {
            true
        } else {
            false
        }
    }
}

impl Cdf for Gaussian {
    fn cdf(&self, x: &f64) -> f64 {
        let errf = ((x - self.mu) / (self.sigma * SQRT_2)).erf();
        0.5 * (1.0 + errf)
    }
}

impl Mean<f64> for Gaussian {
    fn mean(&self) -> Option<f64> {
        Some(self.mu)
    }
}

impl Median<f64> for Gaussian {
    fn median(&self) -> Option<f64> {
        Some(self.mu)
    }
}

impl Mode for Gaussian {
    fn mode(&self) -> Option<f64> {
        Some(self.mu)
    }
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

#[cfg(test)]
mod tests {
    extern crate assert;
    use super::*;
    use std::f64;

    const TOL: f64 = 1E-12;

    #[test]
    fn new() {
        let gauss = Gaussian::new(1.2, 3.0);
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
        let gauss = Gaussian::standard();
        assert::close(gauss.mean().unwrap(), 0.0, TOL);
    }

    #[test]
    fn standard_gaussian_variance_should_be_one() {
        let gauss = Gaussian::standard();
        assert::close(gauss.variance().unwrap(), 1.0, TOL);
    }

    #[test]
    fn mean_should_be_mu() {
        let mu = 3.4;
        let gauss = Gaussian::new(mu, 0.5);
        assert::close(gauss.mean().unwrap(), mu, TOL);
    }

    #[test]
    fn median_should_be_mu() {
        let mu = 3.4;
        let gauss = Gaussian::new(mu, 0.5);
        assert::close(gauss.median().unwrap(), mu, TOL);
    }

    #[test]
    fn mode_should_be_mu() {
        let mu = 3.4;
        let gauss = Gaussian::new(mu, 0.5);
        assert::close(gauss.mode().unwrap(), mu, TOL);
    }

    #[test]
    fn variance_should_be_sigma_squared() {
        let sigma = 0.5;
        let gauss = Gaussian::new(3.4, sigma);
        assert::close(gauss.variance().unwrap(), sigma * sigma, TOL);
    }

    #[test]
    fn draws_should_be_finite() {
        let mut rng = rand::thread_rng();
        let gauss = Gaussian::standard();
        for _ in 0..100 {
            assert!(gauss.draw(&mut rng).is_finite())
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
        assert::close(gauss.ln_pdf(&0.0), -0.91893853320467267, TOL);
    }

    #[test]
    fn standard_ln_pdf_off_zero() {
        let gauss = Gaussian::standard();
        assert::close(gauss.ln_pdf(&2.1), -3.1239385332046727, TOL);
    }

    #[test]
    fn nonstandard_ln_pdf_on_mean() {
        let gauss = Gaussian::new(-1.2, 0.33);
        assert::close(gauss.ln_pdf(&-1.2), 0.18972409131693846, TOL);
    }

    #[test]
    fn nonstandard_ln_pdf_off_mean() {
        let gauss = Gaussian::new(-1.2, 0.33);
        assert::close(gauss.ln_pdf(&0.0), -6.4218461566169447, TOL);
    }

    #[test]
    fn should_contain_finite_values() {
        let gauss = Gaussian::standard();
        assert!(gauss.contains(&0.0));
        assert!(gauss.contains(&10E8));
        assert!(gauss.contains(&-10E8));
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
        let gauss = Gaussian::new(-12.3, 45.6);
        assert::close(gauss.skewness().unwrap(), 0.0, TOL);
    }

    #[test]
    fn kurtosis_should_be_zero() {
        let gauss = Gaussian::new(-12.3, 45.6);
        assert::close(gauss.skewness().unwrap(), 0.0, TOL);
    }

    #[test]
    fn cdf_at_mean_should_be_one_half() {
        let mu1 = 2.3;
        let gauss1 = Gaussian::new(mu1, 0.2);
        assert::close(gauss1.cdf(&mu1), 0.5, TOL);

        let mu2 = -8.0;
        let gauss2 = Gaussian::new(mu2, 100.0);
        assert::close(gauss2.cdf(&mu2), 0.5, TOL);
    }

    #[test]
    fn cdf_value_at_one() {
        let gauss = Gaussian::standard();
        assert::close(gauss.cdf(&1.0), 0.84134474606854293, TOL);
    }

    #[test]
    fn cdf_value_at_neg_two() {
        let gauss = Gaussian::standard();
        assert::close(gauss.cdf(&-2.0), 0.022750131948179195, TOL);
    }

    #[test]
    fn standard_gaussian_entropy() {
        let gauss = Gaussian::standard();
        assert::close(gauss.entropy(), 1.4189385332046727, TOL);
    }

    #[test]
    fn entropy() {
        let gauss = Gaussian::new(3.0, 12.3);
        assert::close(gauss.entropy(), 3.9285377955830447, TOL);
    }
}
