extern crate rand;
extern crate special;

use self::rand::Rng;
use self::special::Error;
use self::rand::distributions::Normal;
use traits::*;

const SQRT_PI: f64 = 1.772453850905515881919427556567825376987457275391;
const HALF_LOG_2PI: f64 = 0.918938533204672669540968854562379419803619384766;
const HALF_LOG_2PI_E: f64 = 1.418938533204672669540968854562379419803619384766;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Gaussian {
    mu: f64,
    sigma: f64,
}

impl Gaussian {
    pub fn new(mu: f64, sigma: f64) -> Self {
        Gaussian { mu: mu, sigma: sigma }
    }

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
        let errf = ((x - self.mu) / (self.sigma * SQRT_PI)).erf();
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
