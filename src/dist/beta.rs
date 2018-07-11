extern crate rand;
extern crate special;

use self::rand::distributions::Gamma;
use self::rand::Rng;
use self::special::Beta as SBeta;
use self::special::Gamma as SGamma;
use traits::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Beta {
    alpha: f64,
    beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Beta {
            alpha: alpha,
            beta: beta,
        }
    }

    pub fn uniform() -> Self {
        Beta::new(1.0, 1.0)
    }

    pub fn jeffreys() -> Self {
        Beta::new(0.5, 0.5)
    }
}

impl Default for Beta {
    fn default() -> Self {
        Beta::jeffreys()
    }
}

impl Rv for Beta {
    type DatumType = f64;

    fn ln_f(&self, x: &f64) -> f64 {
        (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - *x).ln()
            - self.alpha.ln_beta(self.beta)
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> f64 {
        let ga = Gamma::new(self.alpha, 1.0);
        let gb = Gamma::new(self.beta, 1.0);
        let a = rng.sample(ga);
        let b = rng.sample(gb);
        a / (a + b)
    }

    fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<f64> {
        let ga = Gamma::new(self.alpha, 1.0);
        let gb = Gamma::new(self.beta, 1.0);
        (0..n)
            .map(|_| {
                let a = rng.sample(ga);
                let b = rng.sample(gb);
                a / (a + b)
            })
            .collect()
    }

    #[inline]
    fn ln_normalizer(&self) -> f64 {
        0.0
    }
}

impl ContinuousDistr for Beta {}

impl Mean<f64> for Beta {
    fn mean(&self) -> Option<f64> {
        Some(self.alpha / (self.alpha + self.beta))
    }
}

impl Mode for Beta {
    fn mode(&self) -> Option<f64> {
        if self.beta > 1.0 {
            if self.alpha > 1.0 {
                Some((self.alpha - 1.0) / (self.alpha + self.beta - 2.0))
            } else if self.alpha == 1.0 {
                Some(0.0)
            } else {
                None
            }
        } else if self.beta == 1.0 {
            if self.alpha > 1.0 {
                Some(1.0)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl Variance<f64> for Beta {
    fn variance(&self) -> Option<f64> {
        let apb = self.alpha + self.beta;
        Some(self.alpha * self.beta / (apb * apb * (apb + 1.0)))
    }
}

impl Entropy for Beta {
    fn entropy(&self) -> f64 {
        let apb = self.alpha + self.beta;
        self.alpha.ln_beta(self.beta)
            - (self.alpha - 1.0) * self.alpha.digamma()
            - (self.beta - 1.0) * self.beta.digamma() + (apb - 2.0) * apb.digamma()
    }
}

impl Skewness for Beta {
    fn skewness(&self) -> Option<f64> {
        let apb = self.alpha + self.beta;
        let numer = 2.0 * (self.beta - self.alpha) * (apb + 1.0).sqrt();
        let denom = (apb + 2.0) * (self.alpha * self.beta).sqrt();
        Some(numer / denom)
    }
}

impl Kurtosis for Beta {
    fn kurtosis(&self) -> Option<f64> {
        let apb = self.alpha + self.beta;
        let amb = self.alpha - self.beta;
        let atb = self.alpha * self.beta;
        let numer = 6.0 * (amb * amb * (apb + 1.0) - atb * (apb + 2.0));
        let denom = atb * (apb + 2.0) * (apb + 3.0);
        Some(numer / denom)
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
        let beta = Beta::new(1.0, 2.0);
        assert::close(beta.alpha, 1.0, TOL);
        assert::close(beta.beta, 2.0, TOL);
    }

    #[test]
    fn uniform() {
        let beta = Beta::uniform();
        assert::close(beta.alpha, 1.0, TOL);
        assert::close(beta.beta, 1.0, TOL);
    }

    #[test]
    fn jeffreys() {
        let beta = Beta::jeffreys();
        assert::close(beta.alpha, 0.5, TOL);
        assert::close(beta.beta, 0.5, TOL);
    }

    #[test]
    fn ln_pdf_center_value() {
        let beta = Beta::new(1.5, 2.0);
        assert::close(beta.ln_pdf(&0.5), 0.28203506914240184, TOL);
    }

    #[test]
    fn ln_pdf_low_value() {
        let beta = Beta::new(1.5, 2.0);
        assert::close(beta.ln_pdf(&0.01), -0.99087958886522731, TOL);
    }

    #[test]
    fn ln_pdf_high_value() {
        let beta = Beta::new(1.5, 2.0);
        assert::close(beta.ln_pdf(&0.99), -3.2884395139325218, TOL);
    }

    #[test]
    fn draw_should_resturn_values_within_0_to_1() {
        let mut rng = rand::thread_rng();
        let beta = Beta::jeffreys();
        for _ in 0..100 {
            let x = beta.draw(&mut rng);
            assert!(0.0 < x && x < 1.0);
        }
    }

    #[test]
    fn sample_returns_the_correct_number_draws() {
        let mut rng = rand::thread_rng();
        let beta = Beta::jeffreys();
        let xs = beta.sample(103, &mut rng);
        assert_eq!(xs.len(), 103);
    }

    #[test]
    fn uniform_mean() {
        assert::close(Beta::uniform().mean().unwrap(), 0.5, TOL);
    }

    #[test]
    fn jeffreys_mean() {
        assert::close(Beta::jeffreys().mean().unwrap(), 0.5, TOL);
    }

    #[test]
    fn mean() {
        assert::close(Beta::new(1.0, 5.0).mean().unwrap(), 1.0 / 6.0, TOL);
    }

    #[test]
    fn variance() {
        let beta = Beta::new(1.5, 2.0);
        assert::close(beta.variance().unwrap(), 0.054421768707482991, TOL);
    }

    #[test]
    fn mode_for_alpha_and_beta_greater_than_one() {
        let beta = Beta::new(1.5, 2.0);
        assert::close(beta.mode().unwrap(), 0.5 / 1.5, TOL);
    }

    #[test]
    fn mode_for_alpha_one_and_large_beta() {
        let beta = Beta::new(1.0, 2.0);
        assert::close(beta.mode().unwrap(), 0.0, TOL);
    }

    #[test]
    fn mode_for_large_alpha_and_beta_one() {
        let beta = Beta::new(2.0, 1.0);
        assert::close(beta.mode().unwrap(), 1.0, TOL);
    }

    #[test]
    fn mode_for_alpha_less_than_one_is_none() {
        assert!(Beta::new(0.99, 2.0).mode().is_none());
    }

    #[test]
    fn mode_for_beta_less_than_one_is_none() {
        assert!(Beta::new(2.0, 0.99).mode().is_none());
    }

    #[test]
    fn mode_for_alpha_and_beta_less_than_one_is_none() {
        assert!(Beta::new(0.99, 0.99).mode().is_none());
    }

    #[test]
    fn entropy() {
        let beta = Beta::new(1.5, 2.0);
        assert::close(beta.entropy(), -0.10805020110232236, TOL);
    }

    #[test]
    fn uniform_skewness_should_be_zero() {
        assert::close(Beta::uniform().skewness().unwrap(), 0.0, TOL);
    }

    #[test]
    fn jeffreysf_skewness_should_be_zero() {
        assert::close(Beta::jeffreys().skewness().unwrap(), 0.0, TOL);
    }

    #[test]
    fn skewness() {
        let beta = Beta::new(1.5, 2.0);
        assert::close(beta.skewness().unwrap(), 0.22268088570756162, TOL);
    }

    #[test]
    fn kurtosis() {
        let beta = Beta::new(1.5, 2.0);
        assert::close(beta.kurtosis().unwrap(), -0.8601398601398601, TOL);
    }
}
