//! Beta distribution over x in (0, 1)
extern crate rand;
extern crate special;

use self::rand::distributions::Gamma;
use self::rand::Rng;
use self::special::Beta as SBeta;
use self::special::Gamma as SGamma;
use std::f64;
use std::io;

use traits::*;

/// [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution),
/// Beta(α, β) over x in (0, 1).
///
/// # Examples
///
/// Beta as a conjugate prior for Bernoulli
///
/// ```
/// # extern crate rv;
/// use rv::prelude::*;
///
/// // A prior that encodes our strong belief that coins are fair:
/// let beta = Beta::new(5.0, 5.0).unwrap();
///
/// // The posterior predictive probability that a coin will come up heads given
/// // no new observations.
/// let p_prior_heads = beta.pp(&true, &DataOrSuffStat::None); // 0.5
/// assert!((p_prior_heads - 0.5).abs() < 1E-12);
///
/// // Five Bernoulli trials. We flipped a coin five times and it came up head
/// // four times.
/// let flips = vec![true, true, false, true, true];
///
/// // The posterior predictive probability that a coin will come up heads given
/// // the five flips we just saw.
/// let p_pred_heads = beta.pp(&true, &DataOrSuffStat::Data(&flips)); // 9/15
/// assert!((p_pred_heads - 3.0/5.0).abs() < 1E-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Beta {
    pub alpha: f64,
    pub beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> io::Result<Self> {
        let alpha_ok = alpha > 0.0 && alpha.is_finite();
        let beta_ok = beta > 0.0 && beta.is_finite();

        if alpha_ok && beta_ok {
            Ok(Beta { alpha, beta })
        } else {
            let err_kind = io::ErrorKind::InvalidInput;
            let msg = "α and β must be finite and greater than 0";
            let err = io::Error::new(err_kind, msg);
            Err(err)
        }
    }

    /// Create a `Beta` distribution with even density over (0, 1).
    pub fn uniform() -> Self {
        Beta::new(1.0, 1.0).unwrap()
    }

    /// Create a `Beta` distribution with the Jeffrey's parameterization,
    /// *Beta(0.5, 0.5)*.
    pub fn jeffreys() -> Self {
        Beta::new(0.5, 0.5).unwrap()
    }
}

impl Default for Beta {
    fn default() -> Self {
        Beta::jeffreys()
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Beta {
            fn ln_f(&self, x: &$kind) -> f64 {
                (self.alpha - 1.0) * f64::from(*x).ln()
                    + (self.beta - 1.0) * (1.0 - f64::from(*x)).ln()
                    - self.alpha.ln_beta(self.beta)
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let ga = Gamma::new(self.alpha, 1.0);
                let gb = Gamma::new(self.beta, 1.0);
                let a = rng.sample(ga);
                let b = rng.sample(gb);
                (a / (a + b)) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let ga = Gamma::new(self.alpha, 1.0);
                let gb = Gamma::new(self.beta, 1.0);
                (0..n)
                    .map(|_| {
                        let a = rng.sample(ga);
                        let b = rng.sample(gb);
                        (a / (a + b)) as $kind
                    }).collect()
            }
        }

        impl Support<$kind> for Beta {
            fn contains(&self, x: &$kind) -> bool {
                let xf = f64::from(*x);
                0.0 < xf && xf < 1.0
            }
        }

        impl ContinuousDistr<$kind> for Beta {}

        impl Mean<$kind> for Beta {
            fn mean(&self) -> Option<$kind> {
                Some((self.alpha / (self.alpha + self.beta)) as $kind)
            }
        }

        impl Mode<$kind> for Beta {
            fn mode(&self) -> Option<$kind> {
                if self.beta > 1.0 {
                    if self.alpha > 1.0 {
                        let m: f64 =
                            (self.alpha - 1.0) / (self.alpha + self.beta - 2.0);
                        Some(m as $kind)
                    } else if (self.alpha - 1.0).abs() < f64::EPSILON {
                        Some(0.0)
                    } else {
                        None
                    }
                } else if (self.beta - 1.0).abs() < f64::EPSILON {
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
    };
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
            - (self.beta - 1.0) * self.beta.digamma()
            + (apb - 2.0) * apb.digamma()
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
        let beta = Beta::new(1.0, 2.0).unwrap();
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
        let beta = Beta::new(1.5, 2.0).unwrap();
        assert::close(beta.ln_pdf(&0.5), 0.28203506914240184, TOL);
    }

    #[test]
    fn ln_pdf_low_value() {
        let beta = Beta::new(1.5, 2.0).unwrap();
        assert::close(beta.ln_pdf(&0.01), -0.99087958886522731, TOL);
    }

    #[test]
    fn ln_pdf_high_value() {
        let beta = Beta::new(1.5, 2.0).unwrap();
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
        let xs: Vec<f32> = beta.sample(103, &mut rng);
        assert_eq!(xs.len(), 103);
    }

    #[test]
    fn uniform_mean() {
        let mean: f64 = Beta::uniform().mean().unwrap();
        assert::close(mean, 0.5, TOL);
    }

    #[test]
    fn jeffreys_mean() {
        let mean: f64 = Beta::jeffreys().mean().unwrap();
        assert::close(mean, 0.5, TOL);
    }

    #[test]
    fn mean() {
        let mean: f64 = Beta::new(1.0, 5.0).unwrap().mean().unwrap();
        assert::close(mean, 1.0 / 6.0, TOL);
    }

    #[test]
    fn variance() {
        let beta = Beta::new(1.5, 2.0).unwrap();
        assert::close(beta.variance().unwrap(), 0.054421768707482991, TOL);
    }

    #[test]
    fn mode_for_alpha_and_beta_greater_than_one() {
        let mode: f64 = Beta::new(1.5, 2.0).unwrap().mode().unwrap();
        assert::close(mode, 0.5 / 1.5, TOL);
    }

    #[test]
    fn mode_for_alpha_one_and_large_beta() {
        let mode: f64 = Beta::new(1.0, 2.0).unwrap().mode().unwrap();
        assert::close(mode, 0.0, TOL);
    }

    #[test]
    fn mode_for_large_alpha_and_beta_one() {
        let mode: f64 = Beta::new(2.0, 1.0).unwrap().mode().unwrap();
        assert::close(mode, 1.0, TOL);
    }

    #[test]
    fn mode_for_alpha_less_than_one_is_none() {
        let mode_opt: Option<f64> = Beta::new(0.99, 2.0).unwrap().mode();
        assert!(mode_opt.is_none());
    }

    #[test]
    fn mode_for_beta_less_than_one_is_none() {
        let mode_opt: Option<f64> = Beta::new(2.0, 0.99).unwrap().mode();
        assert!(mode_opt.is_none());
    }

    #[test]
    fn mode_for_alpha_and_beta_less_than_one_is_none() {
        let mode_opt: Option<f64> = Beta::new(0.99, 0.99).unwrap().mode();
        assert!(mode_opt.is_none());
    }

    #[test]
    fn entropy() {
        let beta = Beta::new(1.5, 2.0).unwrap();
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
        let beta = Beta::new(1.5, 2.0).unwrap();
        assert::close(beta.skewness().unwrap(), 0.22268088570756162, TOL);
    }

    #[test]
    fn kurtosis() {
        let beta = Beta::new(1.5, 2.0).unwrap();
        assert::close(beta.kurtosis().unwrap(), -0.8601398601398601, TOL);
    }
}
