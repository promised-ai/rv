//! Beta distribution over x in (0, 1)
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::impl_display;
use crate::result;
use crate::traits::*;
use rand::Rng;
use special::Beta as _;
use special::Gamma as _;
use std::f64;

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
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Beta {
    pub alpha: f64,
    pub beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> result::Result<Self> {
        let alpha_ok = alpha > 0.0 && alpha.is_finite();
        let beta_ok = beta > 0.0 && beta.is_finite();

        if alpha_ok && beta_ok {
            Ok(Beta { alpha, beta })
        } else {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "α and β must be finite and greater than 0";
            let err = result::Error::new(err_kind, msg);
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

impl From<&Beta> for String {
    fn from(beta: &Beta) -> String {
        format!("Beta(α: {}, β: {})", beta.alpha, beta.beta)
    }
}

impl_display!(Beta);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Beta {
            fn ln_f(&self, x: &$kind) -> f64 {
                (self.alpha - 1.0) * f64::from(*x).ln()
                    + (self.beta - 1.0) * (1.0 - f64::from(*x)).ln()
                    - self.alpha.ln_beta(self.beta)
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let b = rand::distributions::Beta::new(self.alpha, self.beta);
                rng.sample(b) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let b = rand::distributions::Beta::new(self.alpha, self.beta);
                (0..n).map(|_| rng.sample(b) as $kind).collect()
            }
        }

        impl Support<$kind> for Beta {
            fn supports(&self, x: &$kind) -> bool {
                let xf = f64::from(*x);
                0.0 < xf && xf < 1.0
            }
        }

        impl ContinuousDistr<$kind> for Beta {}

        impl Cdf<$kind> for Beta {
            fn cdf(&self, x: &$kind) -> f64 {
                let ln_beta = self.alpha.ln_beta(self.beta);
                (*x as f64).inc_beta(self.alpha, self.beta, ln_beta)
            }
        }

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
    use super::*;
    use crate::misc::ks_test;
    use std::f64;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

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
    fn cdf_hump_shaped() {
        let beta = Beta::new(1.5, 2.0).unwrap();
        let xs: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let true_cdfs = vec![
            0.07431352501395692,
            0.19677398201998159,
            0.3368493728656773,
            0.48066620434559376,
            0.6187184335382292,
            0.7436128024718239,
            0.8492099269320867,
            0.9302042786399125,
            0.9818872134822818,
        ];
        let cdfs: Vec<f64> = xs.iter().map(|x| beta.cdf(x)).collect();

        assert::close(cdfs, true_cdfs, TOL);
    }

    #[test]
    fn cdf_bowl_shaped() {
        let beta = Beta::new(0.5, 0.7).unwrap();
        let xs: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let true_cdfs = vec![
            0.25502526668462605,
            0.36470920861186545,
            0.45212784899417957,
            0.529074597952903,
            0.6003642321330015,
            0.6688324654577239,
            0.7367590200551991,
            0.8067813209919937,
            0.8837889567707921,
        ];
        let cdfs: Vec<f64> = xs.iter().map(|x| beta.cdf(x)).collect();

        assert::close(cdfs, true_cdfs, TOL);
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

    #[test]
    fn draw_test_alpha_beta_gt_one() {
        let mut rng = rand::thread_rng();
        let beta = Beta::new(1.2, 3.4).unwrap();
        let cdf = |x: f64| beta.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = beta.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }

    #[test]
    fn draw_test_alpha_beta_lt_one() {
        let mut rng = rand::thread_rng();
        let beta = Beta::new(0.2, 0.7).unwrap();
        let cdf = |x: f64| beta.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = beta.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }

    #[test]
    fn beta_u_should_never_draw_1() {
        let mut rng = rand::thread_rng();
        let beta = Beta::new(0.5, 0.5).unwrap();

        let some_1 = beta
            .sample(10_000, &mut rng)
            .drain(..)
            .any(|x: f64| x == 1.0);
        assert!(!some_1);
    }
}
