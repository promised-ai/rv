//! Inverse Gaussian distribution over x in (0, ∞)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use rand::Rng;
use rand_distr::Normal;
use std::fmt;
use std::sync::OnceLock;

use crate::consts::{HALF_LN_2PI, LN_2PI};
use crate::data::InvGaussianSuffStat;
use crate::impl_display;
use crate::traits::{
    Cdf, ContinuousDistr, HasDensity, HasSuffStat, Kurtosis, Mean, Mode,
    Parameterized, Sampleable, Scalable, Shiftable, Skewness, Support,
    Variance,
};

/// [Inverse Gaussian distribution](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution),
/// N<sup>-1</sup>(μ, λ) over real values.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct InvGaussian {
    /// Mean
    mu: f64,
    /// Shape
    lambda: f64,
    /// Cached log(lambda)
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_lambda: OnceLock<f64>,
}

pub struct InvGaussianParameters {
    pub mu: f64,
    pub lambda: f64,
}

crate::impl_shiftable!(InvGaussian);
crate::impl_scalable!(InvGaussian);

impl Parameterized for InvGaussian {
    type Parameters = InvGaussianParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            mu: self.mu(),
            lambda: self.lambda(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.mu, params.lambda)
    }
}

impl PartialEq for InvGaussian {
    fn eq(&self, other: &InvGaussian) -> bool {
        self.mu == other.mu && self.lambda == other.lambda
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum InvGaussianError {
    /// The mu parameter is infinite or NaN
    MuNotFinite { mu: f64 },
    /// The mu parameter is less than or equal to zero
    MuTooLow { mu: f64 },
    /// The lambda parameter is less than or equal to zero
    LambdaTooLow { lambda: f64 },
    /// The lambda parameter is infinite or NaN
    LambdaNotFinite { lambda: f64 },
}

impl InvGaussian {
    /// Create a new Inverse Gaussian distribution
    ///
    /// # Arguments
    /// - mu: mean > 0
    /// - lambda: shape > 0
    ///
    /// ```
    /// use rv::dist::InvGaussian;
    /// let invgauss = InvGaussian::new(1.0, 3.0).unwrap();
    /// ```
    ///
    /// Mu and lambda must be finite and greater than 0.
    /// ```
    /// # use rv::dist::InvGaussian;
    /// use std::f64::{NAN, INFINITY};
    /// assert!(InvGaussian::new(0.0, 3.0).is_err());
    /// assert!(InvGaussian::new(NAN, 3.0).is_err());
    /// assert!(InvGaussian::new(INFINITY, 3.0).is_err());
    ///
    /// assert!(InvGaussian::new(1.0, 0.0).is_err());
    /// assert!(InvGaussian::new(1.0, NAN).is_err());
    /// assert!(InvGaussian::new(1.0, INFINITY).is_err());
    /// ```
    pub fn new(mu: f64, lambda: f64) -> Result<Self, InvGaussianError> {
        if !mu.is_finite() {
            Err(InvGaussianError::MuNotFinite { mu })
        } else if mu <= 0.0 {
            Err(InvGaussianError::MuTooLow { mu })
        } else if lambda <= 0.0 {
            Err(InvGaussianError::LambdaTooLow { lambda })
        } else if !lambda.is_finite() {
            Err(InvGaussianError::LambdaNotFinite { lambda })
        } else {
            Ok(InvGaussian {
                mu,
                lambda,
                ln_lambda: OnceLock::new(),
            })
        }
    }

    /// Creates a new `InvGaussian` without checking whether the parameters are
    /// valid.
    #[inline]
    #[must_use]
    pub fn new_unchecked(mu: f64, lambda: f64) -> Self {
        InvGaussian {
            mu,
            lambda,
            ln_lambda: OnceLock::new(),
        }
    }

    /// Get mu parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvGaussian;
    /// let ig = InvGaussian::new(2.0, 1.5).unwrap();
    ///
    /// assert_eq!(ig.mu(), 2.0);
    /// ```
    #[inline]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Set the value of mu
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvGaussian;
    /// let mut ig = InvGaussian::new(2.0, 1.5).unwrap();
    /// assert_eq!(ig.mu(), 2.0);
    ///
    /// ig.set_mu(1.3).unwrap();
    /// assert_eq!(ig.mu(), 1.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::InvGaussian;
    /// # let mut ig = InvGaussian::new(2.0, 1.5).unwrap();
    /// assert!(ig.set_mu(1.3).is_ok());
    /// assert!(ig.set_mu(0.0).is_err());
    /// assert!(ig.set_mu(-1.0).is_err());
    /// assert!(ig.set_mu(f64::NEG_INFINITY).is_err());
    /// assert!(ig.set_mu(f64::INFINITY).is_err());
    /// assert!(ig.set_mu(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_mu(&mut self, mu: f64) -> Result<(), InvGaussianError> {
        if !mu.is_finite() {
            Err(InvGaussianError::MuNotFinite { mu })
        } else if mu <= 0.0 {
            Err(InvGaussianError::MuTooLow { mu })
        } else {
            self.set_mu_unchecked(mu);
            Ok(())
        }
    }

    /// Set the value of mu without input validation
    #[inline]
    pub fn set_mu_unchecked(&mut self, mu: f64) {
        self.mu = mu;
    }

    /// Get lambda parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvGaussian;
    /// let ig = InvGaussian::new(2.0, 1.5).unwrap();
    ///
    /// assert_eq!(ig.lambda(), 1.5);
    /// ```
    #[inline]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Set the value of lambda
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvGaussian;
    /// let mut ig = InvGaussian::new(1.0, 2.0).unwrap();
    /// assert_eq!(ig.lambda(), 2.0);
    ///
    /// ig.set_lambda(2.3).unwrap();
    /// assert_eq!(ig.lambda(), 2.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::InvGaussian;
    /// # let mut ig = InvGaussian::new(1.0, 2.0).unwrap();
    /// assert!(ig.set_lambda(2.3).is_ok());
    /// assert!(ig.set_lambda(0.0).is_err());
    /// assert!(ig.set_lambda(-1.0).is_err());
    /// assert!(ig.set_lambda(f64::INFINITY).is_err());
    /// assert!(ig.set_lambda(f64::NEG_INFINITY).is_err());
    /// assert!(ig.set_lambda(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_lambda(&mut self, lambda: f64) -> Result<(), InvGaussianError> {
        if lambda <= 0.0 {
            Err(InvGaussianError::LambdaTooLow { lambda })
        } else if !lambda.is_finite() {
            Err(InvGaussianError::LambdaNotFinite { lambda })
        } else {
            self.set_lambda_unchecked(lambda);
            Ok(())
        }
    }

    /// Set the value of lambda without input validation
    #[inline]
    pub fn set_lambda_unchecked(&mut self, lambda: f64) {
        self.ln_lambda = OnceLock::new();
        self.lambda = lambda;
    }

    #[inline]
    fn ln_lambda(&self) -> f64 {
        *self.ln_lambda.get_or_init(|| self.lambda.ln())
    }
}

impl From<&InvGaussian> for String {
    fn from(ig: &InvGaussian) -> String {
        format!("N⁻¹(μ: {}, λ: {})", ig.mu, ig.lambda)
    }
}

impl_display!(InvGaussian);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for InvGaussian {
            fn ln_f(&self, x: &$kind) -> f64 {
                let InvGaussianParameters { mu, lambda } = self.emit_params();
                let xf = f64::from(*x);
                let z = self.ln_lambda() - xf.ln().mul_add(3.0, LN_2PI);
                let err = xf - mu;
                let term = lambda * err * err / (2.0 * mu * mu * xf);
                z.mul_add(0.5, -term)
            }
        }

        impl Sampleable<$kind> for InvGaussian {
            // https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Sampling_from_an_inverse-Gaussian_distribution
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let InvGaussianParameters { mu, lambda } = self.emit_params();
                let g = Normal::new(0.0, 1.0).unwrap();
                let v: f64 = rng.sample(g);
                let y = v * v;
                let mu2 = mu * mu;
                let x = 0.5_f64.mul_add(
                    (mu / lambda).mul_add(
                        -(4.0 * mu * lambda).mul_add(y, mu2 * y * y).sqrt(),
                        mu2 * y / lambda,
                    ),
                    mu,
                );
                let z: f64 = rng.gen();

                if z <= mu / (mu + x) {
                    x as $kind
                } else {
                    (mu2 / x) as $kind
                }
            }
        }

        impl ContinuousDistr<$kind> for InvGaussian {}

        impl Support<$kind> for InvGaussian {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite()
            }
        }

        impl Cdf<$kind> for InvGaussian {
            fn cdf(&self, x: &$kind) -> f64 {
                let xf = f64::from(*x);
                let InvGaussianParameters { mu, lambda } = self.emit_params();
                let gauss = crate::dist::Gaussian::standard();
                let z = (lambda / xf).sqrt();
                let a = z * (xf / mu - 1.0);
                let b = -z * (xf / mu + 1.0);
                (2.0 * lambda / mu)
                    .exp()
                    .mul_add(gauss.cdf(&b), gauss.cdf(&a))
            }
        }
        impl Mean<$kind> for InvGaussian {
            fn mean(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl Mode<$kind> for InvGaussian {
            fn mode(&self) -> Option<$kind> {
                let InvGaussianParameters { mu, lambda } = self.emit_params();
                let a = (1.0 + 0.25 * 9.0 * mu * mu / (lambda * lambda)).sqrt();
                let b = 0.5 * 3.0 * mu / lambda;
                let mode = mu * (a - b);
                Some(mode as $kind)
            }
        }

        impl HasSuffStat<$kind> for InvGaussian {
            type Stat = InvGaussianSuffStat;

            fn empty_suffstat(&self) -> Self::Stat {
                InvGaussianSuffStat::new()
            }

            fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
                let n = stat.n() as f64;
                let mu2 = self.mu * self.mu;
                let t1 = n.mul_add(
                    0.5_f64.mul_add(self.ln_lambda(), -HALF_LN_2PI),
                    -3.0 / 2.0 * stat.sum_ln_x(),
                );
                let t2 = self.lambda() / (2.0 * mu2);
                let t3 = (2.0 * n).mul_add(-self.mu, stat.sum_x());
                let t4 = stat.sum_inv_x().mul_add(mu2, t3);
                t2.mul_add(-t4, t1)
            }
        }
    };
}

impl Variance<f64> for InvGaussian {
    fn variance(&self) -> Option<f64> {
        Some(self.mu.powi(3) / self.lambda)
    }
}

impl Skewness for InvGaussian {
    fn skewness(&self) -> Option<f64> {
        Some(2.0 * (self.mu / self.lambda).sqrt())
    }
}

impl Kurtosis for InvGaussian {
    fn kurtosis(&self) -> Option<f64> {
        Some(15.0 * self.mu / self.lambda)
    }
}

impl_traits!(f32);
impl_traits!(f64);

impl std::error::Error for InvGaussianError {}

impl fmt::Display for InvGaussianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MuNotFinite { mu } => write!(f, "non-finite mu: {mu}"),
            Self::MuTooLow { mu } => {
                write!(f, "mu ({mu}) must be greater than zero")
            }
            Self::LambdaTooLow { lambda } => {
                write!(f, "lambda ({lambda}) must be greater than zero")
            }
            Self::LambdaNotFinite { lambda } => {
                write!(f, "non-finite lambda: {lambda}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;

    const N_TRIES: usize = 10;
    const KS_PVAL: f64 = 0.2;

    crate::test_basic_impls!(
        f64,
        InvGaussian,
        InvGaussian::new(1.0, 2.3).unwrap()
    );

    #[test]
    fn mode_is_highest_point() {
        let mut rng = rand::thread_rng();
        let mu_prior = crate::dist::InvGamma::new_unchecked(2.0, 2.0);
        let lambda_prior = crate::dist::InvGamma::new_unchecked(2.0, 2.0);
        for _ in 0..100 {
            let mu: f64 = mu_prior.draw(&mut rng);
            let lambda: f64 = lambda_prior.draw(&mut rng);
            let ig = InvGaussian::new(mu, lambda).unwrap();
            let mode: f64 = ig.mode().unwrap();
            let ln_f_mode = ig.ln_f(&mode);
            let ln_f_plus = ig.ln_f(&(mode + 1e-4));
            let ln_f_minus = ig.ln_f(&(mode - 1e-4));

            assert!(ln_f_mode > ln_f_plus);
            assert!(ln_f_mode > ln_f_minus);
        }
    }

    #[test]
    fn quad_on_pdf_agrees_with_cdf_x() {
        use peroxide::numerical::integral::{
            gauss_kronrod_quadrature, Integral,
        };
        let ig = InvGaussian::new(1.1, 2.5).unwrap();
        // use pdf to hit `supports(x)` first
        let pdf = |x: f64| ig.pdf(&x);
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let x: f64 = ig.draw(&mut rng);
            let res = gauss_kronrod_quadrature(
                pdf,
                (1e-16, x),
                Integral::G7K15(1e-10, 100),
            );
            let cdf = ig.cdf(&x);
            assert::close(res, cdf, 1e-7);
        }
    }

    #[test]
    fn draw_vs_kl() {
        let mut rng = rand::thread_rng();
        let ig = InvGaussian::new(1.2, 3.4).unwrap();
        let cdf = |x: f64| ig.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = ig.sample(1000, &mut rng);
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
    fn ln_f_stat() {
        use crate::traits::SuffStat;
        
        let data: Vec<f64> = vec![0.1, 0.23, 1.4, 0.65, 0.22, 3.1];
        let mut stat = InvGaussianSuffStat::new();
        stat.observe_many(&data);

        let igauss = InvGaussian::new(0.3, 2.33).unwrap();

        let ln_f_base: f64 = data.iter().map(|x| igauss.ln_f(x)).sum();
        let ln_f_stat: f64 =
            <InvGaussian as HasSuffStat<f64>>::ln_f_stat(&igauss, &stat);

        assert::close(ln_f_base, ln_f_stat, 1e-12);
    }
}
