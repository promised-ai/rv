//! Gaussian/Normal distribution over x in (-∞, ∞)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use rand::Rng;
use rand_distr::Normal;
use special::Error as _;
use std::f64::consts::SQRT_2;
use std::fmt;

use crate::consts::{HALF_LN_2PI, HALF_LN_2PI_E};
use crate::data::GaussianSuffStat;
use crate::impl_display;
use crate::traits::HasDensity;
use crate::traits::{
    Cdf, ContinuousDistr, Entropy, HasSuffStat, InverseCdf, KlDivergence,
    Kurtosis, Mean, Median, Mode, Parameterized, QuadBounds, Sampleable,
    Scalable, Shiftable, Skewness, Support, Variance,
};

/// Gaussian / [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution),
/// N(μ, σ) over real values.
///
/// # Examples
///
/// Compute the [KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
/// between two Gaussians.
///
/// ```
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
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[cfg_attr(feature = "serde1", serde(try_from = "GaussianParameters"))]
#[cfg_attr(feature = "serde1", serde(into = "GaussianParameters"))]
pub struct Gaussian {
    /// Mean
    mu: f64,
    /// Standard deviation
    sigma: f64,
    /// Cached log(sigma)
    ln_sigma: f64,
}

impl PartialEq for Gaussian {
    fn eq(&self, other: &Gaussian) -> bool {
        self.mu == other.mu && self.sigma == other.sigma
    }
}

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct GaussianParameters {
    pub mu: f64,
    pub sigma: f64,
}

impl TryFrom<GaussianParameters> for Gaussian {
    type Error = GaussianError;

    fn try_from(params: GaussianParameters) -> Result<Self, Self::Error> {
        Gaussian::new(params.mu, params.sigma)
    }
}

impl From<Gaussian> for GaussianParameters {
    fn from(gauss: Gaussian) -> Self {
        GaussianParameters {
            mu: gauss.mu,
            sigma: gauss.sigma,
        }
    }
}

impl Parameterized for Gaussian {
    type Parameters = GaussianParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            mu: self.mu(),
            sigma: self.sigma(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.mu, params.sigma)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum GaussianError {
    /// The mu parameter is infinite or NaN
    MuNotFinite { mu: f64 },
    /// The sigma parameter is less than or equal to zero
    SigmaTooLow { sigma: f64 },
    /// The sigma parameter is infinite or NaN
    SigmaNotFinite { sigma: f64 },
}

impl Gaussian {
    /// Create a new Gaussian distribution
    ///
    /// # Arguments
    /// - mu: mean
    /// - sigma: standard deviation
    pub fn new(mu: f64, sigma: f64) -> Result<Self, GaussianError> {
        if !mu.is_finite() {
            Err(GaussianError::MuNotFinite { mu })
        } else if sigma <= 0.0 {
            Err(GaussianError::SigmaTooLow { sigma })
        } else if !sigma.is_finite() {
            Err(GaussianError::SigmaNotFinite { sigma })
        } else {
            Ok(Gaussian {
                mu,
                sigma,
                ln_sigma: sigma.ln(),
            })
        }
    }

    /// Creates a new Gaussian without checking whether the parameters are
    /// valid.
    #[inline]
    #[must_use]
    pub fn new_unchecked(mu: f64, sigma: f64) -> Self {
        Gaussian {
            mu,
            sigma,
            ln_sigma: sigma.ln(),
        }
    }

    /// Standard normal
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gaussian;
    /// let gauss = Gaussian::standard();
    ///
    /// assert_eq!(gauss, Gaussian::new(0.0, 1.0).unwrap());
    /// ```
    #[inline]
    #[must_use]
    pub fn standard() -> Self {
        Gaussian {
            mu: 0.0,
            sigma: 1.0,
            ln_sigma: 0.0,
        }
    }

    /// Get mu parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gaussian;
    /// let gauss = Gaussian::new(2.0, 1.5).unwrap();
    ///
    /// assert_eq!(gauss.mu(), 2.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Set the value of mu
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gaussian;
    /// let mut gauss = Gaussian::new(2.0, 1.5).unwrap();
    /// assert_eq!(gauss.mu(), 2.0);
    ///
    /// gauss.set_mu(1.3).unwrap();
    /// assert_eq!(gauss.mu(), 1.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Gaussian;
    /// # let mut gauss = Gaussian::new(2.0, 1.5).unwrap();
    /// assert!(gauss.set_mu(1.3).is_ok());
    /// assert!(gauss.set_mu(f64::NEG_INFINITY).is_err());
    /// assert!(gauss.set_mu(f64::INFINITY).is_err());
    /// assert!(gauss.set_mu(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_mu(&mut self, mu: f64) -> Result<(), GaussianError> {
        if mu.is_finite() {
            self.set_mu_unchecked(mu);
            Ok(())
        } else {
            Err(GaussianError::MuNotFinite { mu })
        }
    }

    /// Set the value of mu without input validation
    #[inline]
    pub fn set_mu_unchecked(&mut self, mu: f64) {
        self.mu = mu;
    }

    /// Get sigma parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gaussian;
    /// let gauss = Gaussian::new(2.0, 1.5).unwrap();
    ///
    /// assert_eq!(gauss.sigma(), 1.5);
    /// ```
    #[inline]
    #[must_use]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Set the value of sigma
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gaussian;
    /// let mut gauss = Gaussian::standard();
    /// assert_eq!(gauss.sigma(), 1.0);
    ///
    /// gauss.set_sigma(2.3).unwrap();
    /// assert_eq!(gauss.sigma(), 2.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Gaussian;
    /// # let mut gauss = Gaussian::standard();
    /// assert!(gauss.set_sigma(2.3).is_ok());
    /// assert!(gauss.set_sigma(0.0).is_err());
    /// assert!(gauss.set_sigma(-1.0).is_err());
    /// assert!(gauss.set_sigma(f64::INFINITY).is_err());
    /// assert!(gauss.set_sigma(f64::NEG_INFINITY).is_err());
    /// assert!(gauss.set_sigma(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_sigma(&mut self, sigma: f64) -> Result<(), GaussianError> {
        if sigma <= 0.0 {
            Err(GaussianError::SigmaTooLow { sigma })
        } else if !sigma.is_finite() {
            Err(GaussianError::SigmaNotFinite { sigma })
        } else {
            self.set_sigma_unchecked(sigma);
            Ok(())
        }
    }

    /// Set the value of sigma
    #[inline]
    pub fn set_sigma_unchecked(&mut self, sigma: f64) {
        self.sigma = sigma;
        self.ln_sigma = sigma.ln();
    }
}

impl Default for Gaussian {
    fn default() -> Self {
        Gaussian::standard()
    }
}

impl From<&Gaussian> for String {
    fn from(gauss: &Gaussian) -> String {
        format!("N(μ: {}, σ: {})", gauss.mu, gauss.sigma)
    }
}

impl_display!(Gaussian);

impl Shiftable for Gaussian {
    type Output = Self;
    type Error = GaussianError;

    fn shifted(self, shift: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Self::new(self.mu() + shift, self.sigma())
    }

    fn shifted_unchecked(self, shift: f64) -> Self::Output
    where
        Self: Sized,
    {
        Self::new_unchecked(self.mu() + shift, self.sigma())
    }
}

impl Scalable for Gaussian {
    type Output = Self;
    type Error = GaussianError;

    fn scaled(self, scale: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Self::new(self.mu() * scale, self.sigma() * scale)
    }

    fn scaled_unchecked(self, scale: f64) -> Self::Output
    where
        Self: Sized,
    {
        Self::new_unchecked(self.mu() * scale, self.sigma() * scale)
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for Gaussian {
            fn ln_f(&self, x: &$kind) -> f64 {
                let k = (f64::from(*x) - self.mu) / self.sigma;
                (0.5 * k).mul_add(-k, -self.ln_sigma) - HALF_LN_2PI
            }
        }

        impl Sampleable<$kind> for Gaussian {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let g = Normal::new(self.mu, self.sigma).unwrap();
                rng.sample(g) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let g = Normal::new(self.mu, self.sigma).unwrap();
                (0..n).map(|_| rng.sample(g) as $kind).collect()
            }
        }

        impl ContinuousDistr<$kind> for Gaussian {}

        impl Support<$kind> for Gaussian {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite()
            }
        }

        impl Cdf<$kind> for Gaussian {
            fn cdf(&self, x: &$kind) -> f64 {
                let errf =
                    ((f64::from(*x) - self.mu) / (self.sigma * SQRT_2)).error();
                0.5 * (1.0 + errf)
            }
        }

        impl InverseCdf<$kind> for Gaussian {
            fn invcdf(&self, p: f64) -> $kind {
                assert!((0.0..=1.0).contains(&p), "P out of range");

                let x = (self.sigma * SQRT_2)
                    .mul_add(2.0_f64.mul_add(p, -1.0).inv_error(), self.mu);
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

            fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
                // let k = (f64::from(*x) - self.mu) / self.sigma;
                // (0.5 * k).mul_add(-k, -self.ln_sigma()) - HALF_LN_2PI
                let z = (2.0 * self.sigma * self.sigma).recip();
                let n = stat.n() as f64;
                let expterm = stat.sum_x_sq()
                    + self
                        .mu
                        .mul_add(-2.0 * stat.sum_x(), n * self.mu * self.mu);
                -n.mul_add(self.ln_sigma + HALF_LN_2PI, z * expterm)
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
        HALF_LN_2PI_E + self.ln_sigma
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
    #[allow(clippy::suspicious_operation_groupings)]
    fn kl(&self, other: &Self) -> f64 {
        let m1 = self.mu;
        let m2 = other.mu;

        let s1 = self.sigma;
        let s2 = other.sigma;

        let term1 = s2.ln() - s1.ln();
        let term2 = s1.mul_add(s1, (m1 - m2) * (m1 - m2)) / (2.0 * s2 * s2);

        term1 + term2 - 0.5
    }
}

impl QuadBounds for Gaussian {
    fn quad_bounds(&self) -> (f64, f64) {
        self.interval(0.999_999_999_999)
    }
}

#[cfg(feature = "experimental")]
impl_traits!(f16);
impl_traits!(f32);
impl_traits!(f64);

impl std::error::Error for GaussianError {}

impl fmt::Display for GaussianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MuNotFinite { mu } => write!(f, "non-finite mu: {mu}"),
            Self::SigmaTooLow { sigma } => {
                write!(f, "sigma ({sigma}) must be greater than zero")
            }
            Self::SigmaNotFinite { sigma } => {
                write!(f, "non-finite sigma: {sigma}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TOL: f64 = 1E-12;

    use crate::test_basic_impls;
    test_basic_impls!(f64, Gaussian);

    use crate::test_shiftable_cdf;
    use crate::test_shiftable_density;
    use crate::test_shiftable_entropy;
    use crate::test_shiftable_invcdf;
    use crate::test_shiftable_method;

    test_shiftable_method!(Gaussian::new(2.0, 4.0).unwrap(), mean);
    test_shiftable_method!(Gaussian::new(2.0, 4.0).unwrap(), median);
    test_shiftable_method!(Gaussian::new(2.0, 4.0).unwrap(), variance);
    test_shiftable_method!(Gaussian::new(2.0, 4.0).unwrap(), skewness);
    test_shiftable_method!(Gaussian::new(2.0, 4.0).unwrap(), kurtosis);
    test_shiftable_density!(Gaussian::new(2.0, 4.0).unwrap());
    test_shiftable_entropy!(Gaussian::new(2.0, 4.0).unwrap());
    test_shiftable_cdf!(Gaussian::new(2.0, 4.0).unwrap());
    test_shiftable_invcdf!(Gaussian::new(2.0, 4.0).unwrap());

    use crate::test_scalable_cdf;
    use crate::test_scalable_density;
    use crate::test_scalable_entropy;
    use crate::test_scalable_invcdf;
    use crate::test_scalable_method;

    test_scalable_method!(Gaussian::new(2.0, 4.0).unwrap(), mean);
    test_scalable_method!(Gaussian::new(2.0, 4.0).unwrap(), median);
    test_scalable_method!(Gaussian::new(2.0, 4.0).unwrap(), variance);
    test_scalable_method!(Gaussian::new(2.0, 4.0).unwrap(), skewness);
    test_scalable_method!(Gaussian::new(2.0, 4.0).unwrap(), kurtosis);
    test_scalable_density!(Gaussian::new(2.0, 4.0).unwrap());
    test_scalable_entropy!(Gaussian::new(2.0, 4.0).unwrap());
    test_scalable_cdf!(Gaussian::new(2.0, 4.0).unwrap());
    test_scalable_invcdf!(Gaussian::new(2.0, 4.0).unwrap());

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
        let mut rng = rand::rng();
        let gauss = Gaussian::standard();
        for _ in 0..100 {
            let x: f64 = gauss.draw(&mut rng);
            assert!(x.is_finite());
        }
    }

    #[test]
    fn sample_length() {
        let mut rng = rand::rng();
        let gauss = Gaussian::standard();
        let xs: Vec<f64> = gauss.sample(10, &mut rng);
        assert_eq!(xs.len(), 10);
    }

    #[test]
    fn standard_ln_pdf_at_zero() {
        let gauss = Gaussian::standard();
        assert::close(gauss.ln_pdf(&0.0_f64), -0.918_938_533_204_672_7, TOL);
    }

    #[test]
    fn standard_ln_pdf_off_zero() {
        let gauss = Gaussian::standard();
        assert::close(gauss.ln_pdf(&2.1_f64), -3.123_938_533_204_672_7, TOL);
    }

    #[test]
    fn nonstandard_ln_pdf_on_mean() {
        let gauss = Gaussian::new(-1.2, 0.33).unwrap();
        assert::close(gauss.ln_pdf(&-1.2_f64), 0.189_724_091_316_938_46, TOL);
    }

    #[test]
    fn nonstandard_ln_pdf_off_mean() {
        let gauss = Gaussian::new(-1.2, 0.33).unwrap();
        assert::close(gauss.ln_pdf(&0.0_f32), -6.421_846_156_616_945, TOL);
    }

    #[test]
    fn should_contain_finite_values() {
        let gauss = Gaussian::standard();
        assert!(gauss.supports(&0.0_f32));
        assert!(gauss.supports(&10E8_f64));
        assert!(gauss.supports(&-10E8_f64));
    }

    #[test]
    fn should_not_contain_nan() {
        let gauss = Gaussian::standard();
        assert!(!gauss.supports(&f64::NAN));
    }

    #[test]
    fn should_not_contain_positive_or_negative_infinity() {
        let gauss = Gaussian::standard();
        assert!(!gauss.supports(&f64::INFINITY));
        assert!(!gauss.supports(&f64::NEG_INFINITY));
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
        assert::close(gauss.cdf(&1.0_f64), 0.841_344_746_068_542_9, TOL);
    }

    #[test]
    fn cdf_value_at_neg_two() {
        let gauss = Gaussian::standard();
        assert::close(gauss.cdf(&-2.0_f64), 0.022_750_131_948_179_195, TOL);
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
        let mut rng = rand::rng();
        let gauss = Gaussian::standard();
        let xs: Vec<f64> = gauss.sample(100, &mut rng);

        for x in &xs {
            let p = gauss.cdf(x);
            let y: f64 = gauss.quantile(p);
            assert::close(y, *x, TOL);
        }
    }

    #[test]
    fn quad_on_pdf_agrees_with_cdf_x() {
        use peroxide::numerical::integral::{
            gauss_kronrod_quadrature, Integral,
        };
        let ig = Gaussian::new(-2.3, 0.5).unwrap();
        let pdf = |x: f64| ig.f(&x);
        let mut rng = rand::rng();
        for _ in 0..100 {
            let x: f64 = ig.draw(&mut rng);
            let res = gauss_kronrod_quadrature(
                pdf,
                (-10.0, x),
                Integral::G7K15(1e-12, 100),
            );
            let cdf = ig.cdf(&x);
            assert::close(res, cdf, 1e-9);
        }
    }

    #[test]
    fn standard_gaussian_entropy() {
        let gauss = Gaussian::standard();
        assert::close(gauss.entropy(), 1.418_938_533_204_672_7, TOL);
    }

    #[test]
    fn entropy() {
        let gauss = Gaussian::new(3.0, 12.3).unwrap();
        assert::close(gauss.entropy(), 3.928_537_795_583_044_7, TOL);
    }

    #[test]
    fn kl_of_identical_distributions_should_be_zero() {
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

    #[test]
    fn ln_f_after_set_mu_works() {
        let mut gauss = Gaussian::standard();
        assert::close(gauss.ln_pdf(&0.0_f64), -0.918_938_533_204_672_7, TOL);

        gauss.set_mu(1.0).unwrap();
        assert::close(gauss.ln_pdf(&1.0_f64), -0.918_938_533_204_672_7, TOL);
    }

    #[test]
    fn ln_f_after_set_sigm_works() {
        let mut gauss = Gaussian::new(-1.2, 5.0).unwrap();

        gauss.set_sigma(0.33).unwrap();
        assert::close(gauss.ln_pdf(&-1.2_f64), 0.189_724_091_316_938_46, TOL);
        assert::close(gauss.ln_pdf(&0.0_f32), -6.421_846_156_616_945, TOL);
    }

    #[test]
    fn ln_f_stat() {
        use crate::traits::SuffStat;

        let data: Vec<f64> = vec![0.1, 0.23, 1.4, 0.65, 0.22, 3.1];
        let mut stat = GaussianSuffStat::new();
        stat.observe_many(&data);

        let gauss = Gaussian::new(-0.3, 2.33).unwrap();

        let ln_f_base: f64 = data.iter().map(|x| gauss.ln_f(x)).sum();
        let ln_f_stat: f64 =
            <Gaussian as HasSuffStat<f64>>::ln_f_stat(&gauss, &stat);

        assert::close(ln_f_base, ln_f_stat, TOL);
    }

    #[cfg(feature = "serde1")]
    crate::test_serde_params!(Gaussian::new(-1.3, 2.4).unwrap(), Gaussian, f64);
}
