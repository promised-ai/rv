//! Inverse Gamma distribution over x in (0, ∞)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::InvGammaSuffStat;
use crate::impl_display;
use crate::misc::ln_gammafn;
use crate::traits::*;
use rand::Rng;
use special::Gamma as _;
use std::fmt;

/// [Inverse gamma distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
/// IG(α, β) over x in (0, ∞).
///
/// ```math
///             β^α
/// f(x|α, β) = ----  x^(-α-1) e^(-β/x)
///             Γ(α)
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct InvGamma {
    // shape parameter, α
    shape: f64,
    // scale parameter, β
    scale: f64,
}

pub struct InvGammaParameters {
    pub shape: f64,
    pub scale: f64,
}

impl Parameterized for InvGamma {
    type Parameters = InvGammaParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            shape: self.shape(),
            scale: self.scale(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.shape, params.scale)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum InvGammaError {
    /// Shape parameter is less than or equal to zero
    ShapeTooLow { shape: f64 },
    /// Shape parameter is infinite or NaN
    ShapeNotFinite { shape: f64 },
    /// Scale parameter is less than or equal to zero
    ScaleTooLow { scale: f64 },
    /// Scale parameter is infinite or NaN
    ScaleNotFinite { scale: f64 },
}

impl InvGamma {
    /// Create a new `Gamma` distribution with shape (α) and rate (β).
    ///
    /// # Arguments
    /// - shape: shape parameter, α
    /// - scale scale parameter, β
    pub fn new(shape: f64, scale: f64) -> Result<Self, InvGammaError> {
        if shape <= 0.0 {
            Err(InvGammaError::ShapeTooLow { shape })
        } else if scale <= 0.0 {
            Err(InvGammaError::ScaleTooLow { scale })
        } else if !shape.is_finite() {
            Err(InvGammaError::ShapeNotFinite { shape })
        } else if !scale.is_finite() {
            Err(InvGammaError::ScaleNotFinite { scale })
        } else {
            Ok(InvGamma { shape, scale })
        }
    }

    /// Creates a new InvGamma without checking whether the parameters are
    /// valid.
    #[inline]
    pub fn new_unchecked(shape: f64, scale: f64) -> Self {
        InvGamma { shape, scale }
    }

    /// Get the shape parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvGamma;
    /// let ig = InvGamma::new(1.0, 2.0).unwrap();
    /// assert_eq!(ig.shape(), 1.0);
    /// ```
    #[inline]
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Set the shape parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvGamma;
    /// let mut ig = InvGamma::new(2.0, 1.0).unwrap();
    /// assert_eq!(ig.shape(), 2.0);
    ///
    /// ig.set_shape(1.1).unwrap();
    /// assert_eq!(ig.shape(), 1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::InvGamma;
    /// # let mut ig = InvGamma::new(2.0, 1.0).unwrap();
    /// assert!(ig.set_shape(1.1).is_ok());
    /// assert!(ig.set_shape(0.0).is_err());
    /// assert!(ig.set_shape(-1.0).is_err());
    /// assert!(ig.set_shape(f64::INFINITY).is_err());
    /// assert!(ig.set_shape(f64::NEG_INFINITY).is_err());
    /// assert!(ig.set_shape(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_shape(&mut self, shape: f64) -> Result<(), InvGammaError> {
        if shape <= 0.0 {
            Err(InvGammaError::ShapeTooLow { shape })
        } else if !shape.is_finite() {
            Err(InvGammaError::ShapeNotFinite { shape })
        } else {
            self.set_shape_unchecked(shape);
            Ok(())
        }
    }

    /// Set the shape parameter without input validation
    #[inline]
    pub fn set_shape_unchecked(&mut self, shape: f64) {
        self.shape = shape;
    }

    /// Get the scale parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvGamma;
    /// let ig = InvGamma::new(1.0, 2.0).unwrap();
    /// assert_eq!(ig.scale(), 2.0);
    /// ```
    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Set the scale parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvGamma;
    /// let mut ig = InvGamma::new(2.0, 1.0).unwrap();
    /// assert_eq!(ig.scale(), 1.0);
    ///
    /// ig.set_scale(1.1).unwrap();
    /// assert_eq!(ig.scale(), 1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::InvGamma;
    /// # let mut ig = InvGamma::new(2.0, 1.0).unwrap();
    /// assert!(ig.set_scale(1.1).is_ok());
    /// assert!(ig.set_scale(0.0).is_err());
    /// assert!(ig.set_scale(-1.0).is_err());
    /// assert!(ig.set_scale(f64::INFINITY).is_err());
    /// assert!(ig.set_scale(f64::NEG_INFINITY).is_err());
    /// assert!(ig.set_scale(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_scale(&mut self, scale: f64) -> Result<(), InvGammaError> {
        if scale <= 0.0 {
            Err(InvGammaError::ScaleTooLow { scale })
        } else if !scale.is_finite() {
            Err(InvGammaError::ScaleNotFinite { scale })
        } else {
            self.set_scale_unchecked(scale);
            Ok(())
        }
    }

    /// Set the scale parameter without input validation
    #[inline]
    pub fn set_scale_unchecked(&mut self, scale: f64) {
        self.scale = scale;
    }
}

impl Default for InvGamma {
    fn default() -> Self {
        InvGamma {
            shape: 1.0,
            scale: 1.0,
        }
    }
}

impl From<&InvGamma> for String {
    fn from(invgam: &InvGamma) -> String {
        format!("IG(α: {}, β: {})", invgam.shape, invgam.scale)
    }
}

impl_display!(InvGamma);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for InvGamma {
            fn ln_f(&self, x: &$kind) -> f64 {
                // TODO: could cache ln(scale) and ln_gamma(shape)
                let xf = f64::from(*x);
                (self.shape + 1.0).mul_add(
                    -xf.ln(),
                    self.shape
                        .mul_add(self.scale.ln(), -ln_gammafn(self.shape)),
                ) - (self.scale / xf)
            }
        }

        impl Sampleable<$kind> for InvGamma {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let g = rand_distr::Gamma::new(self.shape, self.scale.recip())
                    .unwrap();
                (1.0 / rng.sample(g)) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let g = rand_distr::Gamma::new(self.shape, self.scale.recip())
                    .unwrap();
                (0..n).map(|_| (1.0 / rng.sample(g)) as $kind).collect()
            }
        }

        impl ContinuousDistr<$kind> for InvGamma {}

        impl Support<$kind> for InvGamma {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite() && *x > 0.0
            }
        }

        impl Cdf<$kind> for InvGamma {
            fn cdf(&self, x: &$kind) -> f64 {
                1.0 - (self.scale / f64::from(*x)).inc_gamma(self.shape)
            }
        }

        impl Mean<$kind> for InvGamma {
            fn mean(&self) -> Option<$kind> {
                if self.shape > 1.0 {
                    Some((self.scale / (self.shape - 1.0)) as $kind)
                } else {
                    None
                }
            }
        }

        impl Mode<$kind> for InvGamma {
            fn mode(&self) -> Option<$kind> {
                Some((self.scale / (self.shape + 1.0)) as $kind)
            }
        }

        impl HasSuffStat<$kind> for InvGamma {
            type Stat = InvGammaSuffStat;

            fn empty_suffstat(&self) -> Self::Stat {
                InvGammaSuffStat::new()
            }

            fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
                let n = stat.n() as f64;
                let ln_beta = self.scale.ln();
                let ln_gamma_alpha = ln_gammafn(self.shape);
                let t1 = n * self.shape.mul_add(ln_beta, -ln_gamma_alpha);
                let t2 = (-self.shape - 1.0) * stat.sum_ln_x();
                let t3 = self.scale * stat.sum_inv_x();
                t1 + t2 - t3
            }
        }
    };
}

impl Variance<f64> for InvGamma {
    fn variance(&self) -> Option<f64> {
        if self.shape > 2.0 {
            let numer = self.scale * self.scale;
            let denom =
                (self.shape - 1.0) * (self.shape - 1.0) * (self.shape - 2.0);
            Some(numer / denom)
        } else {
            None
        }
    }
}

impl Entropy for InvGamma {
    fn entropy(&self) -> f64 {
        (1.0 + self.shape).mul_add(
            -self.shape.digamma(),
            self.shape + self.scale.ln() + ln_gammafn(self.shape),
        )
    }
}

impl Skewness for InvGamma {
    fn skewness(&self) -> Option<f64> {
        if self.shape > 3.0 {
            Some(4.0 * (self.shape - 2.0).sqrt() / (self.shape - 3.0))
        } else {
            None
        }
    }
}

impl Kurtosis for InvGamma {
    fn kurtosis(&self) -> Option<f64> {
        if self.shape > 4.0 {
            let krt = 30.0_f64.mul_add(self.shape, -66.0)
                / ((self.shape - 3.0) * (self.shape - 4.0));
            Some(krt)
        } else {
            None
        }
    }
}

impl_traits!(f32);
impl_traits!(f64);

impl std::error::Error for InvGammaError {}

impl fmt::Display for InvGammaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeTooLow { shape } => {
                write!(f, "rate ({}) must be greater than zero", shape)
            }
            Self::ShapeNotFinite { shape } => {
                write!(f, "non-finite rate: {}", shape)
            }
            Self::ScaleTooLow { scale } => {
                write!(f, "scale ({}) must be greater than zero", scale)
            }
            Self::ScaleNotFinite { scale } => {
                write!(f, "non-finite scale: {}", scale)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;
    use crate::test_basic_impls;
    use std::f64;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!(f64, InvGamma, InvGamma::new_unchecked(1.5, 2.3));

    #[test]
    fn new() {
        let ig = InvGamma::new(1.0, 2.0).unwrap();
        assert::close(ig.shape, 1.0, TOL);
        assert::close(ig.scale, 2.0, TOL);
    }

    #[test]
    fn mean() {
        let mean: f64 = InvGamma::new(1.2, 3.4).unwrap().mean().unwrap();
        assert::close(mean, 17.000_000_000_000_004, TOL);
    }

    #[test]
    fn mean_undefined_for_shape_leq_1() {
        let m1_opt: Option<f64> = InvGamma::new(1.001, 3.4).unwrap().mean();
        let m2_opt: Option<f64> = InvGamma::new(1.0, 3.4).unwrap().mean();
        let m3_opt: Option<f64> = InvGamma::new(0.1, 3.4).unwrap().mean();
        assert!(m1_opt.is_some());
        assert!(m2_opt.is_none());
        assert!(m3_opt.is_none());
    }

    #[test]
    fn mode() {
        let m1: f64 = InvGamma::new(2.0, 3.0).unwrap().mode().unwrap();
        let m2: f64 = InvGamma::new(3.0, 2.0).unwrap().mode().unwrap();
        assert::close(m1, 1.0, TOL);
        assert::close(m2, 0.5, TOL);
    }

    #[test]
    fn variance() {
        let ig = InvGamma::new(2.3, 4.5).unwrap();
        assert::close(ig.variance().unwrap(), 39.940_828_402_366_9, TOL);
    }

    #[test]
    fn variance_undefined_for_shape_leq_2() {
        assert!(InvGamma::new(2.001, 3.4).unwrap().variance().is_some());
        assert!(InvGamma::new(2.0, 3.4).unwrap().variance().is_none());
        assert!(InvGamma::new(0.1, 3.4).unwrap().variance().is_none());
    }

    #[test]
    fn ln_pdf_low_value() {
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        assert::close(ig.ln_pdf(&0.1_f64), -9.403_365_266_903_927, TOL);
    }

    #[test]
    fn ln_pdf_at_mean() {
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        assert::close(ig.ln_pdf(&1.0_f64), -0.613_705_638_880_109_5, TOL);
    }

    #[test]
    fn ln_pdf_at_mode() {
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        assert::close(ig.ln_pdf(&0.5_f64), 0.158_883_083_359_671_6, TOL);
    }

    #[test]
    fn quad_on_pdf_agrees_with_cdf_range() {
        use peroxide::numerical::integral::{
            gauss_kronrod_quadrature, Integral,
        };
        let ig = InvGamma::new(5.2, 3.3).unwrap();
        let pdf = |x: f64| ig.f(&x);
        let res = gauss_kronrod_quadrature(
            pdf,
            (1e-16, 1000.0),
            Integral::G7K15(1e-12, 100),
        );
        assert::close(res, 1.0, 1e-9);
    }

    #[test]
    fn quad_on_pdf_agrees_with_cdf_2() {
        use peroxide::numerical::integral::{
            gauss_kronrod_quadrature, Integral,
        };
        let ig = InvGamma::new(2.3, 3.1).unwrap();
        let pdf = |x: f64| ig.f(&x);
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let x: f64 = ig.draw(&mut rng);
            let res = gauss_kronrod_quadrature(
                pdf,
                (1e-16, x),
                Integral::G7K15(1e-12, 100),
            );
            let cdf = ig.cdf(&x);
            assert::close(res, cdf, 1e-9);
        }
    }

    #[test]
    fn ln_pdf_at_mode_should_be_highest() {
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        let x: f64 = ig.mode().unwrap();
        let delta = 1E-6;

        let fx = ig.ln_pdf(&x);
        let fa = ig.ln_pdf(&(x - delta));
        let fb = ig.ln_pdf(&(x + delta));

        assert!(fx > fa);
        assert!(fx > fb);
    }

    #[test]
    fn does_not_contain_negative_values() {
        assert!(!InvGamma::new(1.0, 1.0).unwrap().supports(&-0.000_001_f64));
        assert!(!InvGamma::new(1.0, 1.0).unwrap().supports(&-1.0_f64));
    }

    #[test]
    fn does_not_contain_zero() {
        assert!(!InvGamma::new(1.0, 1.0).unwrap().supports(&0.0_f32));
    }

    #[test]
    fn contains_positive_values() {
        assert!(InvGamma::new(1.0, 1.0).unwrap().supports(&0.000_001_f64));
        assert!(InvGamma::new(1.0, 1.0).unwrap().supports(&1.0_f64));
    }

    #[test]
    fn does_not_contain_infinity() {
        assert!(!InvGamma::new(1.0, 1.0).unwrap().supports(&f64::INFINITY));
    }

    #[test]
    fn sample_return_correct_number_of_draws() {
        let mut rng = rand::thread_rng();
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        let xs: Vec<f64> = ig.sample(103, &mut rng);
        assert_eq!(xs.len(), 103);
    }

    #[test]
    fn draw_always_returns_results_in_support() {
        let mut rng = rand::thread_rng();
        let ig = InvGamma::new(3.0, 2.0).unwrap();
        for _ in 0..100 {
            let x: f64 = ig.draw(&mut rng);
            assert!(x > 0.0 && x.is_finite());
        }
    }

    #[test]
    fn cdf_at_1() {
        let ig = InvGamma::new(1.2, 3.4).unwrap();
        assert::close(ig.cdf(&1.0), 0.048_714_368_540_659_62, TOL);
    }

    #[test]
    fn cdf_at_mean() {
        let ig = InvGamma::new(1.2, 3.4).unwrap();
        assert::close(ig.cdf(&17.0), 0.881_851_180_324_275_2, TOL);
    }

    #[test]
    fn skewness() {
        let ig = InvGamma::new(5.2, 2.4).unwrap();
        assert::close(ig.skewness().unwrap(), 3.252_462_512_726_966_6, TOL);
    }

    #[test]
    fn skewness_undefined_for_alpha_leq_3() {
        assert!(InvGamma::new(3.001, 3.4).unwrap().skewness().is_some());
        assert!(InvGamma::new(3.0, 3.4).unwrap().skewness().is_none());
        assert!(InvGamma::new(0.1, 3.4).unwrap().skewness().is_none());
    }

    #[test]
    fn kurtosis() {
        let ig = InvGamma::new(5.2, 2.4).unwrap();
        assert::close(ig.kurtosis().unwrap(), 34.090_909_090_909_086, TOL);
    }

    #[test]
    fn kurtosis_undefined_for_alpha_leq_4() {
        assert!(InvGamma::new(4.001, 3.4).unwrap().kurtosis().is_some());
        assert!(InvGamma::new(4.0, 3.4).unwrap().kurtosis().is_none());
        assert!(InvGamma::new(0.1, 3.4).unwrap().kurtosis().is_none());
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let ig = InvGamma::new(1.2, 3.4).unwrap();
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
        let data: Vec<f64> = vec![0.1, 0.23, 1.4, 0.65, 0.22, 3.1];
        let mut stat = InvGammaSuffStat::new();
        stat.observe_many(&data);

        let igam = InvGamma::new(0.3, 2.33).unwrap();

        let ln_f_base: f64 = data.iter().map(|x| igam.ln_f(x)).sum();
        let ln_f_stat: f64 =
            <InvGamma as HasSuffStat<f64>>::ln_f_stat(&igam, &stat);

        assert::close(ln_f_base, ln_f_stat, 1e-12);
    }
}
