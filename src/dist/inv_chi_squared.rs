//! Χ</sup>-2</sup> over x in (0, ∞)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::misc::ln_gammafn;
use crate::traits::*;
use rand::Rng;
use special::Gamma;
use std::f64::consts::LN_2;
use std::fmt;
use std::sync::OnceLock;

/// [Χ<sup>-2</sup> distribution](https://en.wikipedia.org/wiki/Inverse-chi-squared_distribution)
/// Χ<sup>-2</sup>(v).
///
/// # Example
///
/// ```
/// use rv::prelude::*;
///
/// let ix2 = InvChiSquared::new(2.0).unwrap();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct InvChiSquared {
    /// Degrees of freedom in (0, ∞)
    v: f64,
    // ln( 2^{-v/2} / gamma(v/2))
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_f_const: OnceLock<f64>,
}

impl PartialEq for InvChiSquared {
    fn eq(&self, other: &InvChiSquared) -> bool {
        self.v == other.v
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum InvChiSquaredError {
    /// v parameter is less than or equal to zero
    VTooLow { v: f64 },
    /// v parameter is infinite or NaN
    VNotFinite { v: f64 },
}

impl InvChiSquared {
    /// Create a new Inverse Chi-squared distribution
    ///
    /// # Arguments
    /// - v: Degrees of freedom in (0, ∞)
    #[inline]
    pub fn new(v: f64) -> Result<Self, InvChiSquaredError> {
        if v <= 0.0 {
            Err(InvChiSquaredError::VTooLow { v })
        } else if !v.is_finite() {
            Err(InvChiSquaredError::VNotFinite { v })
        } else {
            Ok(InvChiSquared {
                v,
                ln_f_const: OnceLock::new(),
            })
        }
    }

    /// Create a new InvChiSquared without checking whether the parameters are
    /// valid.
    #[inline(always)]
    pub fn new_unchecked(v: f64) -> Self {
        InvChiSquared {
            v,
            ln_f_const: OnceLock::new(),
        }
    }

    /// Get the degrees of freedom, `v`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvChiSquared;
    /// let ix2 = InvChiSquared::new(1.2).unwrap();
    /// assert_eq!(ix2.v(), 1.2);
    /// ```
    #[inline(always)]
    pub fn v(&self) -> f64 {
        self.v
    }

    /// Set the degrees of freedom, `k`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::InvChiSquared;
    /// let mut ix2 = InvChiSquared::new(1.2).unwrap();
    ///
    /// ix2.set_v(2.2).unwrap();
    /// assert_eq!(ix2.v(), 2.2);
    /// ```
    ///
    /// Will error given invalid values.
    ///
    /// ```rust
    /// # use rv::dist::InvChiSquared;
    /// # let mut ix2 = InvChiSquared::new(1.2).unwrap();
    /// assert!(ix2.set_v(2.2).is_ok());
    /// assert!(ix2.set_v(0.0).is_err());
    /// assert!(ix2.set_v(-1.0).is_err());
    /// assert!(ix2.set_v(std::f64::NAN).is_err());
    /// assert!(ix2.set_v(std::f64::INFINITY).is_err());
    /// ```
    #[inline]
    pub fn set_v(&mut self, v: f64) -> Result<(), InvChiSquaredError> {
        if !v.is_finite() {
            Err(InvChiSquaredError::VNotFinite { v })
        } else if v > 0.0 {
            self.set_v_unchecked(v);
            Ok(())
        } else {
            Err(InvChiSquaredError::VTooLow { v })
        }
    }

    #[inline(always)]
    pub fn set_v_unchecked(&mut self, v: f64) {
        self.v = v;
        self.ln_f_const = OnceLock::new();
    }

    /// Get ln( 2^{-v/2} / Gamma(v/2) )
    #[inline]
    fn ln_f_const(&self) -> f64 {
        *self.ln_f_const.get_or_init(|| {
            let v2 = self.v / 2.0;
            (-v2).mul_add(LN_2, -ln_gammafn(v2))
        })
    }
}

impl From<&InvChiSquared> for String {
    fn from(ix2: &InvChiSquared) -> String {
        format!("χ⁻²({})", ix2.v)
    }
}

impl_display!(InvChiSquared);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for InvChiSquared {
            fn ln_f(&self, x: &$kind) -> f64 {
                let x64 = f64::from(*x);
                let z = self.ln_f_const();
                (-self.v / 2.0 - 1.0).mul_add(x64.ln(), z) - (2.0 * x64).recip()
            }
        }

        impl Sampleable<$kind> for InvChiSquared {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let x2 = rand_distr::ChiSquared::new(self.v).unwrap();
                let x_inv: f64 = rng.sample(x2);
                x_inv.recip() as $kind
            }
        }

        impl Support<$kind> for InvChiSquared {
            fn supports(&self, x: &$kind) -> bool {
                *x > 0.0 && x.is_finite()
            }
        }

        impl ContinuousDistr<$kind> for InvChiSquared {}

        impl Mean<$kind> for InvChiSquared {
            fn mean(&self) -> Option<$kind> {
                if self.v > 2.0 {
                    Some(1.0 / (self.v as $kind - 2.0))
                } else {
                    None
                }
            }
        }

        impl Mode<$kind> for InvChiSquared {
            fn mode(&self) -> Option<$kind> {
                Some((1.0 / (self.v + 2.0)) as $kind)
            }
        }

        impl Variance<$kind> for InvChiSquared {
            fn variance(&self) -> Option<$kind> {
                if self.v > 4.0 {
                    let denom =
                        (self.v - 2.0) * (self.v - 2.0) * (self.v - 4.0);
                    Some((2.0 / denom) as $kind)
                } else {
                    None
                }
            }
        }

        impl Cdf<$kind> for InvChiSquared {
            fn cdf(&self, x: &$kind) -> f64 {
                let x64 = f64::from(*x);
                1.0 - (2.0 * x64).recip().inc_gamma(self.v / 2.0)
            }
        }
    };
}

impl Skewness for InvChiSquared {
    fn skewness(&self) -> Option<f64> {
        if self.v > 6.0 {
            let v = self.v;
            Some(4.0 / (v - 6.0) * (2.0 * (v - 4.0)).sqrt())
        } else {
            None
        }
    }
}

impl Kurtosis for InvChiSquared {
    fn kurtosis(&self) -> Option<f64> {
        if self.v > 8.0 {
            let v = self.v;
            Some(12.0 * 5.0_f64.mul_add(v, -22.0) / ((v - 6.0) * (v - 8.0)))
        } else {
            None
        }
    }
}

impl_traits!(f64);
impl_traits!(f32);

impl std::error::Error for InvChiSquaredError {}

impl fmt::Display for InvChiSquaredError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VTooLow { v } => {
                write!(f, "v ({}) must be greater than zero", v)
            }
            Self::VNotFinite { v } => write!(f, "v ({}) must be finite", v),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::dist::{Gamma, InvGamma};
    use crate::misc::ks_test;
    use crate::test_basic_impls;
    use std::f64;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!([continuous] InvChiSquared::new(3.2).unwrap());

    #[test]
    fn new() {
        let ix2 = InvChiSquared::new(2.3).unwrap();
        assert::close(ix2.v, 2.3, TOL);
    }

    #[test]
    fn new_should_reject_v_leq_zero() {
        assert!(InvChiSquared::new(f64::MIN_POSITIVE).is_ok());
        assert!(InvChiSquared::new(0.0).is_err());
        assert!(InvChiSquared::new(-f64::MIN_POSITIVE).is_err());
        assert!(InvChiSquared::new(-1.0).is_err());
    }

    #[test]
    fn new_should_reject_non_finite_v() {
        assert!(InvChiSquared::new(f64::INFINITY).is_err());
        assert!(InvChiSquared::new(-f64::NAN).is_err());
        assert!(InvChiSquared::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn mean_is_defined_for_v_gt_2() {
        {
            let m: Option<f64> = InvChiSquared::new_unchecked(0.1).mean();
            assert!(m.is_none());
        }
        {
            let m: Option<f64> = InvChiSquared::new_unchecked(2.0).mean();
            assert!(m.is_none());
        }
        {
            let m: Option<f64> = InvChiSquared::new_unchecked(2.000_001).mean();
            assert!(m.is_some());
        }
    }

    #[test]
    fn mean_values() {
        {
            let m: f64 = InvChiSquared::new_unchecked(2.1).mean().unwrap();
            assert::close(m, 10.0, TOL);
        }
        {
            let m: f64 = InvChiSquared::new_unchecked(4.0).mean().unwrap();
            assert::close(m, 0.5, TOL);
        }
    }

    #[test]
    fn variance_is_defined_for_v_gt_4() {
        {
            let m: Option<f64> = InvChiSquared::new_unchecked(0.1).variance();
            assert!(m.is_none());
        }
        {
            let m: Option<f64> = InvChiSquared::new_unchecked(4.0).variance();
            assert!(m.is_none());
        }
        {
            let m: Option<f64> =
                InvChiSquared::new_unchecked(4.000_001).variance();
            assert!(m.is_some());
        }
    }

    #[test]
    fn variance() {
        let v: f64 = InvChiSquared::new_unchecked(6.0).variance().unwrap();
        assert::close(v, 0.0625, TOL);
    }

    #[test]
    fn skewness() {
        let v: f64 = InvChiSquared::new_unchecked(12.0).skewness().unwrap();
        assert::close(v, 16.0 / 6.0, TOL);
    }

    #[test]
    fn kurtosis() {
        let v: f64 = InvChiSquared::new_unchecked(12.0).kurtosis().unwrap();
        assert::close(v, 19.0, TOL);
    }

    #[test]
    fn pdf_agrees_with_inv_gamma_special_case() {
        let mut rng = rand::thread_rng();
        let v_prior = Gamma::new_unchecked(2.0, 1.0);

        for v in v_prior.sample_stream(&mut rng).take(1000) {
            let ix2 = InvChiSquared::new(v).unwrap();
            let igam = InvGamma::new(v / 2.0, 0.5).unwrap();

            for x in &[0.1_f64, 1.0_f64, 14.2_f64] {
                assert::close(ix2.ln_f(x), igam.ln_f(x), TOL);
            }
        }
    }

    #[test]
    fn cdf_limits_are_0_and_1() {
        let ix2 = InvChiSquared::new(2.5).unwrap();
        assert::close(ix2.cdf(&1e-16), 0.0, TOL);
        assert::close(ix2.cdf(&1E16), 1.0, TOL);
    }

    #[test]
    fn cdf_agrees_with_inv_gamma_special_case() {
        let mut rng = rand::thread_rng();
        let v_prior = Gamma::new_unchecked(2.0, 1.0);

        for v in v_prior.sample_stream(&mut rng).take(1000) {
            let ix2 = InvChiSquared::new(v).unwrap();
            let igam = InvGamma::new(v / 2.0, 0.5).unwrap();

            for x in &[0.1_f64, 1.0_f64, 14.2_f64] {
                assert::close(ix2.cdf(x), igam.cdf(x), TOL);
            }
        }
    }

    #[test]
    fn draw_agrees_with_cdf() {
        let mut rng = rand::thread_rng();
        let ix2 = InvChiSquared::new(1.2).unwrap();
        let cdf = |x: f64| ix2.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = ix2.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }
}
