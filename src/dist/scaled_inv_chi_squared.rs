//! Χ<sup>-2</sup> over x in (0, ∞)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::misc::ln_gammafn;
use crate::traits::*;

use rand::Rng;
use special::Gamma as _;
use std::fmt;
use std::sync::OnceLock;

/// Scaled [Χ<sup>-2</sup> distribution](https://en.wikipedia.org/wiki/Scaled_inverse_chi-squared_distribution)
/// Scaled-Χ<sup>-2</sup>(v, τ<sup>2</sup>).
///
/// # Example
///
/// ```
/// use rv::prelude::*;
///
/// let ix2 = ScaledInvChiSquared::new(2.0, 1.0).unwrap();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct ScaledInvChiSquared {
    /// Degrees of freedom in (0, ∞)
    v: f64,
    t2: f64,
    // ln Gamma(v/2)
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_gamma_v_2: OnceLock<f64>,
    // ln (t2*v/2)^(v/2)
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_f_const: OnceLock<f64>,
}

use crate::impl_shiftable;
impl_shiftable!(ScaledInvChiSquared);

pub struct ScaledInvChiSquaredParameters {
    pub v: f64,
    pub t2: f64,
}

impl Parameterized for ScaledInvChiSquared {
    type Parameters = ScaledInvChiSquaredParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            v: self.v(),
            t2: self.t2(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.v, params.t2)
    }
}

impl PartialEq for ScaledInvChiSquared {
    fn eq(&self, other: &ScaledInvChiSquared) -> bool {
        self.v == other.v
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum ScaledInvChiSquaredError {
    /// v parameter is less than or equal to zero
    VTooLow { v: f64 },
    /// v parameter is infinite or NaN
    VNotFinite { v: f64 },
    /// τ<sup>2</sup> parameter is less than or equal to zero
    T2TooLow { t2: f64 },
    /// τ<sup>2</sup> parameter is infinite or NaN
    T2NotFinite { t2: f64 },
}

impl ScaledInvChiSquared {
    /// Create a new Inverse Chi-squared distribution
    ///
    /// # Arguments
    /// - v: Degrees of freedom in (0, ∞)
    /// - t2: Scale factor in (0, ∞)
    #[inline]
    pub fn new(v: f64, t2: f64) -> Result<Self, ScaledInvChiSquaredError> {
        if v <= 0.0 {
            Err(ScaledInvChiSquaredError::VTooLow { v })
        } else if t2 <= 0.0 {
            Err(ScaledInvChiSquaredError::T2TooLow { t2 })
        } else if !v.is_finite() {
            Err(ScaledInvChiSquaredError::VNotFinite { v })
        } else if !t2.is_finite() {
            Err(ScaledInvChiSquaredError::T2NotFinite { t2 })
        } else {
            Ok(ScaledInvChiSquared {
                v,
                t2,
                ln_gamma_v_2: OnceLock::new(),
                ln_f_const: OnceLock::new(),
            })
        }
    }

    /// Create a new ScaledInvChiSquared without checking whether the parameters are
    /// valid.
    #[inline(always)]
    pub fn new_unchecked(v: f64, t2: f64) -> Self {
        ScaledInvChiSquared {
            v,
            t2,
            ln_gamma_v_2: OnceLock::new(),
            ln_f_const: OnceLock::new(),
        }
    }

    /// Get the degrees of freedom, `v`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::ScaledInvChiSquared;
    /// let ix2 = ScaledInvChiSquared::new(1.2, 3.4).unwrap();
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
    /// # use rv::dist::ScaledInvChiSquared;
    /// let mut ix2 = ScaledInvChiSquared::new(1.2, 3.4).unwrap();
    ///
    /// ix2.set_v(2.2).unwrap();
    /// assert_eq!(ix2.v(), 2.2);
    /// ```
    ///
    /// Will error given invalid values.
    ///
    /// ```rust
    /// # use rv::dist::ScaledInvChiSquared;
    /// # let mut ix2 = ScaledInvChiSquared::new(1.2, 3.4).unwrap();
    /// assert!(ix2.set_v(2.2).is_ok());
    /// assert!(ix2.set_v(0.0).is_err());
    /// assert!(ix2.set_v(-1.0).is_err());
    /// assert!(ix2.set_v(f64::NAN).is_err());
    /// assert!(ix2.set_v(f64::INFINITY).is_err());
    /// ```
    #[inline]
    pub fn set_v(&mut self, v: f64) -> Result<(), ScaledInvChiSquaredError> {
        if !v.is_finite() {
            Err(ScaledInvChiSquaredError::VNotFinite { v })
        } else if v > 0.0 {
            self.set_v_unchecked(v);
            Ok(())
        } else {
            Err(ScaledInvChiSquaredError::VTooLow { v })
        }
    }

    #[inline(always)]
    pub fn set_v_unchecked(&mut self, v: f64) {
        self.v = v;
        self.ln_gamma_v_2 = OnceLock::new();
        self.ln_f_const = OnceLock::new();
    }

    /// Get the scale factor `t2`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::ScaledInvChiSquared;
    /// let ix2 = ScaledInvChiSquared::new(1.2, 3.4).unwrap();
    /// assert_eq!(ix2.t2(), 3.4);
    /// ```
    #[inline(always)]
    pub fn t2(&self) -> f64 {
        self.t2
    }

    /// Set the scale factor `t2`
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::ScaledInvChiSquared;
    /// let mut ix2 = ScaledInvChiSquared::new(1.2, 3.4).unwrap();
    ///
    /// ix2.set_t2(2.2).unwrap();
    /// assert_eq!(ix2.t2(), 2.2);
    /// ```
    ///
    /// Will error given invalid values.
    ///
    /// ```rust
    /// # use rv::dist::ScaledInvChiSquared;
    /// # let mut ix2 = ScaledInvChiSquared::new(1.2, 3.4).unwrap();
    /// assert!(ix2.set_t2(2.2).is_ok());
    /// assert!(ix2.set_t2(0.0).is_err());
    /// assert!(ix2.set_t2(-1.0).is_err());
    /// assert!(ix2.set_t2(f64::NAN).is_err());
    /// assert!(ix2.set_t2(f64::INFINITY).is_err());
    /// ```
    #[inline]
    pub fn set_t2(&mut self, t2: f64) -> Result<(), ScaledInvChiSquaredError> {
        if !t2.is_finite() {
            Err(ScaledInvChiSquaredError::T2NotFinite { t2 })
        } else if t2 > 0.0 {
            self.set_t2_unchecked(t2);
            Ok(())
        } else {
            Err(ScaledInvChiSquaredError::T2TooLow { t2 })
        }
    }

    #[inline(always)]
    pub fn set_t2_unchecked(&mut self, t2: f64) {
        self.t2 = t2;
        self.ln_gamma_v_2 = OnceLock::new();
        self.ln_f_const = OnceLock::new();
    }

    /// Get ln Gamma(v/2)
    #[inline]
    fn ln_gamma_v_2(&self) -> f64 {
        *self.ln_gamma_v_2.get_or_init(|| {
            let v2 = self.v / 2.0;
            ln_gammafn(v2)
        })
    }

    /// Get ln (t2*v/2)^(v/2)
    #[inline]
    fn ln_f_const(&self) -> f64 {
        *self
            .ln_f_const
            .get_or_init(|| 0.5 * self.v * (self.t2 * self.v * 0.5).ln())
    }
}

impl From<&ScaledInvChiSquared> for String {
    fn from(ix2: &ScaledInvChiSquared) -> String {
        format!("Scaled-χ⁻²({}, {})", ix2.v, ix2.t2)
    }
}

impl_display!(ScaledInvChiSquared);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for ScaledInvChiSquared {
            fn ln_f(&self, x: &$kind) -> f64 {
                let x64 = f64::from(*x);
                let term_1 = -self.v * self.t2 / (2.0 * x64);
                let term_2 = self.v.mul_add(0.5, 1.0) * x64.ln();
                self.ln_f_const() - self.ln_gamma_v_2() + term_1 - term_2
            }
        }

        impl Sampleable<$kind> for ScaledInvChiSquared {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let a = 0.5 * self.v;
                let b = 0.5 * self.v * self.t2;
                let ig = crate::dist::InvGamma::new_unchecked(a, b);
                ig.draw(rng)
            }
        }

        impl Support<$kind> for ScaledInvChiSquared {
            fn supports(&self, x: &$kind) -> bool {
                *x > 0.0 && x.is_finite()
            }
        }

        impl ContinuousDistr<$kind> for ScaledInvChiSquared {}

        impl Mean<$kind> for ScaledInvChiSquared {
            fn mean(&self) -> Option<$kind> {
                if self.v > 2.0 {
                    let mean = (self.v * self.t2) / (self.v - 2.0);
                    Some(mean as $kind)
                } else {
                    None
                }
            }
        }

        impl Mode<$kind> for ScaledInvChiSquared {
            fn mode(&self) -> Option<$kind> {
                Some(((self.v * self.t2) / (self.v + 2.0)) as $kind)
            }
        }

        impl Variance<$kind> for ScaledInvChiSquared {
            fn variance(&self) -> Option<$kind> {
                if self.v > 4.0 {
                    let numer = 2.0 * self.v * self.v * self.t2 * self.t2;
                    let v_minus_2 = self.v - 2.0;
                    let denom = v_minus_2 * v_minus_2 * (self.v - 4.0);
                    Some((numer / denom) as $kind)
                } else {
                    None
                }
            }
        }

        impl Cdf<$kind> for ScaledInvChiSquared {
            fn cdf(&self, x: &$kind) -> f64 {
                let x64 = f64::from(*x);
                1.0 - (self.v * self.t2 / (2.0 * x64)).inc_gamma(self.v / 2.0)
            }
        }
    };
}

impl Skewness for ScaledInvChiSquared {
    fn skewness(&self) -> Option<f64> {
        if self.v > 6.0 {
            let v = self.v;
            Some(4.0 / (v - 6.0) * (2.0 * (v - 4.0)).sqrt())
        } else {
            None
        }
    }
}

impl Kurtosis for ScaledInvChiSquared {
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

impl std::error::Error for ScaledInvChiSquaredError {}

impl fmt::Display for ScaledInvChiSquaredError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VTooLow { v } => {
                write!(f, "v ({}) must be greater than zero", v)
            }
            Self::VNotFinite { v } => write!(f, "v ({}) must be finite", v),
            Self::T2TooLow { t2 } => {
                write!(f, "t2 ({}) must be greater than zero", t2)
            }
            Self::T2NotFinite { t2 } => write!(f, "t2 ({}) must be finite", t2),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::dist::{Gamma, InvGamma};
    use crate::misc::ks_test;
    use crate::{test_basic_impls, verify_cache_resets};
    use std::f64;
    use std::f64::consts::PI;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!(
        f64,
        ScaledInvChiSquared,
        ScaledInvChiSquared::new(3.2, 1.4).unwrap()
    );

    #[test]
    fn new() {
        let ix2 = ScaledInvChiSquared::new(2.3, 3.4).unwrap();
        assert::close(ix2.v, 2.3, TOL);
        assert::close(ix2.t2, 3.4, TOL);
    }

    #[test]
    fn new_should_reject_v_leq_zero() {
        assert!(ScaledInvChiSquared::new(f64::MIN_POSITIVE, 1.2).is_ok());
        assert!(ScaledInvChiSquared::new(0.0, 1.2).is_err());
        assert!(ScaledInvChiSquared::new(-f64::MIN_POSITIVE, 1.2).is_err());
        assert!(ScaledInvChiSquared::new(-1.0, 1.3).is_err());
    }

    #[test]
    fn new_should_reject_non_finite_v() {
        assert!(ScaledInvChiSquared::new(f64::INFINITY, 1.2).is_err());
        assert!(ScaledInvChiSquared::new(-f64::NAN, 1.2).is_err());
        assert!(ScaledInvChiSquared::new(f64::NEG_INFINITY, 1.2).is_err());
    }

    #[test]
    fn mean_is_defined_for_v_gt_2() {
        {
            let m: Option<f64> =
                ScaledInvChiSquared::new_unchecked(0.1, 1.2).mean();
            assert!(m.is_none());
        }
        {
            let m: Option<f64> =
                ScaledInvChiSquared::new_unchecked(2.0, 1.2).mean();
            assert!(m.is_none());
        }
        {
            let m: Option<f64> =
                ScaledInvChiSquared::new_unchecked(2.000_001, 1.2).mean();
            assert!(m.is_some());
        }
    }

    #[test]
    fn mean_values() {
        {
            let m: f64 =
                ScaledInvChiSquared::new_unchecked(2.1, 2.0).mean().unwrap();
            assert::close(m, 42.0, TOL);
        }
        {
            let m: f64 =
                ScaledInvChiSquared::new_unchecked(4.0, 2.1).mean().unwrap();
            assert::close(m, 4.2, TOL);
        }
    }

    #[test]
    fn variance_is_defined_for_v_gt_4() {
        {
            let m: Option<f64> =
                ScaledInvChiSquared::new_unchecked(0.1, 1.0).variance();
            assert!(m.is_none());
        }
        {
            let m: Option<f64> =
                ScaledInvChiSquared::new_unchecked(4.0, 1.0).variance();
            assert!(m.is_none());
        }
        {
            let m: Option<f64> =
                ScaledInvChiSquared::new_unchecked(4.000_001, 1.0).variance();
            assert!(m.is_some());
        }
    }

    #[test]
    fn variance() {
        let v: f64 = ScaledInvChiSquared::new_unchecked(6.0, 2.0)
            .variance()
            .unwrap();
        assert::close(v, 9.0, TOL);
    }

    #[test]
    fn skewness() {
        let v: f64 = ScaledInvChiSquared::new_unchecked(12.0, 2.0)
            .skewness()
            .unwrap();
        assert::close(v, 16.0 / 6.0, TOL);
    }

    #[test]
    fn kurtosis() {
        let v: f64 = ScaledInvChiSquared::new_unchecked(12.0, 1.0)
            .kurtosis()
            .unwrap();
        assert::close(v, 19.0, TOL);
    }

    #[test]
    fn pdf_agrees_with_inv_gamma_special_case() {
        let mut rng = rand::thread_rng();
        let v_prior = Gamma::new_unchecked(2.0, 1.0);
        let t2_prior = Gamma::new_unchecked(2.0, 1.0);

        for _ in 0..1000 {
            let v: f64 = v_prior.draw(&mut rng);
            let t2: f64 = t2_prior.draw(&mut rng);
            let ix2 = ScaledInvChiSquared::new(v, t2).unwrap();

            let a = v / 2.0;
            let b = v * t2 / 2.0;
            let igam = InvGamma::new(a, b).unwrap();

            for x in &[0.1_f64, 1.0_f64, 14.2_f64] {
                assert::close(ix2.ln_f(x), igam.ln_f(x), TOL);
            }
        }
    }

    #[test]
    fn cdf_limits_are_0_and_1() {
        let ix2 = ScaledInvChiSquared::new(2.5, 3.4).unwrap();
        assert::close(ix2.cdf(&1e-16), 0.0, TOL);
        assert::close(ix2.cdf(&1E16), 1.0, TOL);
    }

    #[test]
    fn cdf_agrees_with_inv_gamma_special_case() {
        let mut rng = rand::thread_rng();
        let v_prior = Gamma::new_unchecked(2.0, 1.0);
        let t2_prior = Gamma::new_unchecked(2.0, 1.0);

        for _ in 0..1000 {
            let v: f64 = v_prior.draw(&mut rng);
            let t2: f64 = t2_prior.draw(&mut rng);
            let ix2 = ScaledInvChiSquared::new(v, t2).unwrap();

            let a = v / 2.0;
            let b = v * t2 / 2.0;
            let igam = InvGamma::new(a, b).unwrap();

            for x in &[0.1_f64, 1.0_f64, 14.2_f64] {
                assert::close(ix2.cdf(x), igam.cdf(x), TOL);
            }
        }
    }

    #[test]
    fn draw_agrees_with_cdf() {
        let mut rng = rand::thread_rng();
        let ix2 = ScaledInvChiSquared::new(1.2, 3.4).unwrap();
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

    verify_cache_resets!(
        [unchecked],
        ln_f_is_same_after_reset_unchecked_v_identically,
        set_v_unchecked,
        ScaledInvChiSquared::new(1.2, 3.4).unwrap(),
        4.5,
        1.2,
        PI
    );

    verify_cache_resets!(
        [checked],
        ln_f_is_same_after_reset_checked_v_identically,
        set_v,
        ScaledInvChiSquared::new(1.2, 3.4).unwrap(),
        4.5,
        1.2,
        PI
    );

    verify_cache_resets!(
        [unchecked],
        ln_f_is_same_after_reset_unchecked_t2_identically,
        set_t2_unchecked,
        ScaledInvChiSquared::new(1.2, 3.4).unwrap(),
        4.5,
        3.4,
        0.2
    );

    verify_cache_resets!(
        [checked],
        ln_f_is_same_after_reset_checked_t2_identically,
        set_t2,
        ScaledInvChiSquared::new(1.2, 3.4).unwrap(),
        4.5,
        3.4,
        0.2
    );
}
