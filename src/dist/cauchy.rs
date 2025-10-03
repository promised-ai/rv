//! Cauchy distribution over x in (-∞, ∞)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::consts::LN_PI;
use crate::impl_display;
use crate::traits::{
    Cdf, ContinuousDistr, Entropy, HasDensity, InverseCdf, Median, Mode,
    Parameterized, Sampleable, Scalable, Shiftable, Support,
};
use rand::Rng;
use rand_distr::Cauchy as RCauchy;
use std::f64::consts::{FRAC_1_PI, PI};
use std::fmt;

/// [Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
/// over x in (-∞, ∞).
///
/// # Example
/// ```
/// use rv::prelude::*;
///
/// let cauchy = Cauchy::new(1.2, 3.4).expect("Invalid params");
/// let ln_fx = cauchy.ln_pdf(&0.2_f64); // -2.4514716152673368
///
/// assert!((ln_fx + 2.4514716152673368).abs() < 1E-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Cauchy {
    /// location, x<sub>0</sub>, in (-∞, ∞)
    loc: f64,
    /// scale, γ, in (0, ∞)
    scale: f64,
}
pub struct CauchyParameters {
    pub loc: f64,
    pub scale: f64,
}

impl Parameterized for Cauchy {
    type Parameters = CauchyParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            loc: self.loc(),
            scale: self.scale(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.loc, params.scale)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum CauchyError {
    /// Location parameter is infinite or NaN
    LocNotFinite { loc: f64 },
    /// Scale parameter is less than or equal to zero
    ScaleTooLow { scale: f64 },
    /// Scale parameter is infinite or Na]
    ScaleNotFinite { scale: f64 },
}

impl Cauchy {
    /// Creates a new Cauchy distribution
    ///
    /// # Arguments
    /// - loc: location, x<sub>0</sub>, in (-∞, ∞)
    /// - scale: scale, γ, in (0, ∞)
    pub fn new(loc: f64, scale: f64) -> Result<Self, CauchyError> {
        if !loc.is_finite() {
            Err(CauchyError::LocNotFinite { loc })
        } else if scale <= 0.0 {
            Err(CauchyError::ScaleTooLow { scale })
        } else if !scale.is_finite() {
            Err(CauchyError::ScaleNotFinite { scale })
        } else {
            Ok(Cauchy { loc, scale })
        }
    }

    /// Create a new Cauchy without checking whether the parameters are valid.
    #[inline]
    #[must_use]
    pub fn new_unchecked(loc: f64, scale: f64) -> Self {
        Cauchy { loc, scale }
    }

    /// Get the location parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Cauchy;
    /// let c = Cauchy::new(0.1, 1.0).unwrap();
    /// assert_eq!(c.loc(), 0.1);
    /// ```
    #[inline]
    #[must_use]
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// Set the location parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Cauchy;
    /// let mut c = Cauchy::new(0.1, 1.0).unwrap();
    /// assert_eq!(c.loc(), 0.1);
    ///
    /// c.set_loc(2.0).unwrap();
    /// assert_eq!(c.loc(), 2.0);
    /// ```
    /// Will error for invalid parameters
    ///
    /// ```rust
    /// # use rv::dist::Cauchy;
    /// # let mut c = Cauchy::new(0.1, 1.0).unwrap();
    /// assert!(c.set_loc(2.0).is_ok());
    /// assert!(c.set_loc(f64::INFINITY).is_err());
    /// assert!(c.set_loc(f64::NEG_INFINITY).is_err());
    /// assert!(c.set_loc(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_loc(&mut self, loc: f64) -> Result<(), CauchyError> {
        if loc.is_finite() {
            self.set_loc_unchecked(loc);
            Ok(())
        } else {
            Err(CauchyError::LocNotFinite { loc })
        }
    }

    /// Set the location parameter without input validation
    #[inline]
    pub fn set_loc_unchecked(&mut self, loc: f64) {
        self.loc = loc;
    }

    /// Get the scale parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Cauchy;
    /// let c = Cauchy::new(0.1, 1.0).unwrap();
    /// assert_eq!(c.scale(), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Set the scale parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Cauchy;
    /// let mut c = Cauchy::new(0.1, 1.0).unwrap();
    /// assert_eq!(c.scale(), 1.0);
    ///
    /// c.set_scale(2.1).unwrap();
    /// assert_eq!(c.scale(), 2.1);
    /// ```
    ///
    /// Will error for invalid scale.
    /// ```
    /// # use rv::dist::Cauchy;
    /// # let mut c = Cauchy::new(0.1, 1.0).unwrap();
    /// assert!(c.set_scale(0.0).is_err());
    /// assert!(c.set_scale(-1.0).is_err());
    /// assert!(c.set_scale(f64::NAN).is_err());
    /// assert!(c.set_scale(f64::INFINITY).is_err());
    /// ```
    #[inline]
    pub fn set_scale(&mut self, scale: f64) -> Result<(), CauchyError> {
        if !scale.is_finite() {
            Err(CauchyError::ScaleNotFinite { scale })
        } else if scale > 0.0 {
            self.set_scale_unchecked(scale);
            Ok(())
        } else {
            Err(CauchyError::ScaleTooLow { scale })
        }
    }

    /// Set scale parameter without input validation
    #[inline]
    pub fn set_scale_unchecked(&mut self, scale: f64) {
        self.scale = scale;
    }
}

impl Default for Cauchy {
    fn default() -> Self {
        Cauchy::new_unchecked(0.0, 1.0)
    }
}

impl From<&Cauchy> for String {
    fn from(cauchy: &Cauchy) -> String {
        format!("Cauchy(loc: {}, scale: {})", cauchy.loc, cauchy.scale)
    }
}

impl_display!(Cauchy);
use crate::misc::logaddexp;
macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for Cauchy {
            fn ln_f(&self, x: &$kind) -> f64 {
                let ln_scale = self.scale.ln();
                let term = 2.0_f64.mul_add(
                    ((f64::from(*x) - self.loc).abs().ln() - ln_scale),
                    ln_scale,
                );
                -logaddexp(ln_scale, term) - LN_PI
            }
        }

        impl Sampleable<$kind> for Cauchy {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let cauchy = RCauchy::new(self.loc, self.scale).unwrap();
                rng.sample(cauchy) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let cauchy = RCauchy::new(self.loc, self.scale).unwrap();
                (0..n).map(|_| rng.sample(cauchy) as $kind).collect()
            }
        }

        impl Support<$kind> for Cauchy {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite()
            }
        }

        impl ContinuousDistr<$kind> for Cauchy {}

        impl Cdf<$kind> for Cauchy {
            fn cdf(&self, x: &$kind) -> f64 {
                FRAC_1_PI.mul_add(
                    ((f64::from(*x) - self.loc) / self.scale).atan(),
                    0.5,
                )
            }
        }

        impl InverseCdf<$kind> for Cauchy {
            fn invcdf(&self, p: f64) -> $kind {
                self.scale.mul_add((PI * (p - 0.5)).tan(), self.loc) as $kind
            }
        }

        impl Median<$kind> for Cauchy {
            fn median(&self) -> Option<$kind> {
                Some(self.loc as $kind)
            }
        }

        impl Mode<$kind> for Cauchy {
            fn mode(&self) -> Option<$kind> {
                Some(self.loc as $kind)
            }
        }
    };
}

impl Entropy for Cauchy {
    fn entropy(&self) -> f64 {
        (4.0 * PI * self.scale).ln()
    }
}

impl_traits!(f64);
impl_traits!(f32);

impl std::error::Error for CauchyError {}

impl fmt::Display for CauchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocNotFinite { loc } => {
                write!(f, "loc ({loc}) must be finite")
            }
            Self::ScaleTooLow { scale } => {
                write!(f, "scale ({scale}) must be greater than zero")
            }
            Self::ScaleNotFinite { scale } => {
                write!(f, "scale ({scale}) must be finite")
            }
        }
    }
}

impl Shiftable for Cauchy {
    type Output = Cauchy;
    type Error = CauchyError;

    fn shifted(self, shift: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Cauchy::new(self.loc() + shift, self.scale())
    }

    fn shifted_unchecked(self, shift: f64) -> Self::Output
    where
        Self: Sized,
    {
        Cauchy::new_unchecked(self.loc() + shift, self.scale())
    }
}
impl Scalable for Cauchy {
    type Output = Cauchy;
    type Error = CauchyError;

    fn scaled(self, scale: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Cauchy::new(self.loc() * scale, self.scale() * scale)
    }

    fn scaled_unchecked(self, scale: f64) -> Self::Output
    where
        Self: Sized,
    {
        Cauchy::new_unchecked(self.loc() * scale, self.scale() * scale)
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

    test_basic_impls!(f64, Cauchy);

    #[test]
    fn ln_pdf_loc_zero() {
        let c = Cauchy::new(0.0, 1.0).unwrap();
        assert::close(c.ln_pdf(&0.2), -1.183_950_599_002_681_5, TOL);
    }

    #[test]
    fn ln_pdf_loc_nonzero() {
        let c = Cauchy::new(1.2, 3.4).unwrap();
        assert::close(c.ln_pdf(&0.2), -2.451_471_615_267_337, TOL);
    }

    #[test]
    fn cdf_at_loc() {
        let c = Cauchy::new(1.2, 3.4).unwrap();
        assert::close(c.cdf(&1.2), 0.5, TOL);
    }

    #[test]
    fn cdf_off_loc() {
        let c = Cauchy::new(1.2, 3.4).unwrap();
        assert::close(c.cdf(&2.2), 0.591_053_001_855_748_8, TOL);
        assert::close(c.cdf(&0.2), 1.0 - 0.591_053_001_855_748_8, TOL);
    }

    #[test]
    fn inv_cdf_ident() {
        let mut rng = rand::rng();
        let c = Cauchy::default();
        for _ in 0..100 {
            let x: f64 = c.draw(&mut rng);
            let p: f64 = c.cdf(&x);
            let y: f64 = c.invcdf(p);
            assert::close(x, y, 1E-8); // allow a little more error
        }
    }

    #[test]
    fn inv_cdf() {
        let c = Cauchy::new(1.2, 3.4).unwrap();
        let lower: f64 = c.invcdf(0.4);
        let upper: f64 = c.invcdf(0.6);
        assert::close(lower, 0.095_273_032_808_118_61, TOL);
        assert::close(upper, 2.304_726_967_191_881_3, TOL);
    }

    #[test]
    fn median_should_be_loc() {
        let m: f64 = Cauchy::new(1.2, 3.4).unwrap().median().unwrap();
        assert::close(m, 1.2, TOL);
    }

    #[test]
    fn mode_should_be_loc() {
        let m: f64 = Cauchy::new(-0.2, 3.4).unwrap().median().unwrap();
        assert::close(m, -0.2, TOL);
    }

    #[test]
    fn finite_numbers_should_be_in_support() {
        let c = Cauchy::default();
        assert!(c.supports(&0.0_f64));
        assert!(c.supports(&f64::MIN_POSITIVE));
        assert!(c.supports(&f64::MAX));
        assert!(c.supports(&f64::MIN));
    }

    #[test]
    fn non_finite_numbers_should_not_be_in_support() {
        let c = Cauchy::default();
        assert!(!c.supports(&f64::INFINITY));
        assert!(!c.supports(&f64::NEG_INFINITY));
        assert!(!c.supports(&f64::NAN));
    }

    #[test]
    fn entropy() {
        let c = Cauchy::new(1.2, 3.4).unwrap();
        assert::close(c.entropy(), 3.754_799_678_591_406_4, TOL);
    }

    #[test]
    fn loc_does_not_affect_entropy() {
        let c1 = Cauchy::new(1.2, 3.4).unwrap();
        let c2 = Cauchy::new(-99999.9, 3.4).unwrap();
        assert::close(c1.entropy(), c2.entropy(), TOL);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::rng();
        let c = Cauchy::new(1.2, 3.4).unwrap();
        let cdf = |x: f64| c.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = c.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL { acc + 1 } else { acc }
        });

        assert!(passes > 0);
    }

    use crate::test_shiftable_cdf;
    use crate::test_shiftable_density;
    use crate::test_shiftable_entropy;
    use crate::test_shiftable_invcdf;
    use crate::test_shiftable_method;

    test_shiftable_method!(Cauchy::new(2.0, 4.0).unwrap(), median);
    test_shiftable_density!(Cauchy::new(2.0, 4.0).unwrap());
    test_shiftable_entropy!(Cauchy::new(2.0, 4.0).unwrap());
    test_shiftable_cdf!(Cauchy::new(2.0, 4.0).unwrap());
    test_shiftable_invcdf!(Cauchy::new(2.0, 4.0).unwrap());

    use crate::test_scalable_cdf;
    use crate::test_scalable_density;
    use crate::test_scalable_entropy;
    use crate::test_scalable_invcdf;
    use crate::test_scalable_method;

    test_scalable_method!(Cauchy::new(2.0, 4.0).unwrap(), median);
    test_scalable_density!(Cauchy::new(2.0, 4.0).unwrap());
    test_scalable_entropy!(Cauchy::new(2.0, 4.0).unwrap());
    test_scalable_cdf!(Cauchy::new(2.0, 4.0).unwrap());
    test_scalable_invcdf!(Cauchy::new(2.0, 4.0).unwrap());
}
