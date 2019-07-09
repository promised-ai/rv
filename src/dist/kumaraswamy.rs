//! Kumaraswamy distribution over x in (0, 1)
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::consts::EULER_MASCERONI;
use crate::impl_display;
use crate::result;
use crate::traits::*;
use rand::Rng;
use special::Gamma as _;
use std::f64;

/// [Kumaraswamy distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution),
/// Kumaraswamy(α, β) over x in (0, 1).
///
/// # Examples
///
/// The relationship between the CDF and the inverse CDF.
///
/// ```
/// # use rv::prelude::*;
///
/// let kuma = Kumaraswamy::new(2.1, 3.4).unwrap();
///
/// let x = 0.6_f64;
/// let p = kuma.cdf(&x);
/// let y = kuma.invcdf(p);
///
/// assert::close(x, y, 1E-10);
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Kumaraswamy {
    a: f64,
    b: f64,
    ab_ln: f64,
}

impl Default for Kumaraswamy {
    fn default() -> Self {
        Kumaraswamy::uniform()
    }
}

impl From<&Kumaraswamy> for String {
    fn from(kuma: &Kumaraswamy) -> String {
        format!("Kumaraswamy(a: {}, b: {})", kuma.a, kuma.b)
    }
}

impl_display!(Kumaraswamy);

impl Kumaraswamy {
    /// Create a `Beta` distribution with even density over (0, 1).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    ///
    /// let kuma_good = Kumaraswamy::new(1.0, 1.0);
    /// assert!(kuma_good.is_ok());
    ///
    /// // Invalid negative parameter
    /// let kuma_bad  = Kumaraswamy::new(-5.0, 1.0);
    /// assert!(kuma_bad.is_err());
    /// ```
    pub fn new(a: f64, b: f64) -> result::Result<Self> {
        let a_ok = a > 0.0 && a.is_finite();
        let b_ok = b > 0.0 && b.is_finite();

        if a_ok && b_ok {
            Ok(Kumaraswamy {
                a,
                b,
                ab_ln: a.ln() + b.ln(),
            })
        } else {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "a and b must be finite and greater than 0";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        }
    }

    /// Create a `Kumaraswamy` distribution with even density over (0, 1).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// let kuma = Kumaraswamy::uniform();
    /// assert_eq!(kuma, Kumaraswamy::new(1.0, 1.0).unwrap());
    /// ```
    pub fn uniform() -> Self {
        Kumaraswamy {
            a: 1.0,
            b: 1.0,
            ab_ln: 0.0,
        }
    }

    /// Get the `a` parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// let kuma = Kumaraswamy::new(1.0, 5.0).unwrap();
    /// assert_eq!(kuma.a(), 1.0);
    /// ```
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Get the `b` parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// let kuma = Kumaraswamy::new(1.0, 5.0).unwrap();
    /// assert_eq!(kuma.b(), 5.0);
    /// ```
    pub fn b(&self) -> f64 {
        self.b
    }
}

fn invcdf(p: f64, a: f64, b: f64) -> f64 {
    (1.0 - (1.0 - p).powf(b.recip())).powf(a.recip())
}

impl Rv<f64> for Kumaraswamy {
    fn ln_f(&self, x: &f64) -> f64 {
        let a = self.a;
        let b = self.b;
        self.ab_ln + (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x.powf(a)).ln()
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> f64 {
        let p: f64 = rng.gen();
        invcdf(p, self.a, self.b)
    }
}

impl Support<f64> for Kumaraswamy {
    fn supports(&self, x: &f64) -> bool {
        x.is_finite() && 0.0 < *x && *x < 1.0
    }
}

impl ContinuousDistr<f64> for Kumaraswamy {}

impl Cdf<f64> for Kumaraswamy {
    fn cdf(&self, x: &f64) -> f64 {
        1.0 - (1.0 - x.powf(self.a)).powf(self.b)
    }
}

impl InverseCdf<f64> for Kumaraswamy {
    fn invcdf(&self, p: f64) -> f64 {
        invcdf(p, self.a, self.b)
    }
}

impl Mean<f64> for Kumaraswamy {
    fn mean(&self) -> Option<f64> {
        let b = self.b;
        let ar1 = 1.0 + self.a.recip();
        Some(b * ar1.gamma() * b.gamma() / (ar1 + b).gamma())
    }
}

impl Median<f64> for Kumaraswamy {
    fn median(&self) -> Option<f64> {
        Some((1.0 - 2_f64.powf(-self.b.recip())).powf(self.a.recip()))
    }
}

impl Mode<f64> for Kumaraswamy {
    fn mode(&self) -> Option<f64> {
        if self.a < 1.0 || self.b < 1.0 {
            None
        } else if self.a == 1.0 && self.b == 1.0 {
            None
        } else {
            Some(
                ((self.a - 1.0) / (self.a * self.b - 1.0)).powf(self.a.recip()),
            )
        }
    }
}

impl Entropy for Kumaraswamy {
    fn entropy(&self) -> f64 {
        // Harmonic function for reals see:
        // https://en.wikipedia.org/wiki/Harmonic_number#Harmonic_numbers_for_real_and_complex_values
        let hb = self.b.digamma() + EULER_MASCERONI;
        (1.0 - self.b.recip()) + (1.0 - self.a.recip()) * hb - self.ab_ln
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_uniform_midpoint() {
        let kuma = Kumaraswamy::uniform();
    }
}
