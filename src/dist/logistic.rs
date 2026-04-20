#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::traits::{
    Cdf, ContinuousDistr, Entropy, HasDensity, InverseCdf, Kurtosis, Mean,
    Median, Mode, Parameterized, QuadBounds, Sampleable, Scalable, Shiftable,
    Skewness, Support, Variance,
};
use rand::Rng;
use std::f64;

/// The logistic distribution
///
/// Based on the parameterization https://en.wikipedia.org/wiki/Logistic_distribution
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Logistic {
    /// Mean or location of the distribution.
    mu: f64,
    /// Scale, must be positive.
    s: f64,
    /// Precompute s.ln()
    ln_s: f64,
}

impl Default for Logistic {
    fn default() -> Self {
        Self {
            mu: 0.0,
            s: 1.0,
            ln_s: 0.0,
        }
    }
}

impl Logistic {
    /// Create a new [`Logistic`] distribution with `mu` location and `s` scale.
    ///
    /// # Errors
    /// If s <= 0, then this will return an error.
    pub fn new(mu: f64, s: f64) -> Result<Self, LogisticError> {
        if s <= 0.0 {
            Err(LogisticError::NonPositiveScale)
        } else {
            Ok(Self {
                mu,
                s,
                ln_s: s.ln(),
            })
        }
    }

    /// Create a new [`Logistic`] distribution with `mu` location and `s` scale, without checking
    /// for proper parameters.
    pub fn new_unchecked(mu: f64, s: f64) -> Self {
        Self {
            mu,
            s,
            ln_s: s.ln(),
        }
    }

    /// Get the mu (location) parameter from the distribution
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Set the mu (location) value for this distribution, cannot fail.
    pub fn set_mu(&mut self, mu: f64) {
        self.mu = mu;
    }

    /// Get the scale parameter from the distribution
    pub fn s(&self) -> f64 {
        self.s
    }

    /// Set the scale parameter from the distribution
    ///
    /// # Errors
    /// If s <= 0, this will return an error.
    pub fn set_s(&mut self, s: f64) -> Result<(), LogisticError> {
        if s <= 0.0 {
            Err(LogisticError::NonPositiveScale)
        } else {
            self.s = s;
            self.ln_s = s.ln();
            Ok(())
        }
    }
}

impl Scalable for Logistic {
    type Output = Logistic;
    type Error = LogisticError;

    fn scaled(self, scale: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Self::new(self.mu * scale, self.s * scale)
    }

    fn scaled_unchecked(self, scale: f64) -> Self::Output
    where
        Self: Sized,
    {
        Self::new_unchecked(self.mu * scale, self.s * scale)
    }
}

impl Shiftable for Logistic {
    type Output = Logistic;
    type Error = LogisticError;

    fn shifted(self, shift: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Ok(Self::new_unchecked(self.mu + shift, self.s))
    }

    fn shifted_unchecked(self, shift: f64) -> Self::Output
    where
        Self: Sized,
    {
        Self::new_unchecked(self.mu + shift, self.s)
    }
}

impl Parameterized for Logistic {
    type Parameters = LogisticParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            mu: self.mu,
            s: self.s,
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self {
            mu: params.mu,
            s: params.s,
            ln_s: params.s.ln(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct LogisticParameters {
    mu: f64,
    s: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum LogisticError {
    /// The scale provided was not positive
    NonPositiveScale,
}

impl std::error::Error for LogisticError {}

#[cfg_attr(coverage_nightly, coverage(off))]
impl std::fmt::Display for LogisticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonPositiveScale => {
                write!(f, "non-positive scale")
            }
        }
    }
}

impl From<&Logistic> for String {
    fn from(value: &Logistic) -> Self {
        format!("Logistic(μ: {}, s: {})", value.mu, value.s)
    }
}

impl_display!(Logistic);

macro_rules! impl_traits {
    ($kind: ty) => {
        impl HasDensity<$kind> for Logistic {
            fn ln_f(&self, x: &$kind) -> f64 {
                let x = *x as f64;
                let y = -(x - self.mu) / self.s;
                2.0_f64.mul_add(-softplus(y), -y)
            }
        }

        impl Sampleable<$kind> for Logistic {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let x: f64 = rng.random();
                self.s.mul_add(x.ln() - (-x).ln_1p(), self.mu) as $kind
            }
        }

        impl ContinuousDistr<$kind> for Logistic {}

        impl Support<$kind> for Logistic {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite()
            }
        }

        impl Cdf<$kind> for Logistic {
            fn cdf(&self, x: &$kind) -> f64 {
                let x = *x as f64;
                let y = -(x - self.mu) / self.s;
                1.0 / (1.0 + y.exp())
            }
        }

        impl InverseCdf<$kind> for Logistic {
            fn invcdf(&self, p: f64) -> $kind {
                self.s.mul_add(p.ln() - (-p).ln_1p(), self.mu) as $kind
            }
        }

        impl Mean<$kind> for Logistic {
            fn mean(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl Median<$kind> for Logistic {
            fn median(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl Mode<$kind> for Logistic {
            fn mode(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl Variance<$kind> for Logistic {
            fn variance(&self) -> Option<$kind> {
                Some(
                    (self.s.powi(2) * std::f64::consts::PI.powi(2) / 3.0)
                        as $kind,
                )
            }
        }
    };
}

impl_traits!(f64);
impl_traits!(f32);

impl Entropy for Logistic {
    fn entropy(&self) -> f64 {
        self.ln_s + 2.0
    }
}

impl Skewness for Logistic {
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl Kurtosis for Logistic {
    fn kurtosis(&self) -> Option<f64> {
        Some(6.0 / 5.0)
    }
}

impl QuadBounds for Logistic {
    fn quad_bounds(&self) -> (f64, f64) {
        self.interval(0.999_999_999_999)
    }
}

/// An efficient implementation of the log(1+e^x) function
#[inline(always)]
fn softplus(x: f64) -> f64 {
    if x <= -37.0 {
        x.exp()
    } else if x <= 18.0 {
        x.exp().ln_1p()
    } else if x <= 33.0 {
        x + (-x).exp().ln_1p()
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    use crate::{misc::ks_test, traits::Cdf, traits::Sampleable};

    #[test]
    fn sampling_matches_cdf() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1234);
        let dist = Logistic::new_unchecked(2.0, 4.0);
        let sample: Vec<f64> = dist.sample(1000, &mut rng);

        let (_stat, p) = ks_test(&sample, |x| dist.cdf(&x));
        assert!(p > 0.05);
    }
}
