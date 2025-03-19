use crate::traits::*;
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// A wrapper for distributions that adds a scale parameter
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Scaled<D> {
    parent: D,
    scale: f64,

    #[cfg_attr(feature = "serde1", serde(skip))]
    rate: f64,

    #[cfg_attr(feature = "serde1", serde(skip))]
    logjac: OnceLock<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScaledError {
    /// The scale parameter must be a normal (finite, non-zero, non-subnormal) number
    NonNormalScale(f64),
    /// The scale parameter must be positive
    NegativeScale(f64),
}

impl<D> Scaled<D> {
    /// Creates a new scaled distribution with the given parent distribution and
    /// scale factor.
    ///
    /// # Errors
    /// Returns `ScaledError::InvalidScale` if the scale parameter is not a
    /// normal number (i.e., if it's zero, infinite, or NaN).
    ///
    /// Returns `ScaledError::NegativeScale` if the scale parameter is not
    /// positive. We could easily allow scale to be negative, which would give
    /// us "reflected" distributions. But this breaks when we try to allow some
    /// distributions to absorb the scale parameter, so it's disallowed.
    pub fn new(parent: D, scale: f64) -> Result<Self, ScaledError> {
        if !scale.is_normal() {
            Err(ScaledError::NonNormalScale(scale))
        } else if scale <= 0.0 {
            Err(ScaledError::NegativeScale(scale))
        } else {
            Ok(Scaled {
                parent,
                scale,
                rate: scale.recip(),
                logjac: OnceLock::new(),
            })
        }
    }

    /// Creates a new scaled distribution with the given parent distribution and
    /// scale factor, without checking the scale parameter.
    ///
    /// # Safety
    /// The scale parameter must be a positive normal (finite, non-zero,
    /// non-subnormal) number.
    pub fn new_unchecked(parent: D, scale: f64) -> Self {
        Scaled {
            parent,
            scale,
            rate: scale.recip(),
            logjac: OnceLock::new(),
        }
    }

    fn logjac(&self) -> f64 {
        *self.logjac.get_or_init(|| self.scale.abs().ln())
    }
}

use crate::data::ScaledSuffStat;

impl<D> HasSuffStat<f64> for Scaled<D>
where
    D: HasSuffStat<f64>,
{
    type Stat = ScaledSuffStat<D::Stat>;

    fn empty_suffstat(&self) -> Self::Stat {
        ScaledSuffStat::new(self.parent.empty_suffstat(), self.scale)
    }

    fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
        self.parent.ln_f_stat(stat.parent())
    }
}

impl<D> Sampleable<f64> for Scaled<D>
where
    D: Sampleable<f64>,
{
    fn draw<R: Rng>(&self, rng: &mut R) -> f64 {
        self.parent.draw(rng) * self.scale
    }
}

impl<D> HasDensity<f64> for Scaled<D>
where
    D: HasDensity<f64>,
{
    fn ln_f(&self, x: &f64) -> f64 {
        self.parent.ln_f(&(x * self.rate)) - self.logjac()
    }
}

impl<D> Support<f64> for Scaled<D>
where
    D: Support<f64>,
{
    fn supports(&self, x: &f64) -> bool {
        self.parent.supports(&(x * self.rate))
    }
}

impl<D> ContinuousDistr<f64> for Scaled<D> where D: ContinuousDistr<f64> {}

impl<D> Cdf<f64> for Scaled<D>
where
    D: Cdf<f64>,
{
    fn cdf(&self, x: &f64) -> f64 {
        self.parent.cdf(&(x * self.rate))
    }

    fn sf(&self, x: &f64) -> f64 {
        self.parent.sf(&(x * self.rate))
    }
}

impl<D> InverseCdf<f64> for Scaled<D>
where
    D: InverseCdf<f64>,
{
    fn invcdf(&self, p: f64) -> f64 {
        self.parent.invcdf(p) * self.scale
    }

    fn interval(&self, p: f64) -> (f64, f64) {
        let (l, r) = self.parent.interval(p);
        (l * self.scale, r * self.scale)
    }
}

impl<D> Skewness for Scaled<D>
where
    D: Skewness,
{
    fn skewness(&self) -> Option<f64> {
        self.parent.skewness()
    }
}

impl<D> Kurtosis for Scaled<D>
where
    D: Kurtosis,
{
    fn kurtosis(&self) -> Option<f64> {
        self.parent.kurtosis()
    }
}

impl<D> Mean<f64> for Scaled<D>
where
    D: Mean<f64>,
{
    fn mean(&self) -> Option<f64> {
        self.parent.mean().map(|m| m * self.scale)
    }
}

impl<D> Median<f64> for Scaled<D>
where
    D: Median<f64>,
{
    fn median(&self) -> Option<f64> {
        self.parent.median().map(|m| m * self.scale)
    }
}

impl<D> Mode<f64> for Scaled<D>
where
    D: Mode<f64>,
{
    fn mode(&self) -> Option<f64> {
        self.parent.mode().map(|m| m * self.scale)
    }
}

impl<D> Variance<f64> for Scaled<D>
where
    D: Variance<f64>,
{
    fn variance(&self) -> Option<f64> {
        self.parent.variance().map(|v| v * self.scale * self.scale)
    }
}

impl<D> Entropy for Scaled<D>
where
    D: Entropy,
{
    fn entropy(&self) -> f64 {
        self.parent.entropy() + self.logjac()
    }
}

impl<D> Scalable for Scaled<D>
where
    D: Scalable,
{
    type Output = Self;
    type Error = ScaledError;

    fn scaled(self, scale: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Scaled::new(self.parent, self.scale * scale)
    }

    fn scaled_unchecked(self, scale: f64) -> Self::Output
    where
        Self: Sized,
    {
        Scaled {
            parent: self.parent,
            scale: self.scale * scale,
            rate: self.rate / scale,
            logjac: OnceLock::new(),
        }
    }
}

fn option_close(a: Option<f64>, b: Option<f64>, tol: f64) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() < tol,
        (None, None) => true,
        _ => false,
    }
}
