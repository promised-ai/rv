use crate::traits::Parameterized;
use crate::traits::*;
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::OnceLock;

/// A wrapper for distributions that adds a scale parameter
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Scaled<D> {
    parent: D,
    scale: f64,
    rate: f64,
    logjac: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScaledError {
    /// The scale parameter must be a normal (finite, non-zero, non-subnormal) number
    NonNormalScale(f64),
    /// The scale parameter must be positive
    NegativeScale(f64),
}

impl std::error::Error for ScaledError {}

impl fmt::Display for ScaledError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonNormalScale(scale) => {
                write!(f, "non-normal scale: {}", scale)
            }
            Self::NegativeScale(scale) => {
                write!(f, "negative scale: {}", scale)
            }
        }
    }
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
                logjac: scale.abs().ln(),
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
            logjac: scale.abs().ln(),
        }
    }

    pub fn from_parts_unchecked(
        parent: D,
        scale: f64,
        rate: f64,
        logjac: f64,
    ) -> Self {
        Scaled {
            parent,
            scale,
            rate,
            logjac,
        }
    }
    pub fn parent(&self) -> &D {
        &self.parent
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    pub fn logjac(&self) -> f64 {
        self.logjac
    }

    pub fn map_parent_params(
        &self,
        f: impl Fn(D::Parameters) -> D::Parameters,
    ) -> Self
    where
        D: Parameterized,
    {
        let parent = self.parent.map_params(f);
        Self::from_parts_unchecked(parent, self.scale, self.rate, self.logjac)
    }
}

pub struct ScaledParameters<D: Parameterized> {
    parent: D::Parameters,
    scale: f64,
}

impl<D> Parameterized for Scaled<D>
where
    D: Parameterized,
{
    type Parameters = ScaledParameters<D>;

    fn emit_params(&self) -> Self::Parameters {
        ScaledParameters {
            parent: self.parent.emit_params(),
            scale: self.scale,
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        let parent = D::from_params(params.parent);
        Self::new_unchecked(parent, params.scale)
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
        let new_scale = self.scale * scale;
        Scaled {
            parent: self.parent,
            scale: new_scale,
            rate: new_scale.recip(),
            logjac: new_scale.ln(),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::prelude::*;
    use crate::test_scalable_cdf;
    use crate::test_scalable_density;
    use crate::test_scalable_entropy;
    use crate::test_scalable_invcdf;
    use crate::test_scalable_method;

    test_scalable_method!(
        Scaled::new(Gaussian::new(2.0, 4.0).unwrap(), 3.0).unwrap(),
        mean
    );
    test_scalable_method!(
        Scaled::new(Gaussian::new(2.0, 4.0).unwrap(), 3.0).unwrap(),
        median
    );
    test_scalable_method!(
        Scaled::new(Gaussian::new(2.0, 4.0).unwrap(), 3.0).unwrap(),
        variance
    );
    test_scalable_method!(
        Scaled::new(Gaussian::new(2.0, 4.0).unwrap(), 3.0).unwrap(),
        skewness
    );
    test_scalable_method!(
        Scaled::new(Gaussian::new(2.0, 4.0).unwrap(), 3.0).unwrap(),
        kurtosis
    );
    test_scalable_density!(
        Scaled::new(Gaussian::new(2.0, 4.0).unwrap(), 3.0).unwrap()
    );
    test_scalable_entropy!(
        Scaled::new(Gaussian::new(2.0, 4.0).unwrap(), 3.0).unwrap()
    );
    test_scalable_cdf!(
        Scaled::new(Gaussian::new(2.0, 4.0).unwrap(), 3.0).unwrap()
    );
    test_scalable_invcdf!(
        Scaled::new(Gaussian::new(2.0, 4.0).unwrap(), 3.0).unwrap()
    );
}
