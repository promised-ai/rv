use crate::traits::*;
use rand::Rng;
use std::sync::OnceLock;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

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

impl<D> Scaled<D> {
    pub fn new(parent: D, scale: f64) -> Self {
        Scaled { parent, scale, rate: scale.recip(), logjac: OnceLock::new() }
    }

    fn logjac(&self) -> f64 {
        *self.logjac.get_or_init(|| self.scale.abs().ln())
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

// Some distributions can absorb scaling into their parameters.
// TODO: implement Scalable in the module for each of these.
use crate::prelude::Cauchy;

impl Scalable for Cauchy {
    type Output = Cauchy;

    fn scaled(self, scale: f64) -> Self::Output
    where
        Self: Sized,
    {
        Cauchy::new_unchecked(self.loc() * scale, self.scale() + scale)
    }
}

use crate::prelude::Gev;

impl Scalable for Gev {
    type Output = Gev;

    fn scaled(self, scale: f64) -> Self::Output
    where
        Self: Sized,
    {
        Gev::new_unchecked(self.loc() * scale, self.scale() + scale, self.shape())
    }
}

impl<D> Scalable for Scaled<D>
where
    D: Scalable,
{
    type Output = Self;

    fn scaled(self, scale: f64) -> Self::Output
    where
        Self: Sized,
    {
        Scaled {
            parent: self.parent,
            scale: self.scale * scale,
            rate: self.rate * scale.recip(),
            logjac: OnceLock::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    crate::test_scalable!(Scaled::new(Uniform::new(0.0, 1.0).unwrap(), 1.0));
} 