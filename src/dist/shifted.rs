use crate::data::ShiftedSuffStat;
use crate::prelude::Scaled;
use crate::traits::*;
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// A wrapper for distributions that adds a shift parameter
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Shifted<D> {
    parent: D,
    shift: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ShiftedError {
    /// The shift parameter must be a finite number
    NonFiniteShift(f64),
}

impl<D> Shifted<D> {
    pub fn new(parent: D, shift: f64) -> Result<Self, ShiftedError> {
        if !shift.is_finite() {
            Err(ShiftedError::NonFiniteShift(shift))
        } else {
            Ok(Shifted { parent, shift })
        }
    }

    pub fn new_unchecked(parent: D, shift: f64) -> Self {
        Shifted { parent, shift }
    }
}

impl<D> Sampleable<f64> for Shifted<D>
where
    D: Sampleable<f64>,
{
    fn draw<R: Rng>(&self, rng: &mut R) -> f64 {
        self.parent.draw(rng) + self.shift
    }
}

impl<D> HasDensity<f64> for Shifted<D>
where
    D: HasDensity<f64>,
{
    fn f(&self, x: &f64) -> f64 {
        self.parent.f(&(x - self.shift))
    }

    fn ln_f(&self, x: &f64) -> f64 {
        self.parent.ln_f(&(x - self.shift))
    }
}

impl<D> Support<f64> for Shifted<D>
where
    D: Support<f64>,
{
    fn supports(&self, x: &f64) -> bool {
        self.parent.supports(&(x - self.shift))
    }
}

impl<D> ContinuousDistr<f64> for Shifted<D> where D: ContinuousDistr<f64> {}

impl<D> Cdf<f64> for Shifted<D>
where
    D: Cdf<f64>,
{
    fn cdf(&self, x: &f64) -> f64 {
        self.parent.cdf(&(x - self.shift))
    }

    fn sf(&self, x: &f64) -> f64 {
        self.parent.sf(&(x - self.shift))
    }
}

impl<D> InverseCdf<f64> for Shifted<D>
where
    D: InverseCdf<f64>,
{
    fn invcdf(&self, p: f64) -> f64 {
        self.parent.invcdf(p) + self.shift
    }

    fn interval(&self, p: f64) -> (f64, f64) {
        let (l, r) = self.parent.interval(p);
        (l + self.shift, r + self.shift)
    }
}

impl<D> Skewness for Shifted<D>
where
    D: Skewness,
{
    fn skewness(&self) -> Option<f64> {
        self.parent.skewness()
    }
}

impl<D> Kurtosis for Shifted<D>
where
    D: Kurtosis,
{
    fn kurtosis(&self) -> Option<f64> {
        self.parent.kurtosis()
    }
}

impl<D> Mean<f64> for Shifted<D>
where
    D: Mean<f64>,
{
    fn mean(&self) -> Option<f64> {
        self.parent.mean().map(|m| m + self.shift)
    }
}

impl<D> Median<f64> for Shifted<D>
where
    D: Median<f64>,
{
    fn median(&self) -> Option<f64> {
        self.parent.median().map(|m| m + self.shift)
    }
}

impl<D> Mode<f64> for Shifted<D>
where
    D: Mode<f64>,
{
    fn mode(&self) -> Option<f64> {
        self.parent.mode().map(|m| m + self.shift)
    }
}

impl<D> Variance<f64> for Shifted<D>
where
    D: Variance<f64>,
{
    fn variance(&self) -> Option<f64> {
        self.parent.variance()
    }
}

impl<D> Entropy for Shifted<D>
where
    D: Entropy,
{
    fn entropy(&self) -> f64 {
        self.parent.entropy()
    }
}

impl<D> HasSuffStat<f64> for Shifted<D>
where
    D: HasSuffStat<f64>,
{
    type Stat = ShiftedSuffStat<D::Stat>;

    fn empty_suffstat(&self) -> Self::Stat {
        ShiftedSuffStat::new(self.parent.empty_suffstat(), self.shift)
    }

    fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
        self.parent.ln_f_stat(stat.parent())
    }
}

impl<D> Shiftable for Shifted<D>
where
    D: Shiftable,
{
    type Output = Self;
    type Error = ShiftedError;

    fn shifted(self, shift: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Shifted::new(self.parent, self.shift + shift)
    }

    fn shifted_unchecked(self, shift: f64) -> Self::Output
    where
        Self: Sized,
    {
        Shifted::new_unchecked(self.parent, self.shift + shift)
    }
}

#[cfg(test)]
mod tests {

    use crate::prelude::*;
    use crate::test_shiftable_cdf;
    use crate::test_shiftable_density;
    use crate::test_shiftable_entropy;
    use crate::test_shiftable_invcdf;
    use crate::test_shiftable_method;

    test_shiftable_method!(
        Shifted::new(Gaussian::new(2.0, 4.0).unwrap(), 1.0).unwrap(),
        mean
    );
    test_shiftable_method!(
        Shifted::new(Gaussian::new(2.0, 4.0).unwrap(), 1.0).unwrap(),
        median
    );
    test_shiftable_method!(
        Shifted::new(Gaussian::new(2.0, 4.0).unwrap(), 1.0).unwrap(),
        variance
    );
    test_shiftable_method!(
        Shifted::new(Gaussian::new(2.0, 4.0).unwrap(), 1.0).unwrap(),
        skewness
    );
    test_shiftable_method!(
        Shifted::new(Gaussian::new(2.0, 4.0).unwrap(), 1.0).unwrap(),
        kurtosis
    );
    test_shiftable_density!(Shifted::new(
        Gaussian::new(2.0, 4.0).unwrap(),
        1.0
    )
    .unwrap());
    test_shiftable_entropy!(Shifted::new(
        Gaussian::new(2.0, 4.0).unwrap(),
        1.0
    )
    .unwrap());
    test_shiftable_cdf!(
        Shifted::new(Gaussian::new(2.0, 4.0).unwrap(), 1.0).unwrap()
    );
    test_shiftable_invcdf!(
        Shifted::new(Gaussian::new(2.0, 4.0).unwrap(), 1.0).unwrap()
    );
}
