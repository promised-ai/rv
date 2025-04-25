use crate::data::ShiftedSuffStat;
use crate::traits::{Cdf, ContinuousDistr, Entropy, HasDensity, HasSuffStat, InverseCdf, Kurtosis, Mean, Median, Mode, Parameterized, Sampleable, Shiftable, Skewness, Support, Variance};
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Parameters for the Shifted distribution
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct ShiftedParameters<D: Parameterized> {
    /// The parent distribution's parameters
    pub parent: D::Parameters,
    /// The shift parameter
    pub shift: f64,
}

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
        if shift.is_finite() {
            Ok(Shifted { parent, shift })
        } else {
            Err(ShiftedError::NonFiniteShift(shift))
        }
    }

    pub fn new_unchecked(parent: D, shift: f64) -> Self {
        Shifted { parent, shift }
    }

    pub fn shift(&self) -> f64 {
        self.shift
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

impl<D> Parameterized for Shifted<D>
where
    D: Parameterized,
{
    type Parameters = ShiftedParameters<D>;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            parent: self.parent.emit_params(),
            shift: self.shift,
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(D::from_params(params.parent), params.shift)
    }
}

// TODO: We should be able to do something like this, and similarly for Shiftable
// Haven't hit any snags yet, just need to crank through it
// TODO: Remember to add tests for this
//
// d.shifted(shift).scaled(scale) -> d.scaled(scale).shifted(shift * scale)
// This prevents the possibility of chains of alternating shifts and scales
// impl<D> Scalable for Shifted<D>
// where
//     D: Scalable + Shiftable,
// {
//     type Output = Self;
//     type Error = Either<ShiftedError,  <D as Shiftable>::Error>;

//     fn scaled(self, scale: f64) -> Result<Self::Output, Self::Error>
//     where
//         Self: Sized,
//     {
//         match self.parent.scaled(scale) {
//             Ok(scaled_parent) => match Shifted::new(scaled_parent, self.shift * scale) {
//                 Ok(d) => Ok(d),
//                 Err(e) => Err(Either::Left(e)),
//             },
//             Err(e) => Err(Either::Right(e)),
//         }
//     }

//     fn scaled_unchecked(self, scale: f64) -> Self::Output
//     where
//         Self: Sized,
//     {
//         let scaled_parent = self.parent.scaled_unchecked(scale);
//         Shifted::new_unchecked(scaled_parent, self.shift * scale)
//     }
// }

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
