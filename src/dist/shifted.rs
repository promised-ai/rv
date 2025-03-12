use crate::traits::*;
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use crate::data::ShiftedSuffStat;  

/// Trait for distributions that can be shifted by a constant value
pub trait Shiftable {
    type Output;
    fn shifted(self, shift: f64) -> Self::Output
    where
        Self: Sized;
}
/// A wrapper for distributions that adds a shift parameter
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Shifted<D> {
    parent: D,
    shift: f64,
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

/// Macro to implement Shiftable for a distribution type
/// 
/// This macro automatically implements the Shiftable trait for a given type,
/// using the default Shifted<T> as the Output type.
///
/// # Example
/// 
/// ```
/// impl_shiftable!(Gaussian);
/// impl_shiftable!(Beta<T>, T);
/// ```
#[macro_export]
macro_rules! impl_shiftable {
    // Simple case for non-generic types
    ($type:ty) => {
        impl Shiftable for $type {
            type Output = Shifted<Self>;
            
            fn shifted(self, shift: f64) -> Self::Output
            where
                Self: Sized,
            {
                Shifted {
                    parent: self,
                    shift,
                }
            }
        }
    };
}

// Some distributions can absorb shifting into the parameters.
// TODO: implement Shiftable in the module for each of these.
use crate::prelude::Cauchy;
use crate::prelude::Gaussian;
use crate::prelude::Gev;
use crate::prelude::Laplace;
use crate::prelude::Uniform;

// For others on ℝ, we'll fall back on Shifted
use crate::prelude::Beta;
impl_shiftable!(Beta);

use crate::prelude::BetaPrime;
impl_shiftable!(BetaPrime);

use crate::prelude::ChiSquared;
impl_shiftable!(ChiSquared);

use crate::prelude::Exponential;
impl_shiftable!(Exponential);

use crate::prelude::Gamma;
impl_shiftable!(Gamma);

use crate::prelude::Pareto;
impl_shiftable!(Pareto);

use crate::prelude::InvChiSquared;
impl_shiftable!(InvChiSquared);

use crate::prelude::InvGamma;
impl_shiftable!(InvGamma);

use crate::prelude::InvGaussian;
impl_shiftable!(InvGaussian);

use crate::prelude::LogNormal;
impl_shiftable!(LogNormal);

use crate::prelude::StudentsT;
impl_shiftable!(StudentsT);

use crate::prelude::ScaledInvChiSquared;
impl_shiftable!(ScaledInvChiSquared);

use crate::prelude::UnitPowerLaw;
impl_shiftable!(UnitPowerLaw);

use crate::prelude::Kumaraswamy;
impl_shiftable!(Kumaraswamy);
