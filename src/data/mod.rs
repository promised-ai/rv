//! Data utilities
mod partition;
mod suffstat;

pub use self::partition::Partition;
pub use self::suffstat::BernoulliSuffStat;
pub use self::suffstat::CategoricalSuffStat;
pub use self::suffstat::GaussianSuffStat;

extern crate num;
use self::num::traits::FromPrimitive;
use traits::HasSuffStat;

/// The trait that data must implemented by all data used with the
/// `Categorical` distribution
pub trait CategoricalDatum:
    Sized + Into<usize> + Sync + Copy + FromPrimitive
{
}

impl<T> CategoricalDatum for T where
    T: Clone + Into<usize> + Sync + Copy + FromPrimitive
{}

/// Holds either a sufficient statistic of a vector of data.
pub enum DataOrSuffStat<'a, X, Fx>
where
    X: 'a,
    Fx: 'a + HasSuffStat<X>,
{
    /// A `Vec` of raw data
    Data(&'a Vec<X>),
    /// A sufficient statistic
    SuffStat(&'a Fx::Stat),
}
