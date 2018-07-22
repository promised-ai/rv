mod partition;
mod suffstat;

pub use self::partition::Partition;
pub use self::suffstat::BernoulliSuffStat;
pub use self::suffstat::CategoricalSuffStat;
pub use self::suffstat::GaussianSuffStat;

extern crate num;
use self::num::traits::FromPrimitive;
use traits::HasSuffStat;

/// The trait that data must implement to work in a `Categorical` distribution
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
    Data(&'a Vec<X>),
    SuffStat(&'a Fx::Stat),
}
