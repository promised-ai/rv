mod partition;
mod suffstat;

pub use self::partition::Partition;
pub use self::suffstat::BernoulliSuffStat;
pub use self::suffstat::CategoricalSuffStat;
pub use self::suffstat::GaussianSuffStat;

extern crate num;
use self::num::traits::FromPrimitive;

pub trait CategoricalDatum:
    Sized + Into<usize> + Sync + Copy + FromPrimitive
{
}

impl<T> CategoricalDatum for T where
    T: Clone + Into<usize> + Sync + Copy + FromPrimitive
{}
