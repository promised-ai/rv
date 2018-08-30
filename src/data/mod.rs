//! Data utilities
mod partition;
mod suffstat;

pub use self::partition::Partition;
pub use self::suffstat::BernoulliSuffStat;
pub use self::suffstat::CategoricalSuffStat;
pub use self::suffstat::GaussianSuffStat;
pub use self::suffstat::MvGaussianSuffStat;

extern crate num;
use self::num::traits::FromPrimitive;
use traits::{HasSuffStat, SuffStat};

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
#[derive(Debug, Clone)]
pub enum DataOrSuffStat<'a, X, Fx>
where
    X: 'a,
    Fx: 'a + HasSuffStat<X>,
{
    /// A `Vec` of raw data
    Data(&'a Vec<X>),
    /// A sufficient statistic
    SuffStat(&'a Fx::Stat),
    /// No data
    None,
}

impl<'a, X, Fx> DataOrSuffStat<'a, X, Fx>
where
    X: 'a,
    Fx: 'a + HasSuffStat<X>,
{
    /// Get the number of observations
    pub fn n(&self) -> usize {
        match &self {
            DataOrSuffStat::Data(data) => data.len(),
            DataOrSuffStat::SuffStat(s) => s.n(),
            DataOrSuffStat::None => 0,
        }
    }
}
