//! Data utilities
mod partition;
mod suffstat;

pub use partition::Partition;
pub use suffstat::BernoulliSuffStat;
pub use suffstat::CategoricalSuffStat;
pub use suffstat::GaussianSuffStat;
pub use suffstat::MvGaussianSuffStat;
pub use suffstat::PoissonSuffStat;

use crate::traits::{HasSuffStat, SuffStat};

/// The trait that data must implemented by all data used with the
/// `Categorical` distribution
pub trait CategoricalDatum: Sized + Sync + Copy {
    fn into_usize(&self) -> usize;
    fn from_usize(n: usize) -> Self;
}

impl CategoricalDatum for usize {
    fn into_usize(&self) -> usize {
        *self
    }

    fn from_usize(n: usize) -> Self {
        n
    }
}

impl CategoricalDatum for bool {
    fn into_usize(&self) -> usize {
        if *self {
            1
        } else {
            0
        }
    }

    fn from_usize(n: usize) -> Self {
        match n {
            0 => false,
            1 => true,
            _ => panic!("cannot convert {} into bool", n),
        }
    }
}

macro_rules! impl_categorical_datum {
    ($kind:ty) => {
        impl CategoricalDatum for $kind {
            fn into_usize(&self) -> usize {
                *self as usize
            }

            fn from_usize(n: usize) -> Self {
                n as $kind
            }
        }
    };
}

impl_categorical_datum!(u8);
impl_categorical_datum!(u16);
impl_categorical_datum!(u32);

/// Converts to and from a bool
pub trait Booleable: Sized + Sync + Copy {
    /// Convert from bool. Should never panic.
    fn from_bool(b: bool) -> Self;

    /// Convert into bool Will panic if cannot be converted.
    fn into_bool(self) -> bool {
        self.try_into_bool().expect("could not convert into bool")
    }

    /// Try to convert into bool. Returns None if cannot be converted.
    fn try_into_bool(self) -> Option<bool>;
}

macro_rules! impl_booleable {
    ($kind:ty) => {
        impl Booleable for $kind {
            fn try_into_bool(self) -> Option<bool> {
                if self == 1 {
                    Some(true)
                } else if self == 0 {
                    Some(false)
                } else {
                    None
                }
            }

            fn from_bool(b: bool) -> Self {
                if b {
                    1
                } else {
                    0
                }
            }
        }
    };
}

impl_booleable!(u8);
impl_booleable!(u16);
impl_booleable!(u32);
impl_booleable!(u64);
impl_booleable!(usize);

impl_booleable!(i8);
impl_booleable!(i16);
impl_booleable!(i32);
impl_booleable!(i64);
impl_booleable!(isize);

/// Holds either a sufficient statistic of a vector of data.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
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

    /// Determine whether the object contains data
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::DataOrSuffStat;
    /// use rv::dist::Gaussian;
    /// use rv::data::GaussianSuffStat;
    ///
    /// let xs = vec![1.0_f64];
    /// let data: DataOrSuffStat<f64, Gaussian> = DataOrSuffStat::Data(&xs);
    ///
    /// assert!(data.is_data());
    ///
    /// let gauss_stats = GaussianSuffStat::new();
    /// let suffstat: DataOrSuffStat<f64, Gaussian> = DataOrSuffStat::SuffStat(&gauss_stats);
    ///
    /// assert!(!suffstat.is_data());
    /// ```
    pub fn is_data(&self) -> bool {
        match &self {
            DataOrSuffStat::Data(..) => true,
            _ => false,
        }
    }

    /// Determine whether the object contains sufficient statistics
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::DataOrSuffStat;
    /// use rv::dist::Gaussian;
    /// use rv::data::GaussianSuffStat;
    ///
    /// let xs = vec![1.0_f64];
    /// let data: DataOrSuffStat<f64, Gaussian> = DataOrSuffStat::Data(&xs);
    ///
    /// assert!(!data.is_suffstat());
    ///
    /// let gauss_stats = GaussianSuffStat::new();
    /// let suffstat: DataOrSuffStat<f64, Gaussian> = DataOrSuffStat::SuffStat(&gauss_stats);
    ///
    /// assert!(suffstat.is_suffstat());
    /// ```
    pub fn is_suffstat(&self) -> bool {
        match &self {
            DataOrSuffStat::SuffStat(..) => true,
            _ => false,
        }
    }

    /// Determine whether the object is empty
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::DataOrSuffStat;
    /// use rv::dist::Gaussian;
    /// use rv::data::GaussianSuffStat;
    ///
    /// let xs = vec![1.0_f64];
    /// let data: DataOrSuffStat<f64, Gaussian> = DataOrSuffStat::Data(&xs);
    ///
    /// assert!(!data.is_none());
    ///
    /// let gauss_stats = GaussianSuffStat::new();
    /// let suffstat: DataOrSuffStat<f64, Gaussian> = DataOrSuffStat::SuffStat(&gauss_stats);
    ///
    /// assert!(!suffstat.is_none());
    ///
    /// let none: DataOrSuffStat<f64, Gaussian> = DataOrSuffStat::None;
    ///
    /// assert!(none.is_none());
    /// ```
    pub fn is_none(&self) -> bool {
        match &self {
            DataOrSuffStat::None => true,
            _ => false,
        }
    }
}
