//! Data utilities
mod partition;
mod stat;

pub use partition::Partition;
pub use stat::BernoulliSuffStat;
pub use stat::BetaSuffStat;
pub use stat::CategoricalSuffStat;
pub use stat::CdvmSuffStat;
pub use stat::GaussianSuffStat;
pub use stat::InvGammaSuffStat;
pub use stat::InvGaussianSuffStat;
#[cfg(feature = "arraydist")]
pub use stat::MvGaussianSuffStat;
pub use stat::PoissonSuffStat;
pub use stat::ScaledSuffStat;
pub use stat::ShiftedSuffStat;
pub use stat::UnitPowerLawSuffStat;

use crate::dist::{
    Bernoulli, Categorical, Gaussian, InvGamma, InvGaussian, Poisson,
};
use crate::traits::ConjugatePrior;
use crate::traits::HasDensity;
use crate::traits::{HasSuffStat, SuffStat};

pub type BernoulliData<'a, X> = DataOrSuffStat<'a, X, Bernoulli>;
pub type CategoricalData<'a, X> = DataOrSuffStat<'a, X, Categorical>;
pub type GaussianData<'a, X> = DataOrSuffStat<'a, X, Gaussian>;
pub type InvGaussianData<'a, X> = DataOrSuffStat<'a, X, InvGaussian>;
pub type InvGammaData<'a, X> = DataOrSuffStat<'a, X, InvGamma>;
pub type PoissonData<'a, X> = DataOrSuffStat<'a, X, Poisson>;

/// The trait that data must implemented by all data used with the
/// `Categorical` distribution
pub trait CategoricalDatum: Sized + Sync + Copy {
    fn into_usize(self) -> usize;
    fn from_usize(n: usize) -> Self;
}

#[allow(clippy::wrong_self_convention)]
impl CategoricalDatum for usize {
    fn into_usize(self) -> usize {
        self
    }

    fn from_usize(n: usize) -> Self {
        n
    }
}

impl CategoricalDatum for bool {
    fn into_usize(self) -> usize {
        usize::from(self)
    }

    fn from_usize(n: usize) -> Self {
        n > 0
    }
}

macro_rules! impl_categorical_datum {
    ($kind:ty) => {
        impl CategoricalDatum for $kind {
            fn into_usize(self) -> usize {
                self as usize
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

#[allow(clippy::wrong_self_convention)]
impl Booleable for bool {
    fn into_bool(self) -> bool {
        self
    }

    fn try_into_bool(self) -> Option<bool> {
        Some(self)
    }

    fn from_bool(b: bool) -> Self {
        b
    }
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
#[derive(Debug, Clone, PartialEq)]
pub enum DataOrSuffStat<'a, X, Fx>
where
    X: 'a,
    Fx: 'a + HasSuffStat<X>,
{
    /// A `Vec` of raw data
    Data(&'a [X]),
    /// A sufficient statistic
    SuffStat(&'a Fx::Stat),
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
            DataOrSuffStat::SuffStat(s) => <Fx::Stat as SuffStat<X>>::n(s),
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
        matches!(&self, DataOrSuffStat::Data(..))
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
        matches!(&self, DataOrSuffStat::SuffStat(..))
    }
}

/// Convert a `DataOrSuffStat` into a `Stat`
#[inline]
pub fn extract_stat<X, Fx, Pr>(pr: &Pr, x: &DataOrSuffStat<X, Fx>) -> Fx::Stat
where
    Fx: HasSuffStat<X> + HasDensity<X>,
    Fx::Stat: Clone,
    Pr: ConjugatePrior<X, Fx>,
{
    match x {
        DataOrSuffStat::SuffStat(s) => (*s).clone(),
        DataOrSuffStat::Data(xs) => {
            let mut stat = pr.empty_stat();
            stat.observe_many(xs);
            stat
        }
    }
}

/// Convert a `DataOrSuffStat` into a `Stat` then do something with it
pub fn extract_stat_then<X, Fx, Pr, Fnx, Y>(
    pr: &Pr,
    x: &DataOrSuffStat<X, Fx>,
    f_stat: Fnx,
) -> Y
where
    Fx: HasSuffStat<X> + HasDensity<X>,
    Fx::Stat: Clone,
    Pr: ConjugatePrior<X, Fx>,
    Fnx: Fn(Fx::Stat) -> Y,
{
    let stat = extract_stat(pr, x);
    f_stat(stat)
}

#[cfg(test)]
mod tests {
    use super::*;

    mod categorical_datum {
        use super::*;

        #[test]
        fn impl_usize_into_usize() {
            let x = 8_usize;
            assert_eq!(x.into_usize(), 8);
        }

        #[test]
        fn impl_usize_from_usize() {
            let x: usize = CategoricalDatum::from_usize(8_usize);
            assert_eq!(x, 8_usize);
        }

        #[test]
        fn impl_bool_into_usize() {
            assert_eq!(false.into_usize(), 0);
            assert_eq!(true.into_usize(), 1);
        }

        #[test]
        fn impl_bool_from_usize() {
            let x: bool = CategoricalDatum::from_usize(0_usize);
            assert!(!x);

            let y: bool = CategoricalDatum::from_usize(1_usize);
            assert!(y);

            let z: bool = CategoricalDatum::from_usize(122_usize);
            assert!(z);
        }

        macro_rules! catdatum_test {
            ($type: ty, $from_test: ident, $into_test: ident) => {
                #[test]
                fn $from_test() {
                    let x: usize = 123;
                    let y: $type = CategoricalDatum::from_usize(x);
                    assert_eq!(y, 123);
                }

                #[test]
                fn $into_test() {
                    let x: $type = 123;
                    assert_eq!(x.into_usize(), 123_usize);
                }
            };
        }

        catdatum_test!(u8, impl_u8_into_usize, impl_u8_from_usize);
        catdatum_test!(u16, impl_u16_into_usize, impl_u16_from_usize);
        catdatum_test!(u32, impl_u32_into_usize, impl_u32_from_usize);
    }

    mod booleable {
        use super::*;

        macro_rules! booleable_test {
            ($type: ty, $from_test: ident, $into_test: ident) => {
                #[test]
                fn $from_test() {
                    let xt: $type = Booleable::from_bool(true);
                    let xf: $type = Booleable::from_bool(false);

                    assert_eq!(xt, 1);
                    assert_eq!(xf, 0);
                }

                #[test]
                fn $into_test() {
                    let xf: $type = 0;
                    let xt: $type = 1;
                    let xe: $type = 123;

                    assert_eq!(xf.try_into_bool(), Some(false));
                    assert_eq!(xt.try_into_bool(), Some(true));
                    assert_eq!(xe.try_into_bool(), None);
                }
            };
        }

        booleable_test!(u8, impl_u8_from_bool, impl_u8_try_into_bool);
        booleable_test!(u16, impl_u16_from_bool, impl_u16_try_into_bool);
        booleable_test!(u32, impl_u32_from_bool, impl_u32_try_into_bool);
        booleable_test!(u64, impl_u64_from_bool, impl_u64_try_into_bool);
        booleable_test!(usize, impl_usize_from_bool, impl_usize_try_into_bool);

        booleable_test!(i8, impl_i8_from_bool, impl_i8_try_into_bool);
        booleable_test!(i16, impl_i16_from_bool, impl_i16_try_into_bool);
        booleable_test!(i32, impl_i32_from_bool, impl_i32_try_into_bool);
        booleable_test!(i64, impl_i64_from_bool, impl_i64_try_into_bool);
        booleable_test!(isize, impl_isize_from_bool, impl_isize_try_into_bool);
    }
}
