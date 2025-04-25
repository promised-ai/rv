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
pub use stat::VonMisesSuffStat;

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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn is_suffstat(&self) -> bool {
        matches!(&self, DataOrSuffStat::SuffStat(..))
    }
}

#[inline]
pub fn extract_stat<X, Fx, Pr>(pr: &Pr, x: &DataOrSuffStat<X, Fx>) -> Fx::Stat
where
    Fx: HasSuffStat<X> + HasDensity<X>,
    Fx::Stat: Clone,
    Pr: ConjugatePrior<X, Fx>,
{
    extract_stat_then(pr, x, std::clone::Clone::clone)
}

/// Convert a `DataOrSuffStat` into a `Stat` then do something with it
#[inline]
pub fn extract_stat_then<X, Fx, Pr, Fnx, Y>(
    pr: &Pr,
    x: &DataOrSuffStat<X, Fx>,
    f_stat: Fnx,
) -> Y
where
    Fx: HasSuffStat<X> + HasDensity<X>,
    Pr: ConjugatePrior<X, Fx> + ?Sized,
    Fnx: Fn(&Fx::Stat) -> Y,
{
    match x {
        DataOrSuffStat::SuffStat(s) => f_stat(s),
        DataOrSuffStat::Data(xs) => {
            let mut stat = pr.empty_stat();
            stat.observe_many(xs);
            f_stat(&stat)
        }
    }
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

        #[test]
        fn categorical_datum_edge_cases() {
            // Test with zero
            let zero_u8: u8 = CategoricalDatum::from_usize(0);
            assert_eq!(zero_u8, 0);
            assert_eq!(zero_u8.into_usize(), 0);

            let zero_u16: u16 = CategoricalDatum::from_usize(0);
            assert_eq!(zero_u16, 0);
            assert_eq!(zero_u16.into_usize(), 0);

            let zero_u32: u32 = CategoricalDatum::from_usize(0);
            assert_eq!(zero_u32, 0);
            assert_eq!(zero_u32.into_usize(), 0);

            // Test with values close to type limits
            // Maximum values for different types
            let max_u8_value = u8::MAX as usize;
            let u8_val: u8 = CategoricalDatum::from_usize(max_u8_value);
            assert_eq!(u8_val, u8::MAX);
            assert_eq!(u8_val.into_usize(), max_u8_value);

            let max_u16_value = u16::MAX as usize;
            let u16_val: u16 = CategoricalDatum::from_usize(max_u16_value);
            assert_eq!(u16_val, u16::MAX);
            assert_eq!(u16_val.into_usize(), max_u16_value);

            // Test overflow behavior - type limits are respected
            // When converting a value larger than the type can hold, it will truncate
            let overflow_u8: u8 = CategoricalDatum::from_usize(256); // 256 = 2^8, so this will overflow u8
            assert_eq!(overflow_u8, 0); // Should truncate to 0

            let overflow_u16: u16 = CategoricalDatum::from_usize(65536); // 65536 = 2^16, so this will overflow u16
            assert_eq!(overflow_u16, 0); // Should truncate to 0
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

        #[test]
        fn impl_bool_into_bool() {
            let t = true;
            let f = false;
            assert!(t.into_bool());
            assert!(!f.into_bool());
        }

        #[test]
        fn verify_into_bool_error_handling() {
            // For each numeric type, test that invalid values (not 0 or 1) cause panic
            // when using into_bool but return None with try_into_bool

            // u8
            let invalid_u8: u8 = 2;
            assert_eq!(invalid_u8.try_into_bool(), None);

            // u16
            let invalid_u16: u16 = 2;
            assert_eq!(invalid_u16.try_into_bool(), None);

            // u32
            let invalid_u32: u32 = 2;
            assert_eq!(invalid_u32.try_into_bool(), None);

            // u64
            let invalid_u64: u64 = 2;
            assert_eq!(invalid_u64.try_into_bool(), None);

            // usize
            let invalid_usize: usize = 2;
            assert_eq!(invalid_usize.try_into_bool(), None);

            // i8
            let invalid_i8: i8 = 2;
            assert_eq!(invalid_i8.try_into_bool(), None);

            // i16
            let invalid_i16: i16 = 2;
            assert_eq!(invalid_i16.try_into_bool(), None);

            // i32
            let invalid_i32: i32 = 2;
            assert_eq!(invalid_i32.try_into_bool(), None);

            // i64
            let invalid_i64: i64 = 2;
            assert_eq!(invalid_i64.try_into_bool(), None);

            // isize
            let invalid_isize: isize = 2;
            assert_eq!(invalid_isize.try_into_bool(), None);
        }

        #[test]
        #[should_panic(expected = "could not convert into bool")]
        fn verify_into_bool_panics_for_invalid_value() {
            let invalid: u8 = 2;
            let _ = invalid.into_bool(); // This should panic
        }

        #[test]
        fn verify_from_bool_consistency() {
            // Test that from_bool followed by try_into_bool is an identity function
            // for all types

            // Test true
            let true_u8 = u8::from_bool(true);
            assert_eq!(true_u8.try_into_bool(), Some(true));

            let true_u16 = u16::from_bool(true);
            assert_eq!(true_u16.try_into_bool(), Some(true));

            let true_u32 = u32::from_bool(true);
            assert_eq!(true_u32.try_into_bool(), Some(true));

            let true_u64 = u64::from_bool(true);
            assert_eq!(true_u64.try_into_bool(), Some(true));

            let true_usize = usize::from_bool(true);
            assert_eq!(true_usize.try_into_bool(), Some(true));

            let true_i8 = i8::from_bool(true);
            assert_eq!(true_i8.try_into_bool(), Some(true));

            let true_i16 = i16::from_bool(true);
            assert_eq!(true_i16.try_into_bool(), Some(true));

            let true_i32 = i32::from_bool(true);
            assert_eq!(true_i32.try_into_bool(), Some(true));

            let true_i64 = i64::from_bool(true);
            assert_eq!(true_i64.try_into_bool(), Some(true));

            let true_isize = isize::from_bool(true);
            assert_eq!(true_isize.try_into_bool(), Some(true));

            // Test false
            let false_u8 = u8::from_bool(false);
            assert_eq!(false_u8.try_into_bool(), Some(false));

            let false_u16 = u16::from_bool(false);
            assert_eq!(false_u16.try_into_bool(), Some(false));

            let false_u32 = u32::from_bool(false);
            assert_eq!(false_u32.try_into_bool(), Some(false));

            let false_u64 = u64::from_bool(false);
            assert_eq!(false_u64.try_into_bool(), Some(false));

            let false_usize = usize::from_bool(false);
            assert_eq!(false_usize.try_into_bool(), Some(false));

            let false_i8 = i8::from_bool(false);
            assert_eq!(false_i8.try_into_bool(), Some(false));

            let false_i16 = i16::from_bool(false);
            assert_eq!(false_i16.try_into_bool(), Some(false));

            let false_i32 = i32::from_bool(false);
            assert_eq!(false_i32.try_into_bool(), Some(false));

            let false_i64 = i64::from_bool(false);
            assert_eq!(false_i64.try_into_bool(), Some(false));

            let false_isize = isize::from_bool(false);
            assert_eq!(false_isize.try_into_bool(), Some(false));
        }
    }

    mod dataorsuffstat {
        use super::*;
        use crate::data::GaussianSuffStat;
        use crate::dist::Gaussian;
        use crate::traits::{
            ConjugatePrior, HasSuffStat, Sampleable, SuffStat,
        };
        use rand::Rng;

        struct MockConjPrior;

        impl Sampleable<Gaussian> for MockConjPrior {
            fn draw<R: Rng>(&self, _rng: &mut R) -> Gaussian {
                Gaussian::standard()
            }
        }

        // For the tests, we use a simple wrapper around Gaussian
        // that satisfies the Sampleable<Gaussian> trait
        #[derive(Debug, Clone, PartialEq)]
        struct MockPosterior(Gaussian);

        impl Sampleable<Gaussian> for MockPosterior {
            fn draw<R: Rng>(&self, _rng: &mut R) -> Gaussian {
                self.0.clone()
            }
        }

        impl ConjugatePrior<f64, Gaussian> for MockConjPrior {
            type Posterior = MockPosterior;
            type MCache = ();
            type PpCache = ();

            fn posterior(
                &self,
                _x: &crate::data::GaussianData<f64>,
            ) -> Self::Posterior {
                MockPosterior(Gaussian::standard())
            }

            fn ln_m(&self, _x: &crate::data::GaussianData<f64>) -> f64 {
                0.0
            }

            fn ln_m_cache(&self) -> Self::MCache {}

            fn ln_m_with_cache(
                &self,
                _cache: &Self::MCache,
                _x: &crate::data::GaussianData<f64>,
            ) -> f64 {
                0.0
            }

            fn ln_pp(
                &self,
                _y: &f64,
                _x: &crate::data::GaussianData<f64>,
            ) -> f64 {
                0.0
            }

            fn ln_pp_cache(
                &self,
                _x: &crate::data::GaussianData<f64>,
            ) -> Self::PpCache {
            }

            fn ln_pp_with_cache(
                &self,
                _cache: &Self::PpCache,
                _y: &f64,
            ) -> f64 {
                0.0
            }

            fn empty_stat(&self) -> <Gaussian as HasSuffStat<f64>>::Stat {
                GaussianSuffStat::new()
            }
        }

        #[test]
        fn extract_stat_from_suffstat() {
            let pr = MockConjPrior;
            let mut stats = GaussianSuffStat::new();
            stats.observe(&1.0);
            stats.observe(&2.0);

            let data: DataOrSuffStat<f64, Gaussian> =
                DataOrSuffStat::SuffStat(&stats);

            let extracted = extract_stat(&pr, &data);
            assert_eq!(extracted.n(), 2);
            assert_eq!(extracted.sum_x(), 3.0);
        }

        #[test]
        fn extract_stat_from_data() {
            let pr = MockConjPrior;
            let data_vec = vec![1.0, 2.0, 3.0];

            let data: DataOrSuffStat<f64, Gaussian> =
                DataOrSuffStat::Data(&data_vec);

            let extracted = extract_stat(&pr, &data);
            assert_eq!(extracted.n(), 3);
            assert_eq!(extracted.sum_x(), 6.0);
        }

        #[test]
        fn extract_stat_then_from_suffstat() {
            let pr = MockConjPrior;
            let mut stats = GaussianSuffStat::new();
            stats.observe(&1.0);
            stats.observe(&2.0);

            let data: DataOrSuffStat<f64, Gaussian> =
                DataOrSuffStat::SuffStat(&stats);

            let result = extract_stat_then(&pr, &data, |stat| {
                stat.n() * 10 + (stat.sum_x() as usize)
            });

            assert_eq!(result, 23); // 2 * 10 + 3
        }

        #[test]
        fn extract_stat_then_from_data() {
            let pr = MockConjPrior;
            let data_vec = vec![1.0, 2.0, 3.0];

            let data: DataOrSuffStat<f64, Gaussian> =
                DataOrSuffStat::Data(&data_vec);

            let result =
                extract_stat_then(&pr, &data, |stat: &GaussianSuffStat| {
                    stat.n() * 10 + (stat.sum_x() as usize)
                });

            assert_eq!(result, 36); // 3 * 10 + 6
        }

        #[test]
        fn test_n_method_with_data() {
            // Test n() with Data variant
            let data_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let data = DataOrSuffStat::<f64, Gaussian>::Data(&data_vec);

            assert_eq!(data.n(), 5);

            // Empty data case
            let empty_vec: Vec<f64> = vec![];
            let empty_data = DataOrSuffStat::<f64, Gaussian>::Data(&empty_vec);

            assert_eq!(empty_data.n(), 0);
        }

        #[test]
        fn test_n_method_with_suffstat() {
            // Test n() with SuffStat variant
            let mut stats = GaussianSuffStat::new();
            stats.observe(&1.0);
            stats.observe(&2.0);
            stats.observe(&3.0);

            let suffstat = DataOrSuffStat::<f64, Gaussian>::SuffStat(&stats);

            assert_eq!(suffstat.n(), 3);

            // Empty suffstat case
            let empty_stats = GaussianSuffStat::new();
            let empty_suffstat =
                DataOrSuffStat::<f64, Gaussian>::SuffStat(&empty_stats);

            assert_eq!(empty_suffstat.n(), 0);
        }

        #[test]
        fn test_is_data_method() {
            // Test is_data() returns true for Data variant
            let data_vec = vec![1.0, 2.0, 3.0];
            let data = DataOrSuffStat::<f64, Gaussian>::Data(&data_vec);

            assert!(data.is_data());

            // Test is_data() returns false for SuffStat variant
            let stats = GaussianSuffStat::new();
            let suffstat = DataOrSuffStat::<f64, Gaussian>::SuffStat(&stats);

            assert!(!suffstat.is_data());
        }

        #[test]
        fn test_is_suffstat_method() {
            // Test is_suffstat() returns true for SuffStat variant
            let stats = GaussianSuffStat::new();
            let suffstat = DataOrSuffStat::<f64, Gaussian>::SuffStat(&stats);

            assert!(suffstat.is_suffstat());

            // Test is_suffstat() returns false for Data variant
            let data_vec = vec![1.0, 2.0, 3.0];
            let data = DataOrSuffStat::<f64, Gaussian>::Data(&data_vec);

            assert!(!data.is_suffstat());
        }
    }
}
