#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::data::Booleable;
use crate::data::CategoricalDatum;
use crate::data::DataOrSuffStat;
use crate::dist::{Bernoulli, Categorical, Gaussian, Poisson};
use crate::traits::SuffStat;
use nalgebra::{DMatrix, DVector};

use special::Gamma as SGamma;

/// Sufficent statistic for the Bernoulli distribution.
///
/// Contains the number of trials and the number of successes.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct BernoulliSuffStat {
    n: usize,
    k: usize,
}

impl BernoulliSuffStat {
    /// Create a new Bernoulli sufficient statistic
    pub fn new() -> Self {
        BernoulliSuffStat { n: 0, k: 0 }
    }

    /// Get the total number of trials, n.
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::BernoulliSuffStat;
    /// # use rv::traits::SuffStat;
    /// let mut stat = BernoulliSuffStat::new();
    ///
    /// stat.observe(&true);
    /// stat.observe(&false);
    ///
    /// assert_eq!(stat.n(), 2);
    /// ```
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the number of successful trials, k.
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::BernoulliSuffStat;
    /// # use rv::traits::SuffStat;
    /// let mut stat = BernoulliSuffStat::new();
    ///
    /// stat.observe(&true);
    /// stat.observe(&false);
    ///
    /// assert_eq!(stat.k(), 1);
    /// ```
    pub fn k(&self) -> usize {
        self.k
    }
}

impl Default for BernoulliSuffStat {
    fn default() -> Self {
        BernoulliSuffStat::new()
    }
}

impl SuffStat<bool> for BernoulliSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &bool) {
        self.n += 1;
        if *x {
            self.k += 1
        }
    }

    fn forget(&mut self, x: &bool) {
        self.n -= 1;
        if *x {
            self.k -= 1
        }
    }
}

impl<'a> Into<DataOrSuffStat<'a, bool, Bernoulli>> for &'a BernoulliSuffStat {
    fn into(self) -> DataOrSuffStat<'a, bool, Bernoulli> {
        DataOrSuffStat::SuffStat(self)
    }
}

impl<'a> Into<DataOrSuffStat<'a, bool, Bernoulli>> for &'a Vec<bool> {
    fn into(self) -> DataOrSuffStat<'a, bool, Bernoulli> {
        DataOrSuffStat::Data(self)
    }
}

impl<'a, X: Booleable> Into<DataOrSuffStat<'a, X, Bernoulli>>
    for &'a BernoulliSuffStat
{
    fn into(self) -> DataOrSuffStat<'a, X, Bernoulli> {
        DataOrSuffStat::SuffStat(self)
    }
}

impl<'a, X: Booleable> Into<DataOrSuffStat<'a, X, Bernoulli>> for &'a Vec<X> {
    fn into(self) -> DataOrSuffStat<'a, X, Bernoulli> {
        DataOrSuffStat::Data(self)
    }
}

impl<X: Booleable> SuffStat<X> for BernoulliSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &X) {
        self.n += 1;
        if x.into_bool() {
            self.k += 1
        }
    }

    fn forget(&mut self, x: &X) {
        self.n -= 1;
        if x.into_bool() {
            self.k -= 1
        }
    }
}

/// Categorical distribution sufficient statistic.
///
/// Store the number of observations and the count of observations of each
/// instance.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct CategoricalSuffStat {
    n: usize,
    counts: Vec<f64>,
}

impl CategoricalSuffStat {
    pub fn new(k: usize) -> Self {
        CategoricalSuffStat {
            n: 0,
            counts: vec![0.0; k],
        }
    }

    /// Get the total number of trials
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::CategoricalSuffStat;
    /// # use rv::traits::SuffStat;
    /// let mut stat = CategoricalSuffStat::new(3);
    ///
    /// stat.observe(&0_u8);
    /// stat.observe(&1_u8);
    /// stat.observe(&1_u8);
    ///
    /// assert_eq!(stat.n(), 3);
    /// ```
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the number of occurrences of each class, counts
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::CategoricalSuffStat;
    /// # use rv::traits::SuffStat;
    /// let mut stat = CategoricalSuffStat::new(3);
    ///
    /// stat.observe(&0_u8);
    /// stat.observe(&1_u8);
    /// stat.observe(&1_u8);
    ///
    /// assert_eq!(*stat.counts(), vec![1.0, 2.0, 0.0]);
    /// ```
    pub fn counts(&self) -> &Vec<f64> {
        &self.counts
    }
}

impl<'a, X> Into<DataOrSuffStat<'a, X, Categorical>> for &'a CategoricalSuffStat
where
    X: CategoricalDatum,
{
    fn into(self) -> DataOrSuffStat<'a, X, Categorical> {
        DataOrSuffStat::SuffStat(self)
    }
}

// XXX Can't just implement for X: CategoricalDatum because of orphan type rule
macro_rules! impl_into_dos_for_categorical {
    ($kind: ty) => {
        impl<'a> Into<DataOrSuffStat<'a, $kind, Categorical>>
            for &'a Vec<$kind>
        {
            fn into(self) -> DataOrSuffStat<'a, $kind, Categorical> {
                DataOrSuffStat::Data(self)
            }
        }
    };
}

impl_into_dos_for_categorical!(bool);
impl_into_dos_for_categorical!(usize);
impl_into_dos_for_categorical!(u8);
impl_into_dos_for_categorical!(u16);
impl_into_dos_for_categorical!(u32);

impl<X: CategoricalDatum> SuffStat<X> for CategoricalSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &X) {
        let ix = x.into_usize();
        self.n += 1;
        self.counts[ix] += 1.0;
    }

    fn forget(&mut self, x: &X) {
        let ix = x.into_usize();
        self.n -= 1;
        self.counts[ix] -= 1.0;
    }
}

/// Poisson sufficient statistic.
///
/// Holds the number of observarions and their sum.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct PoissonSuffStat {
    /// Number of observations
    n: usize,
    /// Sum of observations
    sum: f64,
    /// Sum of Log(x!)
    sum_log_fact: f64,
}

impl PoissonSuffStat {
    /// Create a new empty SuffStat
    pub fn new() -> Self {
        Self {
            n: 0,
            sum: 0.0,
            sum_log_fact: 0.0,
        }
    }

    /// Get the number of observations
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sum of all observations
    pub fn sum(&self) -> f64 {
        self.sum
    }

    pub fn sum_log_fact(&self) -> f64 {
        self.sum_log_fact
    }
}

impl Default for PoissonSuffStat {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_poisson_suffstat {
    ($kind:ty) => {
        impl<'a> Into<DataOrSuffStat<'a, $kind, Poisson>>
            for &'a PoissonSuffStat
        {
            fn into(self) -> DataOrSuffStat<'a, $kind, Poisson> {
                DataOrSuffStat::SuffStat(self)
            }
        }

        impl<'a> Into<DataOrSuffStat<'a, $kind, Poisson>> for &'a Vec<$kind> {
            fn into(self) -> DataOrSuffStat<'a, $kind, Poisson> {
                DataOrSuffStat::Data(self)
            }
        }

        impl SuffStat<$kind> for PoissonSuffStat {
            fn n(&self) -> usize {
                self.n
            }

            fn observe(&mut self, x: &$kind) {
                let xf = f64::from(*x);
                self.n += 1;
                self.sum += xf;
                self.sum_log_fact += f64::from(*x + 1).ln_gamma().0;
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = f64::from(*x);
                    self.n -= 1;
                    self.sum -= xf;
                    self.sum_log_fact -= f64::from(*x + 1).ln_gamma().0;
                } else {
                    self.n = 0;
                    self.sum = 0.0;
                    self.sum_log_fact = 0.0;
                }
            }
        }
    };
}

impl_poisson_suffstat!(u8);
impl_poisson_suffstat!(u16);
impl_poisson_suffstat!(u32);

/// Gaussian sufficient statistic.
///
/// Holds the number of observations, their sum, and the sum of their squared
/// values.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct GaussianSuffStat {
    /// Number of observations
    n: usize,
    /// Sum of `x`
    sum_x: f64,
    /// Sum of `x^2`
    sum_x_sq: f64,
}

impl GaussianSuffStat {
    pub fn new() -> Self {
        GaussianSuffStat {
            n: 0,
            sum_x: 0.0,
            sum_x_sq: 0.0,
        }
    }

    /// Get the number of observations
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sum of observations
    pub fn sum_x(&self) -> f64 {
        self.sum_x
    }

    /// Get the sum of squared observations
    pub fn sum_x_sq(&self) -> f64 {
        self.sum_x_sq
    }
}

impl Default for GaussianSuffStat {
    fn default() -> Self {
        GaussianSuffStat::new()
    }
}

macro_rules! impl_gaussian_suffstat {
    ($kind:ty) => {
        impl<'a> Into<DataOrSuffStat<'a, $kind, Gaussian>>
            for &'a GaussianSuffStat
        {
            fn into(self) -> DataOrSuffStat<'a, $kind, Gaussian> {
                DataOrSuffStat::SuffStat(self)
            }
        }

        impl<'a> Into<DataOrSuffStat<'a, $kind, Gaussian>> for &'a Vec<$kind> {
            fn into(self) -> DataOrSuffStat<'a, $kind, Gaussian> {
                DataOrSuffStat::Data(self)
            }
        }

        // TODO: store in a more nuerically stable form
        impl SuffStat<$kind> for GaussianSuffStat {
            fn n(&self) -> usize {
                self.n
            }

            fn observe(&mut self, x: &$kind) {
                let xf = f64::from(*x);
                self.n += 1;
                self.sum_x += xf;
                self.sum_x_sq += xf.powi(2);
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = f64::from(*x);
                    self.n -= 1;
                    self.sum_x -= xf;
                    self.sum_x_sq -= xf.powi(2);
                } else {
                    self.n = 0;
                    self.sum_x = 0.0;
                    self.sum_x_sq = 0.0;
                }
            }
        }
    };
}

impl_gaussian_suffstat!(f32);
impl_gaussian_suffstat!(f64);

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct MvGaussianSuffStat {
    n: usize,
    sum_x: DVector<f64>,
    sum_x_sq: DMatrix<f64>,
}

impl MvGaussianSuffStat {
    pub fn new(dims: usize) -> Self {
        MvGaussianSuffStat {
            n: 0,
            sum_x: DVector::zeros(dims),
            sum_x_sq: DMatrix::zeros(dims, dims),
        }
    }

    /// Get the number of observations
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sum of observations
    pub fn sum_x(&self) -> &DVector<f64> {
        &self.sum_x
    }

    /// Get the sum of X^2
    pub fn sum_x_sq(&self) -> &DMatrix<f64> {
        &self.sum_x_sq
    }
}

impl SuffStat<DVector<f64>> for MvGaussianSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &DVector<f64>) {
        self.n += 1;
        if self.n == 1 {
            self.sum_x = x.clone();
            self.sum_x_sq = x * x.transpose();
        } else {
            self.sum_x += x;
            self.sum_x_sq += x * x.transpose();
        }
    }

    fn forget(&mut self, x: &DVector<f64>) {
        self.n -= 1;
        if self.n > 0 {
            self.sum_x -= x;
            self.sum_x_sq -= x * x.transpose();
        } else {
            let dims = self.sum_x.len();
            self.sum_x = DVector::zeros(dims);
            self.sum_x_sq = DMatrix::zeros(dims, dims);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod bernoulli {
        use super::*;

        #[test]
        fn new_should_be_empty() {
            let stat = BernoulliSuffStat::new();
            assert_eq!(stat.n, 0);
            assert_eq!(stat.k, 0);
        }

        #[test]
        fn observe_1() {
            let mut stat = BernoulliSuffStat::new();
            stat.observe(&1_u8);
            assert_eq!(stat.n, 1);
            assert_eq!(stat.k, 1);
        }

        #[test]
        fn observe_true() {
            let mut stat = BernoulliSuffStat::new();
            stat.observe(&true);
            assert_eq!(stat.n, 1);
            assert_eq!(stat.k, 1);
        }

        #[test]
        fn observe_0() {
            let mut stat = BernoulliSuffStat::new();
            stat.observe(&0_i8);
            assert_eq!(stat.n, 1);
            assert_eq!(stat.k, 0);
        }

        #[test]
        fn observe_false() {
            let mut stat = BernoulliSuffStat::new();
            stat.observe(&false);
            assert_eq!(stat.n, 1);
            assert_eq!(stat.k, 0);
        }
    }

    mod categorical {
        use super::*;

        #[test]
        fn new() {
            let sf = CategoricalSuffStat::new(4);
            assert_eq!(sf.counts.len(), 4);
            assert_eq!(sf.n, 0);
            assert!(sf.counts.iter().all(|&ct| ct.abs() < 1E-12))
        }
    }
}
