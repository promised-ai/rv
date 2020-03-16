#[cfg(feature = "serde1")]
use serde_derive::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::Gaussian;
use crate::traits::SuffStat;

/// Gaussian sufficient statistic.
///
/// Holds the number of observations, their sum, and the sum of their squared
/// values.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct GaussianSuffStat {
    /// Number of observations
    n: usize,
    /// Sum of `x`
    sum_x: f64,
    /// Sum of `x^2`
    sum_x_sq: f64,
}

impl GaussianSuffStat {
    #[inline]
    pub fn new() -> Self {
        GaussianSuffStat {
            n: 0,
            sum_x: 0.0,
            sum_x_sq: 0.0,
        }
    }

    /// Get the number of observations
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sum of observations
    #[inline]
    pub fn sum_x(&self) -> f64 {
        self.sum_x
    }

    /// Get the sum of squared observations
    #[inline]
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

        // TODO: store in a more numerically stable form
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
