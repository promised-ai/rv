#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::InvGaussian;
use crate::suffstat_traits::SuffStat;

/// Gaussian sufficient statistic.
///
/// Holds the number of observations, their sum, and the sum of their squared
/// values.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct InvGaussianSuffStat {
    /// Number of observations
    n: usize,
    /// sum of `x`
    sum_x: f64,
    /// sum of `1/x`
    sum_inv_x: f64,
    /// sum of ln(x)
    sum_ln_x: f64,
}

impl InvGaussianSuffStat {
    #[inline]
    pub fn new() -> Self {
        InvGaussianSuffStat {
            n: 0,
            sum_x: 0.0,
            sum_inv_x: 0.0,
            sum_ln_x: 0.0,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    pub fn from_parts_unchecked(
        n: usize,
        sum_x: f64,
        sum_inv_x: f64,
        sum_ln_x: f64,
    ) -> Self {
        InvGaussianSuffStat {
            n,
            sum_x,
            sum_inv_x,
            sum_ln_x,
        }
    }

    /// Get the number of observations
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sample mean
    #[inline]
    pub fn mean(&self) -> f64 {
        self.sum_x / self.n as f64
    }

    /// Sum of `x`
    #[inline]
    pub fn sum_x(&self) -> f64 {
        self.sum_x
    }

    /// Sum of `1/x`
    #[inline]
    pub fn sum_inv_x(&self) -> f64 {
        self.sum_inv_x
    }

    #[inline]
    pub fn sum_ln_x(&self) -> f64 {
        self.sum_ln_x
    }
}

impl Default for InvGaussianSuffStat {
    fn default() -> Self {
        InvGaussianSuffStat::new()
    }
}

macro_rules! impl_invgaussian_suffstat {
    ($kind:ty) => {
        impl<'a> From<&'a InvGaussianSuffStat>
            for DataOrSuffStat<'a, $kind, InvGaussian>
        {
            fn from(stat: &'a InvGaussianSuffStat) -> Self {
                DataOrSuffStat::SuffStat(stat)
            }
        }

        impl<'a> From<&'a Vec<$kind>>
            for DataOrSuffStat<'a, $kind, InvGaussian>
        {
            fn from(xs: &'a Vec<$kind>) -> Self {
                DataOrSuffStat::Data(xs.as_slice())
            }
        }

        impl<'a> From<&'a [$kind]> for DataOrSuffStat<'a, $kind, InvGaussian> {
            fn from(xs: &'a [$kind]) -> Self {
                DataOrSuffStat::Data(xs)
            }
        }

        impl From<&Vec<$kind>> for InvGaussianSuffStat {
            fn from(xs: &Vec<$kind>) -> Self {
                let mut stat = InvGaussianSuffStat::new();
                stat.observe_many(xs);
                stat
            }
        }

        impl From<&[$kind]> for InvGaussianSuffStat {
            fn from(xs: &[$kind]) -> Self {
                let mut stat = InvGaussianSuffStat::new();
                stat.observe_many(xs);
                stat
            }
        }

        impl SuffStat<$kind> for InvGaussianSuffStat {
            fn n(&self) -> usize {
                self.n
            }

            fn observe(&mut self, x: &$kind) {
                let xf = f64::from(*x);

                self.n += 1;

                self.sum_x += xf;
                self.sum_inv_x += xf.recip();
                self.sum_ln_x += xf.ln();
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = f64::from(*x);

                    self.sum_x -= xf;
                    self.sum_inv_x -= xf.recip();
                    self.sum_ln_x -= xf.ln();
                    self.n -= 1;
                } else {
                    self.n = 0;
                    self.sum_x = 0.0;
                    self.sum_inv_x = 0.0;
                    self.sum_ln_x = 0.0;
                }
            }
        }
    };
}

impl_invgaussian_suffstat!(f32);
impl_invgaussian_suffstat!(f64);
