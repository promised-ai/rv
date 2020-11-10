#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::Poisson;
use crate::misc::ln_fact;
use crate::traits::SuffStat;

/// Poisson sufficient statistic.
///
/// Holds the number of observations and their sum.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct PoissonSuffStat {
    /// Number of observations
    n: usize,
    /// Sum of observations
    sum: f64,
    /// Sum of Log(x!)
    sum_ln_fact: f64,
}

impl PoissonSuffStat {
    /// Create a new empty SuffStat
    #[inline]
    pub fn new() -> Self {
        Self {
            n: 0,
            sum: 0.0,
            sum_ln_fact: 0.0,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    pub fn from_parts_unchecked(n: usize, sum: f64, sum_ln_fact: f64) -> Self {
        Self {
            n,
            sum,
            sum_ln_fact,
        }
    }

    /// Get the number of observations
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sum of all observations
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    #[inline]
    pub fn sum_ln_fact(&self) -> f64 {
        self.sum_ln_fact
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
                self.sum_ln_fact += ln_fact(*x as usize);
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = f64::from(*x);
                    self.n -= 1;
                    self.sum -= xf;
                    self.sum_ln_fact -= ln_fact(*x as usize);
                } else {
                    self.n = 0;
                    self.sum = 0.0;
                    self.sum_ln_fact = 0.0;
                }
            }
        }
    };
}

impl_poisson_suffstat!(u8);
impl_poisson_suffstat!(u16);
impl_poisson_suffstat!(u32);
