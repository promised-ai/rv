#[cfg(feature = "serde1")]
use serde_derive::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::Poisson;
use crate::traits::SuffStat;
use special::Gamma as _;

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
