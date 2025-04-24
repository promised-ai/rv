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
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct PoissonSuffStat {
    /// Number of observations
    n: usize,
    /// Sum of observations
    sum: f64,
    /// Sum of Log(x!)
    sum_ln_fact: f64,
}

impl PoissonSuffStat {
    /// Create a new empty `SuffStat`
    #[inline]
    #[must_use] pub fn new() -> Self {
        Self {
            n: 0,
            sum: 0.0,
            sum_ln_fact: 0.0,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    #[must_use] pub fn from_parts_unchecked(n: usize, sum: f64, sum_ln_fact: f64) -> Self {
        Self {
            n,
            sum,
            sum_ln_fact,
        }
    }

    /// Get the number of observations
    #[inline]
    #[must_use] pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sum of all observations
    #[inline]
    #[must_use] pub fn sum(&self) -> f64 {
        self.sum
    }

    #[inline]
    #[must_use] pub fn sum_ln_fact(&self) -> f64 {
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
        impl<'a> From<&'a PoissonSuffStat>
            for DataOrSuffStat<'a, $kind, Poisson>
        {
            fn from(stat: &'a PoissonSuffStat) -> Self {
                DataOrSuffStat::SuffStat(stat)
            }
        }

        impl<'a> From<&'a Vec<$kind>> for DataOrSuffStat<'a, $kind, Poisson> {
            fn from(xs: &'a Vec<$kind>) -> Self {
                DataOrSuffStat::Data(xs)
            }
        }

        impl<'a> From<&'a [$kind]> for DataOrSuffStat<'a, $kind, Poisson> {
            fn from(xs: &'a [$kind]) -> Self {
                DataOrSuffStat::Data(xs)
            }
        }

        impl SuffStat<$kind> for PoissonSuffStat {
            fn n(&self) -> usize {
                self.n
            }

            fn observe(&mut self, x: &$kind) {
                let xf = *x as f64;
                self.n += 1;
                self.sum += xf;
                self.sum_ln_fact += ln_fact(*x as usize);
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = *x as f64;
                    self.n -= 1;
                    self.sum -= xf;
                    self.sum_ln_fact -= ln_fact(*x as usize);
                } else {
                    self.n = 0;
                    self.sum = 0.0;
                    self.sum_ln_fact = 0.0;
                }
            }

            fn merge(&mut self, other: Self) {
                self.n += other.n;
                self.sum += other.sum;
                self.sum_ln_fact += other.sum_ln_fact;
            }
        }
    };
}

impl_poisson_suffstat!(u8);
impl_poisson_suffstat!(u16);
impl_poisson_suffstat!(u32);
impl_poisson_suffstat!(usize);
