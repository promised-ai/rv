#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::Beta;
use crate::suffstat_traits::SuffStat; 

/// Inverse Gamma sufficient statistic.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct BetaSuffStat {
    /// Number of observations
    n: usize,
    /// sum of `ln(x)`
    sum_ln_x: f64,
    /// sum of `ln(1 - x)`
    sum_ln_1mx: f64,
}

impl BetaSuffStat {
    #[inline]
    pub fn new() -> Self {
        BetaSuffStat {
            n: 0,
            sum_ln_x: 0.0,
            sum_ln_1mx: 0.0,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    pub fn from_parts_unchecked(
        n: usize,
        sum_ln_x: f64,
        sum_ln_1mx: f64,
    ) -> Self {
        BetaSuffStat {
            n,
            sum_ln_x,
            sum_ln_1mx,
        }
    }

    /// Get the number of observations
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Sum of `ln(x)`
    #[inline]
    pub fn sum_ln_x(&self) -> f64 {
        self.sum_ln_x
    }

    /// Sum of `ln(1-x)`
    #[inline]
    pub fn sum_ln_1mx(&self) -> f64 {
        self.sum_ln_1mx
    }
}

impl Default for BetaSuffStat {
    fn default() -> Self {
        BetaSuffStat::new()
    }
}

macro_rules! impl_suffstat {
    ($kind:ty) => {
        impl<'a> From<&'a BetaSuffStat> for DataOrSuffStat<'a, $kind, Beta> {
            fn from(stat: &'a BetaSuffStat) -> Self {
                DataOrSuffStat::SuffStat(stat)
            }
        }

        impl<'a> From<&'a Vec<$kind>> for DataOrSuffStat<'a, $kind, Beta> {
            fn from(xs: &'a Vec<$kind>) -> Self {
                DataOrSuffStat::Data(xs.as_slice())
            }
        }

        impl<'a> From<&'a [$kind]> for DataOrSuffStat<'a, $kind, Beta> {
            fn from(xs: &'a [$kind]) -> Self {
                DataOrSuffStat::Data(xs)
            }
        }

        impl From<&Vec<$kind>> for BetaSuffStat {
            fn from(xs: &Vec<$kind>) -> Self {
                let mut stat = BetaSuffStat::new();
                stat.observe_many(xs);
                stat
            }
        }

        impl From<&[$kind]> for BetaSuffStat {
            fn from(xs: &[$kind]) -> Self {
                let mut stat = BetaSuffStat::new();
                stat.observe_many(xs);
                stat
            }
        }

        impl SuffStat<$kind> for BetaSuffStat {
            fn n(&self) -> usize {
                self.n
            }

            fn observe(&mut self, x: &$kind) {
                let xf = f64::from(*x);

                self.n += 1;

                self.sum_ln_x += xf.ln();
                self.sum_ln_1mx += (1.0 - xf).ln();
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = f64::from(*x);

                    self.sum_ln_x -= xf.ln();
                    self.sum_ln_1mx -= (1.0 - xf).ln();
                    self.n -= 1;
                } else {
                    self.n = 0;
                    self.sum_ln_x = 0.0;
                    self.sum_ln_1mx = 0.0;
                }
            }
        }
    };
}

impl_suffstat!(f32);
impl_suffstat!(f64);
