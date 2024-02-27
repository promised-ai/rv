#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::UnitPowerLaw;
use crate::suffstat_traits::SuffStat;

/// Inverse Gamma sufficient statistic.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct UnitPowerLawSuffStat {
    /// Number of observations
    pub n: usize,
    /// sum of `ln(x)`
    pub sum_ln_x: f64,
}

impl UnitPowerLawSuffStat {
    #[inline]
    pub fn new() -> Self {
        UnitPowerLawSuffStat {
            n: 0,
            sum_ln_x: 0.0,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    pub fn from_parts_unchecked(n: usize, sum_ln_x: f64) -> Self {
        UnitPowerLawSuffStat { n, sum_ln_x }
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
}

impl Default for UnitPowerLawSuffStat {
    fn default() -> Self {
        UnitPowerLawSuffStat::new()
    }
}

macro_rules! impl_suffstat {
    ($kind:ty) => {
        impl<'a> From<&'a UnitPowerLawSuffStat>
            for DataOrSuffStat<'a, $kind, UnitPowerLaw>
        {
            fn from(stat: &'a UnitPowerLawSuffStat) -> Self {
                DataOrSuffStat::SuffStat(stat)
            }
        }

        impl<'a> From<&'a Vec<$kind>>
            for DataOrSuffStat<'a, $kind, UnitPowerLaw>
        {
            fn from(xs: &'a Vec<$kind>) -> Self {
                DataOrSuffStat::Data(xs.as_slice())
            }
        }

        impl<'a> From<&'a [$kind]> for DataOrSuffStat<'a, $kind, UnitPowerLaw> {
            fn from(xs: &'a [$kind]) -> Self {
                DataOrSuffStat::Data(xs)
            }
        }

        impl From<&Vec<$kind>> for UnitPowerLawSuffStat {
            fn from(xs: &Vec<$kind>) -> Self {
                let mut stat = UnitPowerLawSuffStat::new();
                stat.observe_many(xs);
                stat
            }
        }

        impl From<&[$kind]> for UnitPowerLawSuffStat {
            fn from(xs: &[$kind]) -> Self {
                let mut stat = UnitPowerLawSuffStat::new();
                stat.observe_many(xs);
                stat
            }
        }

        impl SuffStat<$kind> for UnitPowerLawSuffStat {
            fn n(&self) -> usize {
                self.n
            }

            fn observe(&mut self, x: &$kind) {
                let xf = f64::from(*x);
                self.n += 1;
                self.sum_ln_x += xf.ln();
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = f64::from(*x);
                    self.sum_ln_x -= xf.ln();
                    self.n -= 1;
                } else {
                    self.n = 0;
                    self.sum_ln_x = 0.0;
                }
            }

            fn observe_many(&mut self, xs: &[$kind]) {
                self.n += xs.len();
                self.sum_ln_x +=
                    xs.iter().map(|x| f64::from(*x)).product::<f64>().ln();
            }

            fn forget_many(&mut self, xs: &[$kind]) {
                self.n -= xs.len();
                self.sum_ln_x -=
                    xs.iter().map(|x| f64::from(*x)).product::<f64>().ln();
            }
        }
    };
}

impl_suffstat!(f32);
impl_suffstat!(f64);
