#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::InvGamma;
use crate::traits::SuffStat;

/// Inverse Gamma sufficient statistic.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct InvGammaSuffStat {
    /// Number of observations
    n: usize,
    /// sum of `ln(x)`
    sum_ln_x: f64,
    /// sum of `1/x`
    sum_inv_x: f64,
}

impl InvGammaSuffStat {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        InvGammaSuffStat {
            n: 0,
            sum_ln_x: 0.0,
            sum_inv_x: 0.0,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    #[must_use]
    pub fn from_parts_unchecked(
        n: usize,
        sum_ln_x: f64,
        sum_inv_x: f64,
    ) -> Self {
        InvGammaSuffStat {
            n,
            sum_ln_x,
            sum_inv_x,
        }
    }

    /// Get the number of observations
    #[inline]
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Sum of `ln(x)`
    #[inline]
    #[must_use]
    pub fn sum_ln_x(&self) -> f64 {
        self.sum_ln_x
    }

    /// Sum of `1/x`
    #[inline]
    #[must_use]
    pub fn sum_inv_x(&self) -> f64 {
        self.sum_inv_x
    }
}

impl Default for InvGammaSuffStat {
    fn default() -> Self {
        InvGammaSuffStat::new()
    }
}

macro_rules! impl_suffstat {
    ($kind:ty) => {
        impl<'a> From<&'a InvGammaSuffStat>
            for DataOrSuffStat<'a, $kind, InvGamma>
        {
            fn from(stat: &'a InvGammaSuffStat) -> Self {
                DataOrSuffStat::SuffStat(stat)
            }
        }

        impl<'a> From<&'a Vec<$kind>> for DataOrSuffStat<'a, $kind, InvGamma> {
            fn from(xs: &'a Vec<$kind>) -> Self {
                DataOrSuffStat::Data(xs.as_slice())
            }
        }

        impl<'a> From<&'a [$kind]> for DataOrSuffStat<'a, $kind, InvGamma> {
            fn from(xs: &'a [$kind]) -> Self {
                DataOrSuffStat::Data(xs)
            }
        }

        impl From<&Vec<$kind>> for InvGammaSuffStat {
            fn from(xs: &Vec<$kind>) -> Self {
                let mut stat = InvGammaSuffStat::new();
                stat.observe_many(xs);
                stat
            }
        }

        impl From<&[$kind]> for InvGammaSuffStat {
            fn from(xs: &[$kind]) -> Self {
                let mut stat = InvGammaSuffStat::new();
                stat.observe_many(xs);
                stat
            }
        }

        impl SuffStat<$kind> for InvGammaSuffStat {
            fn n(&self) -> usize {
                self.n
            }

            fn observe(&mut self, x: &$kind) {
                let xf = f64::from(*x);

                self.n += 1;

                self.sum_ln_x += xf.ln();
                self.sum_inv_x += xf.recip();
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = f64::from(*x);

                    self.sum_ln_x -= xf.ln();
                    self.sum_inv_x -= xf.recip();
                    self.n -= 1;
                } else {
                    self.n = 0;
                    self.sum_ln_x = 0.0;
                    self.sum_inv_x = 0.0;
                }
            }

            fn merge(&mut self, other: Self) {
                self.n += other.n;
                self.sum_ln_x += other.sum_ln_x;
                self.sum_inv_x += other.sum_inv_x;
            }
        }
    };
}

impl_suffstat!(f32);
impl_suffstat!(f64);
