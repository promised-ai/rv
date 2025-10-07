#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::UnitPowerLaw;
use crate::traits::SuffStat;

/// Inverse Gamma sufficient statistic.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct UnitPowerLawSuffStat {
    /// Number of observations
    n: usize,
    /// sum of `ln(x)`
    sum_ln_x: f64,
}

impl UnitPowerLawSuffStat {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        UnitPowerLawSuffStat {
            n: 0,
            sum_ln_x: 0.0,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    ///
    /// # Example
    /// ```rust
    /// use rv::data::UnitPowerLawSuffStat;
    /// use rv::prelude::SuffStat;
    ///
    /// let data: Vec<f64> = vec![0.1, 0.2, 0.3];
    ///
    /// let mut stat_b = UnitPowerLawSuffStat::new();
    /// stat_b.observe_many(&data);
    ///
    /// let n = data.len();
    /// let sum_ln_x = data.iter().map(|x: &f64| x.ln()).sum();
    ///
    /// let stat_a = UnitPowerLawSuffStat::from_parts_unchecked(n, sum_ln_x);
    ///
    /// assert_eq!(stat_a.n(), stat_b.n());
    /// assert::close(stat_a.sum_ln_x(), stat_b.sum_ln_x(), 1e-10);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_parts_unchecked(n: usize, sum_ln_x: f64) -> Self {
        UnitPowerLawSuffStat { n, sum_ln_x }
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

            fn merge(&mut self, other: Self) {
                self.n += other.n;
                self.sum_ln_x += other.sum_ln_x;
            }
        }
    };
}

impl_suffstat!(f32);
impl_suffstat!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_forget() {
        let mut stat = UnitPowerLawSuffStat::new();

        stat.observe(&0.1);
        stat.observe(&0.2);

        assert_eq!(stat.n(), 2);
        assert::close(stat.sum_ln_x, (0.1_f64).ln() + (0.2_f64).ln(), 1e-10);

        stat.forget(&0.1);

        assert_eq!(stat.n(), 1);
        assert::close(stat.sum_ln_x, (0.2_f64).ln(), 1e-10);

        stat.forget(&0.2);

        assert_eq!(stat.n(), 0);
        assert_eq!(stat.sum_ln_x, 0.0);
    }

    #[test]
    fn merge() {
        let mut a = UnitPowerLawSuffStat::new();
        let mut b = UnitPowerLawSuffStat::new();
        let mut c = UnitPowerLawSuffStat::new();

        a.observe_many(&[0.1_f64, 0.2, 0.3]);
        b.observe_many(&[0.9_f64, 0.8, 0.7]);

        c.observe_many(&[0.1_f64, 0.2, 0.3, 0.9, 0.8, 0.7]);

        <UnitPowerLawSuffStat as SuffStat<f64>>::merge(&mut a, b);

        assert_eq!(a, c);
    }
}
