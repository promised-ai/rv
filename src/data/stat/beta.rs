#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::Beta;
use crate::traits::SuffStat;

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
    #[must_use]
    pub fn new() -> Self {
        BetaSuffStat {
            n: 0,
            sum_ln_x: 0.0,
            sum_ln_1mx: 0.0,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    ///
    /// # Example
    /// ```rust
    /// use rv::data::BetaSuffStat;
    /// use rv::prelude::SuffStat;
    ///
    /// let data: Vec<f64> = vec![0.1, 0.2, 0.3];
    ///
    /// let mut stat_b = BetaSuffStat::new();
    /// stat_b.observe_many(&data);
    ///
    /// let n = data.len();
    /// let sum_ln_x = data.iter().map(|x: &f64| x.ln()).sum();
    /// let sum_ln_1mx = data.iter().map(|x: &f64| (1.0-x).ln()).sum();
    ///
    /// let stat_a = BetaSuffStat::from_parts_unchecked(n, sum_ln_x, sum_ln_1mx);
    ///
    /// assert_eq!(stat_a, stat_b);
    /// ```
    #[inline]
    #[must_use]
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

    /// Sum of `ln(1-x)`
    #[inline]
    #[must_use]
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

            fn merge(&mut self, other: Self) {
                self.n += other.n;
                self.sum_ln_x += other.sum_ln_x;
                self.sum_ln_1mx += other.sum_ln_1mx;
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
        let mut stat = BetaSuffStat::new();

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
        let mut a = BetaSuffStat::new();
        let mut b = BetaSuffStat::new();
        let mut c = BetaSuffStat::new();

        a.observe_many(&[0.1_f64, 0.2, 0.3]);
        b.observe_many(&[0.9_f64, 0.8, 0.7]);

        c.observe_many(&[0.1_f64, 0.2, 0.3, 0.9, 0.8, 0.7]);

        <BetaSuffStat as SuffStat<f64>>::merge(&mut a, b);

        assert_eq!(a, c);
    }
}
