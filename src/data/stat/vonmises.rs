#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::traits::SuffStat;

/// VonMises sufficient statistic.
///
/// Holds the number of observations, their sum, and the sum of their squared
/// values.
#[derive(Debug, Clone, PartialEq, Copy)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct VonMisesSuffStat {
    /// Number of observations
    n: usize,
    /// ∑ⱼ cos(2πxⱼ/m)
    sum_cos: f64,
    /// ∑ⱼ sin(2πxⱼ/m)
    sum_sin: f64,
}

impl VonMisesSuffStat {
    #[inline]
    pub fn new() -> Self {
        VonMisesSuffStat {
            n: 0,
            sum_cos: 0.0,
            sum_sin: 0.0,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    pub fn from_parts_unchecked(n: usize, sum_cos: f64, sum_sin: f64) -> Self {
        VonMisesSuffStat {
            n,
            sum_cos,
            sum_sin,
        }
    }

    /// Create a sufficient statistic from a slice of data
    ///
    /// Note that we can't have the usual From trait without const generics
    /// because we need to know the modulus.
    pub fn from_data(xs: &[f64]) -> Self {
        let mut stat = VonMisesSuffStat::new();
        for x in xs {
            stat.observe(x);
        }
        stat
    }

    /// Get the number of observations
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sum of cosines
    #[inline]
    pub fn sum_cos(&self) -> f64 {
        self.sum_cos
    }

    /// Get the sum of sines
    #[inline]
    pub fn sum_sin(&self) -> f64 {
        self.sum_sin
    }
}

impl Default for VonMisesSuffStat {
    fn default() -> Self {
        Self::new()
    }
}

impl SuffStat<f64> for VonMisesSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &f64) {
        self.sum_cos += x.cos();
        self.sum_sin += x.sin();
        self.n += 1;
    }

    fn forget(&mut self, x: &f64) {
        self.sum_cos -= x.cos();
        self.sum_sin -= x.sin();
        self.n -= 1;
    }

    fn merge(&mut self, other: Self) {
        if other.n == 0 {
            return;
        }
        self.n += other.n;
        self.sum_cos += other.sum_cos;
        self.sum_sin += other.sum_sin;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_suffstat_has_zero_n() {
        let stat = VonMisesSuffStat::new();
        assert_eq!(stat.n(), 0);
    }

    #[test]
    fn observe_increments_n() {
        let mut stat = VonMisesSuffStat::new();
        stat.observe(&1.0);
        assert_eq!(stat.n(), 1);
    }

    #[test]
    fn forget_decrements_n() {
        let mut stat = VonMisesSuffStat::new();
        stat.observe(&1.0);
        stat.forget(&1.0);
        assert_eq!(stat.n(), 0);
    }

    #[test]
    fn merge_adds_n() {
        let mut stat1 = VonMisesSuffStat::new();
        let mut stat2 = VonMisesSuffStat::new();
        stat1.observe(&1.0);
        stat2.observe(&2.0);
        stat1.merge(stat2);
        assert_eq!(stat1.n(), 2);
    }

    #[test]
    fn merge_empty_stat_does_nothing() {
        let mut stat1 = VonMisesSuffStat::new();
        let stat2 = VonMisesSuffStat::new();
        stat1.observe(&1.0);
        stat1.merge(stat2);
        assert_eq!(stat1.n(), 1);
    }

    #[test]
    fn from_data_empty_vec() {
        let data: Vec<f64> = vec![];
        let stat = VonMisesSuffStat::from_data(&data);
        assert_eq!(stat.n(), 0);
    }
}
