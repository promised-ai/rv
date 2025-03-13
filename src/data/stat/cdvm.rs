#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::traits::SuffStat;

/// Cdvm sufficient statistic.
///
/// Holds the number of observations, their sum, and the sum of their squared
/// values.
#[derive(Debug, Clone, PartialEq, Copy)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct CdvmSuffStat {
    /// Modulus of the circular distribution
    modulus: usize,
    /// Number of observations
    n: usize,
    /// ∑ⱼ cos(2πxⱼ/m)
    sum_cos: f64,
    /// ∑ⱼ sin(2πxⱼ/m)
    sum_sin: f64,

    /// Cached 2π/m
    #[cfg_attr(feature = "serde1", serde(skip))]
    twopi_over_m: f64
}

impl CdvmSuffStat {
    #[inline]
    pub fn new(modulus: usize) -> Self {
        CdvmSuffStat {
            modulus,
            n: 0,
            sum_cos: 0.0,
            sum_sin: 0.0,
            twopi_over_m: 2.0 * std::f64::consts::PI / modulus as f64,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    pub fn from_parts_unchecked(
        modulus: usize,
        n: usize,
        sum_cos: f64,
        sum_sin: f64,
    ) -> Self {
        CdvmSuffStat {
            modulus,
            n,
            sum_cos,
            sum_sin,
            twopi_over_m: 2.0 * std::f64::consts::PI / modulus as f64,
        }
    }

    /// Create a sufficient statistic from a slice of data
    ///
    /// Note that we can't have the usual From trait without const generics
    /// because we need to know the modulus.
    pub fn from_data(modulus: usize, xs: &[usize]) -> Self {
        let mut stat = CdvmSuffStat::new(modulus);
        for x in xs {
            stat.observe(x);
        }
        stat
    }

    /// Get the modulus
    pub fn modulus(&self) -> usize {
        self.modulus
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

impl SuffStat<usize> for CdvmSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &usize) {
        if *x >= self.modulus {
            panic!("x must be less than modulus");
        }
        let angle = self.twopi_over_m * (*x as f64);
        self.sum_cos += angle.cos();
        self.sum_sin += angle.sin();
        self.n += 1;
    }

    fn forget(&mut self, x: &usize) {
        if *x >= self.modulus {
            panic!("x must be less than modulus");
        }
        let angle = self.twopi_over_m * (*x as f64);
        self.sum_cos -= angle.cos();
        self.sum_sin -= angle.sin();
        self.n -= 1;
    }

    fn merge(&mut self, other: Self) {
        assert_eq!(self.modulus, other.modulus);

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
        let stat = CdvmSuffStat::new(4);
        assert_eq!(stat.n(), 0);
    }

    #[test]
    fn observe_increments_n() {
        let mut stat = CdvmSuffStat::new(4);
        stat.observe(&1);
        assert_eq!(stat.n(), 1);
    }

    #[test]
    fn forget_decrements_n() {
        let mut stat = CdvmSuffStat::new(4);
        stat.observe(&1);
        stat.forget(&1);
        assert_eq!(stat.n(), 0);
    }

    #[test]
    fn merge_adds_n() {
        let mut stat1 = CdvmSuffStat::new(4);
        let mut stat2 = CdvmSuffStat::new(4);
        stat1.observe(&1);
        stat2.observe(&2);
        stat1.merge(stat2);
        assert_eq!(stat1.n(), 2);
    }

    #[test]
    fn merge_empty_stat_does_nothing() {
        let mut stat1 = CdvmSuffStat::new(4);
        let stat2 = CdvmSuffStat::new(4);
        stat1.observe(&1);
        stat1.merge(stat2);
        assert_eq!(stat1.n(), 1);
    }

    #[test]
    #[should_panic(expected = "x must be less than modulus")]
    fn observe_panics_if_x_too_large() {
        let mut stat = CdvmSuffStat::new(4);
        stat.observe(&4);
    }

    #[test]
    #[should_panic(expected = "x must be less than modulus")]
    fn forget_panics_if_x_too_large() {
        let mut stat = CdvmSuffStat::new(4);
        stat.forget(&4);
    }

    #[test]
    #[should_panic]
    fn merge_panics_if_modulus_mismatch() {
        let mut stat1 = CdvmSuffStat::new(4);
        let stat2 = CdvmSuffStat::new(3);
        stat1.merge(stat2);
    }

    #[test]
    fn from_data_empty_vec() {
        let data: Vec<usize> = vec![];
        let stat = CdvmSuffStat::from_data(4, &data);
        assert_eq!(stat.modulus, 4);
        assert_eq!(stat.n(), 0);
    }

    #[test]
    fn from_data_sets_correct_modulus() {
        let data = vec![0, 1, 2, 3];
        let stat = CdvmSuffStat::from_data(4, &data);
        assert_eq!(stat.modulus, 4);
        assert_eq!(stat.n(), 4);
    }
}
