#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::traits::SuffStat;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct MvGaussianSuffStat {
    n: usize,
    sum_x: DVector<f64>,
    sum_x_sq: DMatrix<f64>,
}

impl MvGaussianSuffStat {
    #[inline]
    #[must_use]
    pub fn new(dims: usize) -> Self {
        MvGaussianSuffStat {
            n: 0,
            sum_x: DVector::zeros(dims),
            sum_x_sq: DMatrix::zeros(dims, dims),
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    ///
    /// # Example
    /// ```rust
    /// use rv::data::MvGaussianSuffStat;
    /// use rv::prelude::SuffStat;
    /// use nalgebra::{DVector, dvector, DMatrix};
    ///
    /// let data: Vec<DVector<f64>> = vec![
    ///   dvector![0.1, 0.2],
    ///   dvector![0.3, 0.4],
    /// ];
    ///
    /// let mut stat_b = MvGaussianSuffStat::new(2);
    /// stat_b.observe_many(&data);
    ///
    /// let n = data.len();
    /// let sum_x: DVector<f64> = data.iter().sum();
    /// let sum_x_sq: DMatrix<f64> = data.iter().map(|x| x * x.transpose()).sum();
    ///
    /// let stat_a = MvGaussianSuffStat::from_parts_unchecked(n, sum_x, sum_x_sq);
    ///
    /// assert_eq!(stat_a.n(), stat_b.n());
    /// assert::close(stat_a.sum_x()[0], stat_b.sum_x()[0], 1e-10);
    /// assert::close(stat_a.sum_x()[1], stat_b.sum_x()[1], 1e-10);
    /// assert::close(stat_a.sum_x_sq()[(0, 0)], stat_b.sum_x_sq()[(0, 0)], 1e-10);
    /// assert::close(stat_a.sum_x_sq()[(0, 1)], stat_b.sum_x_sq()[(0, 1)], 1e-10);
    /// assert::close(stat_a.sum_x_sq()[(1, 0)], stat_b.sum_x_sq()[(1, 0)], 1e-10);
    /// assert::close(stat_a.sum_x_sq()[(1, 1)], stat_b.sum_x_sq()[(1, 1)], 1e-10);
    ///
    /// ```
    #[inline]
    #[must_use]
    pub fn from_parts_unchecked(
        n: usize,
        sum_x: DVector<f64>,
        sum_x_sq: DMatrix<f64>,
    ) -> Self {
        MvGaussianSuffStat { n, sum_x, sum_x_sq }
    }

    /// Get the number of observations
    #[inline]
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sum of observations
    #[inline]
    #[must_use]
    pub fn sum_x(&self) -> &DVector<f64> {
        &self.sum_x
    }

    /// Get the sum of X^2
    #[inline]
    #[must_use]
    pub fn sum_x_sq(&self) -> &DMatrix<f64> {
        &self.sum_x_sq
    }
}

impl SuffStat<DVector<f64>> for MvGaussianSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &DVector<f64>) {
        self.n += 1;
        if self.n == 1 {
            self.sum_x = x.clone();
            self.sum_x_sq = x * x.transpose();
        } else {
            self.sum_x += x;
            self.sum_x_sq += x * x.transpose();
        }
    }

    fn forget(&mut self, x: &DVector<f64>) {
        self.n -= 1;
        if self.n > 0 {
            self.sum_x -= x;
            self.sum_x_sq -= x * x.transpose();
        } else {
            let dims = self.sum_x.len();
            self.sum_x = DVector::zeros(dims);
            self.sum_x_sq = DMatrix::zeros(dims, dims);
        }
    }

    fn merge(&mut self, other: Self) {
        self.n += other.n;
        self.sum_x += other.sum_x;
        self.sum_x_sq += other.sum_x_sq;
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dvector;

    use super::*;

    #[test]
    fn observe_forget() {
        let mut stat = MvGaussianSuffStat::new(2);

        stat.observe(&dvector![0.1, 0.2]);
        stat.observe(&dvector![0.3, 0.4]);

        dbg!(&stat.sum_x_sq);

        assert_eq!(stat.n(), 2);
        assert::close(stat.sum_x()[0], 0.4, 1e-10);
        assert::close(stat.sum_x()[1], 0.6, 1e-10);
        assert::close(stat.sum_x_sq()[(0, 0)], 0.1, 1e-10);
        assert::close(stat.sum_x_sq()[(0, 1)], 0.14, 1e-10);
        assert::close(stat.sum_x_sq()[(1, 0)], 0.14, 1e-10);
        assert::close(stat.sum_x_sq()[(1, 1)], 0.2, 1e-10);

        stat.forget(&dvector![0.1, 0.2]);
        assert_eq!(stat.n(), 1);
        assert::close(stat.sum_x()[0], 0.3, 1e-10);
        assert::close(stat.sum_x()[1], 0.4, 1e-10);
        assert::close(stat.sum_x_sq()[(0, 0)], 0.09, 1e-10);
        assert::close(stat.sum_x_sq()[(0, 1)], 0.12, 1e-10);
        assert::close(stat.sum_x_sq()[(1, 0)], 0.12, 1e-10);
        assert::close(stat.sum_x_sq()[(1, 1)], 0.16, 1e-10);

        stat.forget(&dvector![0.3, 0.4]);
        assert_eq!(stat.n(), 0);
        assert::close(stat.sum_x()[0], 0.0, 1e-10);
        assert::close(stat.sum_x()[1], 0.0, 1e-10);
        assert::close(stat.sum_x_sq()[(0, 0)], 0.0, 1e-10);
        assert::close(stat.sum_x_sq()[(0, 1)], 0.0, 1e-10);
        assert::close(stat.sum_x_sq()[(1, 0)], 0.0, 1e-10);
        assert::close(stat.sum_x_sq()[(1, 1)], 0.0, 1e-10);
    }

    #[test]
    fn merge() {
        let mut a = MvGaussianSuffStat::new(2);
        let mut b = MvGaussianSuffStat::new(2);
        let mut c = MvGaussianSuffStat::new(2);

        a.observe(&dvector![0.1, 0.2]);
        b.observe(&dvector![0.3, 0.4]);

        c.observe(&dvector![0.1, 0.2]);
        c.observe(&dvector![0.3, 0.4]);

        <MvGaussianSuffStat as SuffStat<DVector<f64>>>::merge(&mut a, b);

        assert_eq!(a.n(), c.n());
        assert::close(a.sum_x()[0], c.sum_x()[0], 1e-10);
        assert::close(a.sum_x()[1], c.sum_x()[1], 1e-10);
        assert::close(a.sum_x_sq()[(0, 0)], c.sum_x_sq()[(0, 0)], 1e-10);
        assert::close(a.sum_x_sq()[(0, 1)], c.sum_x_sq()[(0, 1)], 1e-10);
        assert::close(a.sum_x_sq()[(1, 0)], c.sum_x_sq()[(1, 0)], 1e-10);
        assert::close(a.sum_x_sq()[(1, 1)], c.sum_x_sq()[(1, 1)], 1e-10);
    }
}
