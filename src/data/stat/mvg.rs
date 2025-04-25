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
    #[must_use] pub fn new(dims: usize) -> Self {
        MvGaussianSuffStat {
            n: 0,
            sum_x: DVector::zeros(dims),
            sum_x_sq: DMatrix::zeros(dims, dims),
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    #[must_use] pub fn from_parts_unchecked(
        n: usize,
        sum_x: DVector<f64>,
        sum_x_sq: DMatrix<f64>,
    ) -> Self {
        MvGaussianSuffStat { n, sum_x, sum_x_sq }
    }

    /// Get the number of observations
    #[inline]
    #[must_use] pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sum of observations
    #[inline]
    #[must_use] pub fn sum_x(&self) -> &DVector<f64> {
        &self.sum_x
    }

    /// Get the sum of X^2
    #[inline]
    #[must_use] pub fn sum_x_sq(&self) -> &DMatrix<f64> {
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
