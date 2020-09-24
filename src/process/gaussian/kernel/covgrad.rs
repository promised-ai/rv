#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use nalgebra::DMatrix;
use std::fmt;
use std::ops::{Index, IndexMut};

/// Representation of the gradient of the covariance matrix
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct CovGrad {
    slices: Vec<DMatrix<f64>>,
}

impl fmt::Display for CovGrad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.slices.iter().map(|s| write!(f, "{}", s)).collect()
    }
}

impl CovGrad {
    /// Create a new cov-grad with given slices
    pub fn new(slices: &[DMatrix<f64>]) -> Self {
        Self {
            slices: slices.to_vec(),
        }
    }

    /// Component wise multiplication
    pub fn component_mul(&self, other: &DMatrix<f64>) -> Self {
        let new_slices =
            self.slices.iter().map(|s| s.component_mul(other)).collect();
        Self { slices: new_slices }
    }

    /// Left multiplication by another matrix
    pub fn left_mul(&self, other: &DMatrix<f64>) -> Self {
        let new_slices = self.slices.iter().map(|s| other * s).collect();
        Self { slices: new_slices }
    }

    /// Right multiplication by another matrix
    pub fn right_mul(&self, other: &DMatrix<f64>) -> Self {
        let new_slices = self.slices.iter().map(|s| s * other).collect();
        Self { slices: new_slices }
    }

    /// Check if this is relatively eq to another matrix
    pub fn relative_eq(&self, other: &CovGrad, rel: f64, abs: f64) -> bool {
        assert!(
            self.slices.len() == other.slices.len(),
            "Cannot compare dissimilarly shaped CovMats"
        );
        self.slices
            .iter()
            .zip(other.slices.iter())
            .map(|(a, b)| a.relative_eq(b, rel, abs))
            .all(|x| x)
    }

    /// Concatenate columns from another matrix
    pub fn concat_cols(&self, other: &Self) -> Self {
        let slices = [self.slices.clone(), other.slices.clone()].concat();
        Self { slices }
    }

    /// Create a new cov-grad with all zeros
    pub fn zeros(n: usize, m: usize) -> Self {
        Self {
            slices: (0..m).map(|_| DMatrix::zeros(n, n)).collect(),
        }
    }

    /// Create a new CovMat from a sequence of column slices
    pub fn from_column_slices(n: usize, m: usize, slice: &[f64]) -> Self {
        assert_eq!(
            n * n * m,
            slice.len(),
            "An incorrect number of points were given"
        );
        let mut slices = Vec::with_capacity(m);

        for k in 0..m {
            let start = n * n * k;
            let end = start + n * n;
            slices.push(DMatrix::from_column_slice(n, n, &slice[start..end]));
        }

        Self { slices }
    }

    /// Create a new CovMat from a sequence of row slices
    pub fn from_row_slices(n: usize, m: usize, slice: &[f64]) -> Self {
        assert_eq!(
            n * n * m,
            slice.len(),
            "An incorrect number of points were given"
        );
        let mut slices = Vec::with_capacity(m);

        for k in 0..m {
            let start = n * n * k;
            let end = start + n * n;
            slices.push(DMatrix::from_row_slice(n, n, &slice[start..end]));
        }

        Self { slices }
    }
}

impl Index<usize> for CovGrad {
    type Output = DMatrix<f64>;

    fn index(&self, k: usize) -> &Self::Output {
        assert!(
            k < self.slices.len(),
            "The requested value was outside of available values"
        );
        &self.slices[k]
    }
}

impl IndexMut<usize> for CovGrad {
    fn index_mut(&mut self, k: usize) -> &mut Self::Output {
        assert!(
            k < self.slices.len(),
            "The requested value was outside of available values"
        );
        &mut self.slices[k]
    }
}

impl Index<(usize, usize, usize)> for CovGrad {
    type Output = f64;

    fn index(&self, (i, j, k): (usize, usize, usize)) -> &Self::Output {
        assert!(
            k < self.slices.len(),
            "The requested value was outside of available values"
        );
        &self.slices[k][(i, j)]
    }
}

impl IndexMut<(usize, usize, usize)> for CovGrad {
    fn index_mut(
        &mut self,
        (i, j, k): (usize, usize, usize),
    ) -> &mut Self::Output {
        assert!(
            k < self.slices.len(),
            "The requested value was outside of available values"
        );
        &mut self.slices[k][(i, j)]
    }
}
