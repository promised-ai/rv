#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use nalgebra::DMatrix;
use std::fmt;
use std::ops::{Index, IndexMut};

/// Representation of the gradient of the covariance matrix
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct CovGrad {
    slices: Vec<DMatrix<f64>>,
}

impl fmt::Display for CovGrad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.slices.iter().try_for_each(|s| write!(f, "{s}"))
    }
}

impl CovGrad {
    /// Create a new cov-grad with given slices
    pub fn new(slices: &[DMatrix<f64>]) -> Result<Self, CovGradError> {
        if slices.is_empty() {
            return Err(CovGradError::Empty);
        }

        let shapes: Vec<(usize, usize)> =
            slices.iter().map(nalgebra::Matrix::shape).collect();
        if shapes
            .iter()
            .zip(shapes.iter().skip(1))
            .all(|(a, b)| a == b)
        {
            Ok(Self {
                slices: slices.to_vec(),
            })
        } else {
            Err(CovGradError::ShapeMismatch(shapes))
        }
    }

    /// Create a new unchecked `CovGrad`
    #[must_use]
    pub fn new_unchecked(slices: &[DMatrix<f64>]) -> Self {
        Self {
            slices: slices.to_vec(),
        }
    }

    /// Component wise multiplication
    pub fn component_mul(
        &self,
        other: &DMatrix<f64>,
    ) -> Result<Self, CovGradError> {
        if other.shape() == self.slices[0].shape() {
            let new_slices =
                self.slices.iter().map(|s| s.component_mul(other)).collect();
            Ok(Self { slices: new_slices })
        } else {
            Err(CovGradError::ShapeMismatch(vec![
                self.slices[0].shape(),
                other.shape(),
            ]))
        }
    }

    /// Left multiplication by another matrix
    pub fn left_mul(&self, other: &DMatrix<f64>) -> Result<Self, CovGradError> {
        if other.shape() == self.slices[0].shape() {
            let new_slices = self.slices.iter().map(|s| other * s).collect();
            Ok(Self { slices: new_slices })
        } else {
            Err(CovGradError::ShapeMismatch(vec![
                self.slices[0].shape(),
                other.shape(),
            ]))
        }
    }

    /// Right multiplication by another matrix
    pub fn right_mul(
        &self,
        other: &DMatrix<f64>,
    ) -> Result<Self, CovGradError> {
        if other.shape() == self.slices[0].shape() {
            let new_slices = self.slices.iter().map(|s| s * other).collect();
            Ok(Self { slices: new_slices })
        } else {
            Err(CovGradError::ShapeMismatch(vec![
                self.slices[0].shape(),
                other.shape(),
            ]))
        }
    }

    /// Check if this is relatively eq to another matrix
    #[must_use]
    pub fn relative_eq(&self, other: &CovGrad, rel: f64, abs: f64) -> bool {
        assert!(
            self.slices.len() == other.slices.len(),
            "Cannot compare dissimilarly shaped CovMats"
        );
        self.slices
            .iter()
            .zip(other.slices.iter())
            .all(|(a, b)| a.relative_eq(b, rel, abs))
    }

    /// Concatenate columns from another matrix
    pub fn concat_cols(&self, other: &Self) -> Result<Self, CovGradError> {
        if other.slices[0].shape() == self.slices[0].shape() {
            let slices = [self.slices.clone(), other.slices.clone()].concat();
            Ok(Self { slices })
        } else {
            Err(CovGradError::ShapeMismatch(vec![
                self.slices[0].shape(),
                other.slices[0].shape(),
            ]))
        }
    }

    /// Create a new cov-grad with all zeros
    #[must_use]
    pub fn zeros(n: usize, m: usize) -> Self {
        Self {
            slices: (0..m).map(|_| DMatrix::zeros(n, n)).collect(),
        }
    }

    /// Create a new `CovMat` from a sequence of column slices
    pub fn from_column_slices(
        n: usize,
        m: usize,
        slice: &[f64],
    ) -> Result<Self, CovGradError> {
        if n * n * m == slice.len() {
            let mut slices = Vec::with_capacity(m);

            for k in 0..m {
                let start = n * n * k;
                let end = start + n * n;
                slices.push(DMatrix::from_column_slice(
                    n,
                    n,
                    &slice[start..end],
                ));
            }

            Ok(Self { slices })
        } else {
            Err(CovGradError::ImproperSize(n * n * m, slice.len()))
        }
    }

    /// Create a new `CovMat` from a sequence of row slices
    pub fn from_row_slices(
        n: usize,
        m: usize,
        slice: &[f64],
    ) -> Result<Self, CovGradError> {
        if n * n * m == slice.len() {
            let mut slices = Vec::with_capacity(m);

            for k in 0..m {
                let start = n * n * k;
                let end = start + n * n;
                slices.push(DMatrix::from_row_slice(n, n, &slice[start..end]));
            }

            Ok(Self { slices })
        } else {
            Err(CovGradError::ImproperSize(n * n * m, slice.len()))
        }
    }
}

impl Index<usize> for CovGrad {
    type Output = DMatrix<f64>;

    fn index(&self, k: usize) -> &Self::Output {
        &self.slices[k]
    }
}

impl IndexMut<usize> for CovGrad {
    fn index_mut(&mut self, k: usize) -> &mut Self::Output {
        &mut self.slices[k]
    }
}

impl Index<(usize, usize, usize)> for CovGrad {
    type Output = f64;

    fn index(&self, (i, j, k): (usize, usize, usize)) -> &Self::Output {
        &self.slices[k][(i, j)]
    }
}

impl IndexMut<(usize, usize, usize)> for CovGrad {
    fn index_mut(
        &mut self,
        (i, j, k): (usize, usize, usize),
    ) -> &mut Self::Output {
        &mut self.slices[k][(i, j)]
    }
}

/// Error from constructing a `CovGrad`
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum CovGradError {
    /// The shapes of the slices do not match
    ShapeMismatch(Vec<(usize, usize)>),
    /// A `CovGrad` cannot be empty
    Empty,
    /// Improper number of points to construct a `CovGrad`
    ImproperSize(usize, usize),
}

impl std::error::Error for CovGradError {}

impl std::fmt::Display for CovGradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CovGradError::ShapeMismatch(shapes) => writeln!(f, "Cannot create Covariance Gradient: Shape Mismatch: Shapes {shapes:?}"),
            CovGradError::Empty => writeln!(f, "Cannot create an empty CovGrad"),
            CovGradError::ImproperSize(expected, given) => writeln!(f, "Cannot create Covariance Gradient with given shapes. Given: {given}, Expected: {expected}"),
        }
    }
}
