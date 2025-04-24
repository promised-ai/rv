use super::{CovGrad, CovGradError, Kernel, KernelError};
use nalgebra::base::constraint::{SameNumberOfColumns, ShapeConstraint};
use nalgebra::base::storage::Storage;
use nalgebra::{dvector, DMatrix, DVector, Dim, Matrix};
use std::f64;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct ConstantKernel {
    scale: f64,
}

impl ConstantKernel {
    /// Create a new kernel with the given constant value
    pub fn new(value: f64) -> Result<Self, KernelError> {
        if value <= 0.0 {
            Err(KernelError::ParameterOutOfBounds {
                name: "value".to_string(),
                given: value,
                bounds: (0.0, f64::INFINITY),
            })
        } else {
            Ok(Self { scale: value })
        }
    }

    /// Create a new constant function kernel without checking the parameters
    #[must_use]
    pub fn new_unchecked(scale: f64) -> Self {
        Self { scale }
    }
}

impl Default for ConstantKernel {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

impl std::convert::TryFrom<f64> for ConstantKernel {
    type Error = KernelError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl Kernel for ConstantKernel {
    fn n_parameters(&self) -> usize {
        1
    }

    fn covariance<R1, R2, C1, C2, S1, S2>(
        &self,
        x1: &Matrix<f64, R1, C1, S1>,
        x2: &Matrix<f64, R2, C2, S2>,
    ) -> DMatrix<f64>
    where
        R1: Dim,
        R2: Dim,
        C1: Dim,
        C2: Dim,
        S1: Storage<f64, R1, C1>,
        S2: Storage<f64, R2, C2>,
        ShapeConstraint: SameNumberOfColumns<C1, C2>,
    {
        DMatrix::from_element(x1.nrows(), x2.nrows(), self.scale)
    }

    fn is_stationary(&self) -> bool {
        true
    }

    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>,
    {
        DVector::from_element(x.nrows(), self.scale)
    }

    fn parameters(&self) -> DVector<f64> {
        dvector![self.scale.ln()]
    }

    fn reparameterize(&self, param_vec: &[f64]) -> Result<Self, KernelError> {
        match param_vec {
            [] => Err(KernelError::MissingParameters(1)),
            [value] => Self::new(value.exp()),
            _ => Err(KernelError::ExtraneousParameters(param_vec.len() - 1)),
        }
    }

    fn covariance_with_gradient<R, C, S>(
        &self,
        x: &Matrix<f64, R, C, S>,
    ) -> Result<(DMatrix<f64>, CovGrad), CovGradError>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>,
    {
        let cov = self.covariance(x, x);
        let grad = CovGrad::new_unchecked(&[DMatrix::from_element(
            x.nrows(),
            x.nrows(),
            self.scale,
        )]);
        Ok((cov, grad))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_kernel() {
        let kernel = ConstantKernel::new(3.0).unwrap();
        assert::close(kernel.parameters()[0], 3.0_f64.ln(), 1E-10);
        assert!(kernel.parameters().relative_eq(
            &dvector![3.0_f64.ln()],
            1E-8,
            1E-8,
        ));

        let x = DMatrix::from_column_slice(2, 2, &[1.0, 3.0, 2.0, 4.0]);
        let y = DMatrix::from_column_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);

        let (cov, grad) = kernel.covariance_with_gradient(&x).unwrap();

        let expected_cov = DMatrix::from_row_slice(2, 2, &[3.0, 3.0, 3.0, 3.0]);

        let expected_grad =
            CovGrad::from_row_slices(2, 1, &[3.0, 3.0, 3.0, 3.0]).unwrap();

        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));

        let cov = kernel.covariance(&x, &y);
        let expected_cov = DMatrix::from_row_slice(2, 2, &[3.0, 3.0, 3.0, 3.0]);

        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
    }
}
