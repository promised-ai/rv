use super::{CovGrad, Kernel};
use nalgebra::base::constraint::{SameNumberOfColumns, ShapeConstraint};
use nalgebra::base::storage::Storage;
use nalgebra::{DMatrix, DVector, Dim, Matrix};
use std::f64;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ConstantKernel {
    value: f64,
    lower_bound: f64,
    upper_bound: f64,
}

impl ConstantKernel {
    pub fn new(value: f64) -> Self {
        Self {
            value,
            lower_bound: 1E-5,
            upper_bound: 1E5,
        }
    }

    pub fn with_bounds(self, lower_bound: f64, upper_bound: f64) -> Self {
        Self {
            lower_bound,
            upper_bound,
            ..self
        }
    }
}

impl std::convert::TryFrom<f64> for ConstantKernel {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if value < 0.0 {
            Err("Constant Kernel values must not be negative")
        } else {
            Ok(ConstantKernel::new(value))
        }
    }
}

impl Kernel for ConstantKernel {
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
        DMatrix::from_element(x1.nrows(), x2.nrows(), self.value)
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
        DVector::from_element(x.nrows(), self.value)
    }

    fn parameters(&self) -> Vec<f64> {
        vec![self.value.ln()]
    }

    fn parameter_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![self.lower_bound], vec![self.upper_bound])
    }

    fn from_parameters(param_vec: &[f64]) -> Self {
        Self::new(param_vec[0].exp())
    }

    fn consume_parameters(param_vec: &[f64]) -> (Self, &[f64]) {
        assert!(param_vec.len() > 0, "ConstantKernel requires one parameter");
        let (cur, next) = param_vec.split_at(1);
        let ck = Self::from_parameters(cur);
        (ck, next)
    }

    fn covariance_with_gradient<R, C, S>(
        &self,
        x: &Matrix<f64, R, C, S>,
    ) -> (DMatrix<f64>, CovGrad)
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>,
    {
        let cov = self.covariance(x, x);
        let grad = CovGrad::new(&[DMatrix::from_element(
            x.nrows(),
            x.nrows(),
            self.value,
        )]);
        (cov, grad)
    }
}

#[cfg(test)]
mod tests {
    use crate::test::relative_eq;

    use super::*;

    #[test]
    fn constant_kernel() {
        let kernel = ConstantKernel::new(3.0);
        assert::close(kernel.parameters()[0], 3.0_f64.ln(), 1E-10);
        assert!(relative_eq(
            kernel.parameters(),
            vec![3.0_f64.ln()],
            1E-8,
            1E-8,
        ));

        let x = DMatrix::from_column_slice(2, 2, &[1.0, 3.0, 2.0, 4.0]);
        let y = DMatrix::from_column_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);

        let (cov, grad) = kernel.covariance_with_gradient(&x);

        let expected_cov = DMatrix::from_row_slice(2, 2, &[3.0, 3.0, 3.0, 3.0]);

        let expected_grad =
            CovGrad::from_row_slices(2, 1, &[3.0, 3.0, 3.0, 3.0]);

        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));

        let cov = kernel.covariance(&x, &y);
        let expected_cov = DMatrix::from_row_slice(2, 2, &[3.0, 3.0, 3.0, 3.0]);

        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
    }
}
