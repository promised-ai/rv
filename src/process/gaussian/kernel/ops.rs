use super::{CovGrad, CovGradError, Kernel, KernelError};
use nalgebra::base::constraint::{SameNumberOfColumns, ShapeConstraint};
use nalgebra::base::storage::Storage;
use nalgebra::{DMatrix, DVector, Dim, Matrix};
use std::f64;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Kernel representing the sum of two other kernels
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct AddKernel<A, B>
where
    A: Kernel,
    B: Kernel,
{
    a: A,
    b: B,
}

impl<A, B, C> std::ops::Mul<C> for AddKernel<A, B>
where
    A: Kernel,
    B: Kernel,
    C: Kernel,
{
    type Output = ProductKernel<Self, C>;

    fn mul(self, rhs: C) -> Self::Output {
        ProductKernel::new(self, rhs)
    }
}

impl<A, B, C> std::ops::Add<C> for AddKernel<A, B>
where
    A: Kernel,
    B: Kernel,
    C: Kernel,
{
    type Output = AddKernel<Self, C>;

    fn add(self, rhs: C) -> Self::Output {
        AddKernel::new(self, rhs)
    }
}

impl<A, B> AddKernel<A, B>
where
    A: Kernel,
    B: Kernel,
{
    /// Construct a new Kernel from two other Kernels
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> Kernel for AddKernel<A, B>
where
    A: Kernel,
    B: Kernel,
{
    fn n_parameters(&self) -> usize {
        self.a.n_parameters() + self.b.n_parameters()
    }

    fn is_stationary(&self) -> bool {
        self.a.is_stationary() && self.b.is_stationary()
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
        let a = self.a.covariance(x1, x2);
        let b = self.b.covariance(x1, x2);
        assert_eq!(a.shape(), b.shape());
        a + b
    }

    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>,
    {
        let a = self.a.diag(x);
        let b = self.b.diag(x);
        a.zip_map(&b, |y1, y2| y1 + y2)
    }

    fn parameters(&self) -> DVector<f64> {
        let a = self.a.parameters();
        let b = self.b.parameters();

        DVector::from_iterator(
            self.a.n_parameters() + self.b.n_parameters(),
            a.into_iter().chain(b.into_iter()).copied(),
        )
    }

    fn reparameterize(&self, params: &[f64]) -> Result<Self, KernelError> {
        let (a_params, b_params) = params.split_at(self.a.n_parameters());

        let a = self.a.reparameterize(a_params)?;
        let b = self.b.reparameterize(b_params)?;

        Ok(Self::new(a, b))
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
        let (cov_a, grad_a) = self.a.covariance_with_gradient(x)?;
        let (cov_b, grad_b) = self.b.covariance_with_gradient(x)?;

        let new_cov = cov_a + cov_b;

        let new_grad = grad_a.concat_cols(&grad_b)?;
        Ok((new_cov, new_grad))
    }
}

/// Kernel representing the product of two other kernels
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct ProductKernel<A, B>
where
    A: Kernel,
    B: Kernel,
{
    a: A,
    b: B,
}

impl<A, B> ProductKernel<A, B>
where
    A: Kernel,
    B: Kernel,
{
    /// Construct a new Kernel from two other Kernels
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B, C> std::ops::Mul<C> for ProductKernel<A, B>
where
    A: Kernel,
    B: Kernel,
    C: Kernel,
{
    type Output = ProductKernel<Self, C>;

    fn mul(self, rhs: C) -> Self::Output {
        ProductKernel::new(self, rhs)
    }
}

impl<A, B, C> std::ops::Add<C> for ProductKernel<A, B>
where
    A: Kernel,
    B: Kernel,
    C: Kernel,
{
    type Output = AddKernel<Self, C>;

    fn add(self, rhs: C) -> Self::Output {
        AddKernel::new(self, rhs)
    }
}

impl<A, B> Kernel for ProductKernel<A, B>
where
    A: Kernel,
    B: Kernel,
{
    fn n_parameters(&self) -> usize {
        self.a.n_parameters() + self.b.n_parameters()
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
        let cov_a = self.a.covariance(x1, x2);
        let cov_b = self.b.covariance(x1, x2);
        cov_a.component_mul(&cov_b)
    }

    fn is_stationary(&self) -> bool {
        self.a.is_stationary() && self.b.is_stationary()
    }

    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>,
    {
        let a = self.a.diag(x);
        let b = self.b.diag(x);
        assert_eq!(a.shape(), b.shape());
        a.zip_map(&b, |y1, y2| y1 * y2)
    }

    fn parameters(&self) -> DVector<f64> {
        let a = self.a.parameters();
        let b = self.b.parameters();
        DVector::from_iterator(
            self.a.n_parameters() + self.b.n_parameters(),
            a.into_iter().chain(b.into_iter()).copied(),
        )
    }

    fn reparameterize(&self, params: &[f64]) -> Result<Self, KernelError> {
        let (a_params, b_params) = params.split_at(self.a.n_parameters());

        let a = self.a.reparameterize(a_params)?;
        let b = self.b.reparameterize(b_params)?;

        Ok(Self::new(a, b))
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
        let (cov_a, grad_a) = self.a.covariance_with_gradient(x)?;
        let (cov_b, grad_b) = self.b.covariance_with_gradient(x)?;

        let new_cov = cov_a.component_mul(&cov_b);
        let new_grad = grad_a
            .component_mul(&cov_b)?
            .concat_cols(&grad_b.component_mul(&cov_a)?)?;

        Ok((new_cov, new_grad))
    }
}

#[cfg(test)]
mod tests {
    use crate::process::gaussian::kernel::{
        ConstantKernel, RBFKernel, WhiteKernel,
    };

    use super::*;

    #[test]
    fn add_kernel() -> Result<(), KernelError> {
        let kernel =
            AddKernel::new(ConstantKernel::new(3.0)?, WhiteKernel::new(2.0)?);
        let x = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);

        let expected_cov = DMatrix::from_row_slice(2, 2, &[5.0, 3.0, 3.0, 5.0]);

        let expected_grad = CovGrad::new_unchecked(&[
            DMatrix::from_row_slice(2, 2, &[3.0, 3.0, 3.0, 3.0]),
            DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]),
        ]);

        let (cov, grad) = kernel.covariance_with_gradient(&x)?;
        assert!(cov.relative_eq(&expected_cov, 1E-7, 1E-7));
        assert!(grad.relative_eq(&expected_grad, 1E-7, 1E-7));
        Ok(())
    }

    #[test]
    fn product_kernel() -> Result<(), KernelError> {
        let var_name = ConstantKernel::new(3.0)?;
        let kernel = var_name * RBFKernel::new(5.0)?;
        let x = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let y = DMatrix::from_row_slice(2, 2, &[5.0, 7.0, 6.0, 8.0]);

        let expected_cov = DMatrix::from_row_slice(
            2,
            2,
            &[
                1.321_294_963_517_997_8,
                0.885_690_500_772_042_5,
                2.313_154_757_410_699,
                1.819_591_979_137_900_3,
            ],
        );
        let cov = kernel.covariance(&x, &y);
        assert!(cov.relative_eq(&expected_cov, 1E-7, 1E-7));

        // Symmetric Cov and Grad
        let expected_cov = DMatrix::from_row_slice(
            2,
            2,
            &[3.0, 2.556_431_366_898_634, 2.556_431_366_898_634, 3.0],
        );

        let expected_grad = CovGrad::new_unchecked(&[
            DMatrix::from_row_slice(
                2,
                2,
                &[3.0, 2.556_431_366_898_634, 2.556_431_366_898_634, 3.0],
            ),
            DMatrix::from_row_slice(
                2,
                2,
                &[0.0, 0.818_058_037_407_562_8, 0.818_058_037_407_562_8, 0.0],
            ),
        ]);

        let (cov, grad) = kernel.covariance_with_gradient(&x)?;
        assert!(cov.relative_eq(&expected_cov, 1E-7, 1E-7));
        assert!(grad.relative_eq(&expected_grad, 1E-7, 1E-7));
        Ok(())
    }
}
