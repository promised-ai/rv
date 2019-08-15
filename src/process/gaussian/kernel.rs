//! Gaussian Processes
use nalgebra::base::constraint::{SameNumberOfColumns, ShapeConstraint};
use nalgebra::base::storage::Storage;
use nalgebra::base::EuclideanNorm;
use nalgebra::base::Norm;
use nalgebra::{DMatrix, DVector, Dim, Matrix};
use std::f64;

/// Kernel Function
pub trait Kernel {
    /// Returns the covariance matrix for two equal sized vectors
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
        ShapeConstraint: SameNumberOfColumns<C1, C2>;
    /// Reports if the given kernel function is stationary.
    fn is_stationary(&self) -> bool;
    /// Returns the diagnal of the kernel(x, x)
    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>;
    /// Return the corresponding parameter vector
    fn parameters(&self) -> DVector<f64>;
    /// Create a new kernel of the given type from the provided parameters.
    fn from_parameters(param_vec: DVector<f64>) -> Self;
}
/*
/// Kernel representing the sum of two other kernels
pub struct AddKernel<A, B>
    where
        A: Kernel,
        B: Kernel,
{
    a: A,
    b: B,
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
    fn covariance(&self, x1: &DMatrix<f64>, x2: &DMatrxi<f64>) -> DMatrix<f64> {
        self.a.covariance(x1, x2) + self.b.covariance(x1, x2)
    }

    fn is_stationary(&self) -> bool {
        self.a.is_stationary() && self.b.is_stationary()
    }

    fn diag(&self, x: &DMatrix<f64>) -> DVector<f64> {
        let a = self.a.diag(x);
        let b = self.b.diag(x);
        &a + &b
    }

    fn parameters(&self) -> DVector<f64> {
        let a = self.a.parameters();
        let b = self.b.parameters();

        DVector::from_columns(&[a, b])
    }
}


/// Kernel representing the product of two other kernels
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

impl<A, B> Kernel for ProductKernel<A, B>
    where
        A: Kernel,
        B: Kernel,
{
    fn covariance(&self, x1: &DMatrix<f64>, x2: &DMatrix<f64>) -> DMatrix<f64> {
        self.a.covariance(x1, x2).component_mul(&self.b.covariance(x1, x2))
    }

    fn is_stationary(&self) -> bool {
        self.a.is_stationary() && self.b.is_stationary()
    }

    fn diag(&self, x: &DMatrix<f64>) -> DVector<f64> {
        let a = self.a.diag(x);
        let b = self.b.diag(x);
        a.zip_map(&b, |y1, y2| y1 * y2)
    }

    fn parameters(&self) -> DVector<f64> {
        let a = self.a.parameters();
        let b = self.b.parameters();

        DVector::from_columns(&[a, b])
    }
}

*/

/// Radial-basis function (RBF) kernel
/// The distance metric here is L2 (Euclidean).
///
/// ```math
///     K(\mathbf{x}, \mathbf{x'}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x'}\|^2}{2\sigma^2}\right)
/// ```
///
/// # Parameters
/// * `sigma_f` - Sigma on the function output.
/// * `l` - Length scale.
///
pub struct RBFKernel {
    sigma_f: f64,
    l: f64,
}

impl RBFKernel {
    pub fn new(sigma_f: f64, l: f64) -> Self {
        Self { sigma_f, l }
    }
}

impl Kernel for RBFKernel {
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
        let m = x1.nrows();
        let n = x2.nrows();

        let mut dm: DMatrix<f64> = DMatrix::zeros(m, n);

        let metric = EuclideanNorm {};
        for i in 0..m {
            for j in 0..n {
                let d = metric.metric_distance(&x1.row(i), &x2.row(j)) / self.l;
                dm[(i, j)] = d * d;
            }
        }

        let m = self.sigma_f * self.sigma_f;
        dm.map(|e| m * (-0.5 * e).exp())
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
        DVector::repeat(x.nrows(), 1.0)
    }

    fn parameters(&self) -> DVector<f64> {
        DVector::from_column_slice(&[self.sigma_f, self.l])
    }

    fn from_parameters(param_vec: DVector<f64>) -> Self {
        assert_eq!(param_vec.len(), 2);
        Self::new(param_vec[0], param_vec[1])
    }
}

/*

/// Rational Quadratic Kernel
///
/// # Parameters
/// `scale` -- Length scale
/// `mixture` -- Mixture Scale
pub struct RationalQuadratic {
    sigma_f: f64,
    scale: f64,
    mixture: f64,
}

impl Kernel for RationalQuadratic
{
    fn covariance(&self, x1: &DMatrix<f64>, x2: &DMatrix<f64>) -> DMatrix<f64> {
        assert!(x1.len() == x2.len());
        let diff = x1 - x2;
        let diff_t = diff.clone().transpose();
        let l = (diff * &diff_t) / (2.0 * self.scale * self.scale * self.mixture);
        let m = self.sigma_f * self.sigma_f;
        l.map(|e| m * (1.0 + e.powf(-self.mixture)))
    }

    fn is_stationary(&self) -> bool {
        true
    }

    fn diag(&self, x: &DMatrix<f64>) -> DVector<f64> {
        DVector::repeat(x.len(), 1.0)
    }

    fn parameters(&self) -> DVector<f64> {
        DVector::from_column_slice(&[self.sigma_f, self.scale, self.mixture])
    }
}

/// Exp Sine^2 Kernel
/// k(x_i, x_j) = exp(-2 (sin(pi / periodicity * d(x_i, x_j)) / length_scale) ^ 2)
pub struct ExpSineSquaredKernel {
    sigma_f: f64,
    periodicity: f64,
    length_scale: f64,
}

impl Kernel for ExpSineSquaredKernel
{
    fn covariance(&self, x1: &DMatrix<f64>, x2: &DMatrix<f64>) -> DMatrix<f64> {
        assert!(x1.len() == x2.len());
        let diff = x1 - x2;
        let diff_t = diff.clone().transpose();
        let sinarg = f64::consts::PI * (diff * &diff_t) / self.periodicity;
        let m = self.sigma_f * self.sigma_f;
        sinarg.map(|e| m * (2.0 * (e.sin().powi(2)) / (self.length_scale * self.length_scale)).exp())
    }

    fn is_stationary(&self) -> bool {
        true
    }

    fn diag(&self, x: &DMatrix<f64>) -> DVector<f64> {
        DVector::repeat(x.len(), 1.0)
    }

    fn parameters(&self) -> DVector<f64> {
        DVector::from_column_slice(&[self.sigma_f, self.periodicity, self.length_scale])
    }
}

*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbf_1d() {
        let xs = DVector::from_column_slice(&[0.0, 1.0, 2.0, 3.0]);
        let kernel = RBFKernel::new(1.0, 1.0);

        let cov = kernel.covariance(&xs, &xs);
        let expected_cov = DMatrix::from_column_slice(
            4,
            4,
            &[
                1., 0.60653066, 0.13533528, 0.011109, 0.60653066, 1.,
                0.60653066, 0.13533528, 0.13533528, 0.60653066, 1., 0.60653066,
                0.011109, 0.13533528, 0.60653066, 1.,
            ],
        );

        assert!(expected_cov.relative_eq(&cov, 1E-8, 1E-8));
        let expected_diag = DVector::from_column_slice(&[1., 1., 1., 1.]);
        assert_eq!(kernel.diag(&xs), expected_diag);
        assert!(kernel.is_stationary());
    }

    #[test]
    fn rbf_2d() {
        use nalgebra::Matrix4x2;

        let kernel = RBFKernel::new(1.0, 1.0);
        let xs =
            Matrix4x2::from_column_slice(&[0., 1., 2., 3., 4., 5., 6., 7.]);
        let expected_cov = DMatrix::from_column_slice(
            4,
            4,
            &[
                1.00000000e+00,
                3.67879441e-01,
                1.83156389e-02,
                1.23409804e-04,
                3.67879441e-01,
                1.00000000e+00,
                3.67879441e-01,
                1.83156389e-02,
                1.83156389e-02,
                3.67879441e-01,
                1.00000000e+00,
                3.67879441e-01,
                1.23409804e-04,
                1.83156389e-02,
                3.67879441e-01,
                1.00000000e+00,
            ],
        );

        let cov = kernel.covariance(&xs, &xs);
        assert!(expected_cov.relative_eq(&cov, 1E-8, 1E-8));
    }

    #[test]
    fn rbf_different_sizes() {
        use nalgebra::Matrix5x1;
        let kernel = RBFKernel::new(1.0, 1.0);

        let x1 = Matrix5x1::from_column_slice(&[-4., -3., -2., -1., 1.]);
        let x2 = DMatrix::from_column_slice(
            10,
            1,
            &[-5., -4., -3., -2., -1., 0., 1., 2., 3., 4.],
        );

        let cov = kernel.covariance(&x1, &x2);
        let expected_cov = DMatrix::from_row_slice(
            5,
            10,
            &[
                6.06530660e-01,
                1.00000000e+00,
                6.06530660e-01,
                1.35335283e-01,
                1.11089965e-02,
                3.35462628e-04,
                3.72665317e-06,
                1.52299797e-08,
                2.28973485e-11,
                1.26641655e-14,
                1.35335283e-01,
                6.06530660e-01,
                1.00000000e+00,
                6.06530660e-01,
                1.35335283e-01,
                1.11089965e-02,
                3.35462628e-04,
                3.72665317e-06,
                1.52299797e-08,
                2.28973485e-11,
                1.11089965e-02,
                1.35335283e-01,
                6.06530660e-01,
                1.00000000e+00,
                6.06530660e-01,
                1.35335283e-01,
                1.11089965e-02,
                3.35462628e-04,
                3.72665317e-06,
                1.52299797e-08,
                3.35462628e-04,
                1.11089965e-02,
                1.35335283e-01,
                6.06530660e-01,
                1.00000000e+00,
                6.06530660e-01,
                1.35335283e-01,
                1.11089965e-02,
                3.35462628e-04,
                3.72665317e-06,
                1.52299797e-08,
                3.72665317e-06,
                3.35462628e-04,
                1.11089965e-02,
                1.35335283e-01,
                6.06530660e-01,
                1.00000000e+00,
                6.06530660e-01,
                1.35335283e-01,
                1.11089965e-02,
            ],
        );
        assert!(cov.relative_eq(&expected_cov, 1E-5, 1E-5));
    }
}
