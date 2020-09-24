use super::{e2_norm, CovGrad, Kernel};
use nalgebra::base::constraint::{SameNumberOfColumns, ShapeConstraint};
use nalgebra::base::storage::Storage;
use nalgebra::{DMatrix, DVector, Dim, Matrix};
use std::f64;

/// Radial-basis function (RBF) kernel
/// The distance metric here is L2 (Euclidean).
///
/// ```math
///     K(\mathbf{x}, \mathbf{x'}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x'}\|^2}{2\sigma^2}\right)
/// ```
///
/// # Parameters
/// * `l` - Length scale.
///
#[derive(Clone, Debug, PartialEq)]
pub struct RBFKernel {
    length_scale: f64,
    lower_bound: f64,
    upper_bound: f64,
}

impl RBFKernel {
    pub fn new(length_scale: f64) -> Self {
        Self {
            length_scale,
            upper_bound: 1E5,
            lower_bound: 1E-5,
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

        for i in 0..m {
            for j in 0..n {
                let d = e2_norm(&x1.row(i), &x2.row(j), self.length_scale);
                dm[(i, j)] = d;
            }
        }

        dm.map(|e| (-0.5 * e).exp())
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

    fn parameters(&self) -> Vec<f64> {
        vec![self.length_scale.ln()]
    }

    fn parameter_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![self.lower_bound], vec![self.upper_bound])
    }

    fn consume_parameters(params: &[f64]) -> (Self, &[f64]) {
        assert!(!params.is_empty(), "RBFKernel requires one parameters");
        let (cur, next) = params.split_at(1);
        let ck = Self::from_parameters(cur);
        (ck, next)
    }

    fn from_parameters(params: &[f64]) -> Self {
        assert_eq!(
            params.len(),
            1,
            "The parameter vector for RBFKernel must be of length 1"
        );
        Self::new(params[0].exp())
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
        let n = x.nrows();

        let mut dm = DMatrix::zeros(n, n);
        let mut grad = CovGrad::zeros(n, 1);

        for i in 0..n {
            for j in 0..i {
                // Save covariance
                let d2 = e2_norm(&x.row(i), &x.row(j), self.length_scale);
                let exp_d2 = (-d2 / 2.0).exp();
                let cov_ij = exp_d2;

                dm[(i, j)] = cov_ij;
                dm[(j, i)] = cov_ij;

                // Save gradient
                let dc_dl = d2 * cov_ij;
                grad[(i, j, 0)] = dc_dl;
                grad[(j, i, 0)] = dc_dl;
            }
            dm[(i, i)] = 1.0;
        }

        (dm, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbf_gradient() {
        const E: f64 = std::f64::consts::E;
        let x = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let r = RBFKernel::new(1.0);
        let (cov, grad) = r.covariance_with_gradient(&x);

        let expected_cov = DMatrix::from_row_slice(
            2,
            2,
            &[1.0, 1.0 / E.powi(4), 1.0 / E.powi(4), 1.0],
        );

        let expected_grad = CovGrad::from_column_slices(
            2,
            1,
            &[0.0, 8.0 / E.powi(4), 8.0 / E.powi(4), 0.0],
        );
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));

        let r = RBFKernel::new(4.0);
        let (cov, grad) = r.covariance_with_gradient(&x);

        let expected_cov = DMatrix::from_row_slice(
            2,
            2,
            &[1.0, 1.0 / E.powf(1.0 / 4.0), 1.0 / E.powf(1.0 / 4.0), 1.0],
        );

        let expected_grad = CovGrad::from_column_slices(
            2,
            1,
            &[
                0.0,
                (1.0 / 2.0) / E.powf(0.25),
                (1.0 / 2.0) / E.powf(0.25),
                0.0,
            ],
        );

        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));
    }

    #[test]
    fn rbf_simple() {
        let kernel = RBFKernel::new(1.0);
        assert::close(kernel.parameters()[0], 0.0, 1E-10);
        assert_eq!(kernel, RBFKernel::from_parameters(&[0.0]));
        assert!(kernel.is_stationary());
    }

    #[test]
    fn rbf_1d() {
        let xs = DVector::from_column_slice(&[0.0, 1.0, 2.0, 3.0]);
        let kernel = RBFKernel::new(1.0);

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
    }

    #[test]
    fn rbf_2d() {
        use nalgebra::Matrix4x2;

        let kernel = RBFKernel::new(1.0);
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
        let kernel = RBFKernel::new(1.0);

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
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
    }
}
