use super::{CovGrad, CovGradError, Kernel, KernelError, e2_norm};
use nalgebra::base::constraint::{SameNumberOfColumns, ShapeConstraint};
use nalgebra::base::storage::Storage;
use nalgebra::{DMatrix, DVector, Dim, Matrix, dvector};
use std::f64;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

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
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct RBFKernel {
    length_scale: f64,
}

impl RBFKernel {
    /// Create a new rbf kernel with the given length scale
    pub fn new(length_scale: f64) -> Result<Self, KernelError> {
        if length_scale <= 0.0 {
            Err(KernelError::ParameterOutOfBounds {
                name: "length_scale".to_string(),
                given: length_scale,
                bounds: (0.0, f64::INFINITY),
            })
        } else {
            Ok(Self { length_scale })
        }
    }

    /// Create a new `RBFKernel` without checking parameters
    #[must_use]
    pub fn new_unchecked(length_scale: f64) -> Self {
        Self { length_scale }
    }
}

impl Default for RBFKernel {
    fn default() -> Self {
        Self { length_scale: 1.0 }
    }
}

impl Kernel for RBFKernel {
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

    fn parameters(&self) -> DVector<f64> {
        dvector![self.length_scale.ln()]
    }

    fn reparameterize(&self, params: &[f64]) -> Result<Self, KernelError> {
        match params {
            [] => Err(KernelError::MissingParameters(1)),
            [value] => Self::new(value.exp()),
            _ => Err(KernelError::ExtraneousParameters(params.len() - 1)),
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

        Ok((dm, grad))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbf_gradient() -> Result<(), KernelError> {
        const E: f64 = std::f64::consts::E;
        let x = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let r = RBFKernel::default();
        let (cov, grad) = r.covariance_with_gradient(&x)?;

        let expected_cov = DMatrix::from_row_slice(
            2,
            2,
            &[1.0, 1.0 / E.powi(4), 1.0 / E.powi(4), 1.0],
        );

        let expected_grad = CovGrad::from_column_slices(
            2,
            1,
            &[0.0, 8.0 / E.powi(4), 8.0 / E.powi(4), 0.0],
        )
        .unwrap();
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));

        let r = RBFKernel::new(4.0).expect("Has valid parameter");
        let (cov, grad) = r.covariance_with_gradient(&x)?;

        let expected_cov = DMatrix::from_row_slice(
            2,
            2,
            &[
                1.0,
                1.0 / (1.0_f64 / 4.0).exp(),
                1.0 / (1.0_f64 / 4.0).exp(),
                1.0,
            ],
        );

        let expected_grad = CovGrad::from_column_slices(
            2,
            1,
            &[
                0.0,
                (1.0 / 2.0) / 0.25_f64.exp(),
                (1.0 / 2.0) / 0.25_f64.exp(),
                0.0,
            ],
        )
        .unwrap();

        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));
        Ok(())
    }

    #[test]
    fn rbf_simple() {
        let kernel = RBFKernel::default();
        assert::close(kernel.parameters()[0], 0.0, 1E-10);
        assert_eq!(
            kernel,
            kernel
                .reparameterize(&[0.0])
                .expect("Should create kernel from params")
        );
        assert!(kernel.is_stationary());
    }

    #[test]
    fn rbf_1d() {
        let xs = DVector::from_column_slice(&[0.0, 1.0, 2.0, 3.0]);
        let kernel = RBFKernel::default();

        let cov = kernel.covariance(&xs, &xs);
        let expected_cov = DMatrix::from_column_slice(
            4,
            4,
            &[
                1.,
                0.606_530_66,
                0.135_335_28,
                0.011_109,
                0.606_530_66,
                1.,
                0.606_530_66,
                0.135_335_28,
                0.135_335_28,
                0.606_530_66,
                1.,
                0.606_530_66,
                0.011_109,
                0.135_335_28,
                0.606_530_66,
                1.,
            ],
        );

        assert!(expected_cov.relative_eq(&cov, 1E-8, 1E-8));
        let expected_diag = DVector::from_column_slice(&[1., 1., 1., 1.]);
        assert_eq!(kernel.diag(&xs), expected_diag);
    }

    #[test]
    fn rbf_2d() {
        use nalgebra::Matrix4x2;

        let kernel = RBFKernel::default();
        let xs =
            Matrix4x2::from_column_slice(&[0., 1., 2., 3., 4., 5., 6., 7.]);
        let expected_cov = DMatrix::from_column_slice(
            4,
            4,
            &[
                1.000_000_00e+00,
                3.678_794_41e-01,
                1.831_563_89e-02,
                1.234_098_04e-04,
                3.678_794_41e-01,
                1.000_000_00e+00,
                3.678_794_41e-01,
                1.831_563_89e-02,
                1.831_563_89e-02,
                3.678_794_41e-01,
                1.000_000_00e+00,
                3.678_794_41e-01,
                1.234_098_04e-04,
                1.831_563_89e-02,
                3.678_794_41e-01,
                1.000_000_00e+00,
            ],
        );

        let cov = kernel.covariance(&xs, &xs);
        assert!(expected_cov.relative_eq(&cov, 1E-8, 1E-8));
    }

    #[test]
    fn rbf_different_sizes() {
        use nalgebra::Matrix5x1;
        let kernel = RBFKernel::default();

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
                6.065_306_60e-01,
                1.000_000_00e+00,
                6.065_306_60e-01,
                1.353_352_83e-01,
                1.110_899_65e-02,
                3.354_626_28e-04,
                3.726_653_17e-06,
                1.522_997_97e-08,
                2.289_734_85e-11,
                1.266_416_55e-14,
                1.353_352_83e-01,
                6.065_306_60e-01,
                1.000_000_00e+00,
                6.065_306_60e-01,
                1.353_352_83e-01,
                1.110_899_65e-02,
                3.354_626_28e-04,
                3.726_653_17e-06,
                1.522_997_97e-08,
                2.289_734_85e-11,
                1.110_899_65e-02,
                1.353_352_83e-01,
                6.065_306_60e-01,
                1.000_000_00e+00,
                6.065_306_60e-01,
                1.353_352_83e-01,
                1.110_899_65e-02,
                3.354_626_28e-04,
                3.726_653_17e-06,
                1.522_997_97e-08,
                3.354_626_28e-04,
                1.110_899_65e-02,
                1.353_352_83e-01,
                6.065_306_60e-01,
                1.000_000_00e+00,
                6.065_306_60e-01,
                1.353_352_83e-01,
                1.110_899_65e-02,
                3.354_626_28e-04,
                3.726_653_17e-06,
                1.522_997_97e-08,
                3.726_653_17e-06,
                3.354_626_28e-04,
                1.110_899_65e-02,
                1.353_352_83e-01,
                6.065_306_60e-01,
                1.000_000_00e+00,
                6.065_306_60e-01,
                1.353_352_83e-01,
                1.110_899_65e-02,
            ],
        );
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
    }
}
