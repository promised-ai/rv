use super::{CovGrad, CovGradError, Kernel, KernelError};
use nalgebra::base::constraint::{SameNumberOfColumns, ShapeConstraint};
use nalgebra::base::storage::Storage;
use nalgebra::{DMatrix, DVector, Dim, Matrix};
use std::f64;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Squared Exponential function (SEard) kernel
/// The distance metric here is L2 (Euclidean).
///
/// ```math
///     k(a, b) = exp(-0.5 * (a - b)' * M * (a - b))
/// ```
///
/// # Parameters
/// * `M` - Length scale for each dimension.
///
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct SEardKernel {
    length_scale: DVector<f64>,
}

impl SEardKernel {
    /// Create a new seard kernel with the given length scale
    pub fn new(length_scale: DVector<f64>) -> Result<Self, KernelError> {
        if length_scale.iter().all(|x| x > &0.0) {
            Ok(Self { length_scale })
        } else {
            Err(KernelError::ParameterOutOfBounds {
                name: "length_scale".to_string(),
                given: *length_scale
                    .iter()
                    .min_by(|&&a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
                bounds: (0.0, std::f64::INFINITY),
            })
        }
    }

    /// Create a new SEardKernel without checking parameters
    pub fn new_unchecked(length_scale: DVector<f64>) -> Self {
        Self { length_scale }
    }
}

impl Kernel for SEardKernel {
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
        // k(a, b) = exp(-0.5 * (a - b)' * M * (a - b))

        let m = x1.nrows();
        let n = x2.nrows();
        let c = x1.ncols();

        let mut dm: DMatrix<f64> = DMatrix::zeros(m, n);

        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                let a = x1.row(i);
                let b = x2.row(j);
                for k in 1..c {
                    s += ((a[k] - b[k]) / self.length_scale[k]).powi(2);
                }
                dm[(i, j)] = s;
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
        self.length_scale.iter().map(|x| x.ln()).collect()
    }

    fn consume_parameters(
        params: &[f64],
    ) -> Result<(Self, &[f64]), KernelError> {
        if params.len() < self.length_scale.nrows() {
            Err(KernelError::MisingParameters(self.length_scale.nrows()))
        } else {
            let (cur, next) = params.split_at(self.length_scale.nrows());
            let ck = Self::from_parameters(cur)?;
            Ok((ck, next))
        }
    }

    fn from_parameters(params: &[f64]) -> Result<Self, KernelError> {
        if params.len() == self.length_scale.nrows() {
            let exped: Vec<f64> = params.iter().map(|x| x.exp()).collect();
            Ok(Self::new(DVector::from_row_slice(&exped)))
        } else if params.len() > self.length_scale.nrows() {
            Err(KernelError::ExtraniousParameters(
                params.len() - self.length_scale.nrows(),
            ))
        } else {
            Err(KernelError::MisingParameters(
                self.length_scale.nrows() - params.len(),
            ))
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

        let mut cov = DMatrix::zeros(n, n);
        let mut grad = CovGrad::zeros(n, 1);

        for i in 0..n {
            for j in 0..i {
                // Save covariance
                let mut d2: f64 = 0.0;
                for k in 0..x.ncols() {
                    let a = x.row(i);
                    let b = x.row(j);
                    d2 += ((a - b) / self.length_scale[k]).powi(2);
                }
                let cov_ij = (-d2 / 2.0_f64).exp();

                cov[(i, j)] = cov_ij;
                cov[(j, i)] = cov_ij;

                for k in 0..x.ncols() {
                    // Compute effect on cov for l_k
                    let a = x.row(i);
                    let b = x.row(j);
                    grad[(i, j, k)] = -2.0 * (a - b).pow(2) * cov_ij
                        / self.length_scale[k].powi(3);
                    grad[(j, i, k)] = grad[(i, j, k)];
                }
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
        let r = SEardKernel::default();
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

        let r = SEardKernel::new(4.0).expect("Has valid parameter");
        let (cov, grad) = r.covariance_with_gradient(&x)?;

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
        )
        .unwrap();

        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));
        Ok(())
    }

    #[test]
    fn rbf_simple() {
        let kernel = SEardKernel::default();
        assert::close(kernel.parameters()[0], 0.0, 1E-10);
        assert_eq!(
            kernel,
            SEardKernel::from_parameters(&[0.0])
                .expect("Should create kernel from params")
        );
        assert!(kernel.is_stationary());
    }

    #[test]
    fn rbf_1d() {
        let xs = DVector::from_column_slice(&[0.0, 1.0, 2.0, 3.0]);
        let kernel = SEardKernel::default();

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

        let kernel = SEardKernel::default();
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
        let kernel = SEardKernel::default();

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
