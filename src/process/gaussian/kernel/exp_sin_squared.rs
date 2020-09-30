use super::{CovGrad, CovGradError, Kernel, KernelError};
use nalgebra::{
    base::constraint::{SameNumberOfColumns, ShapeConstraint},
    EuclideanNorm,
};
use nalgebra::{base::storage::Storage, Norm};
use nalgebra::{DMatrix, DVector, Dim, Matrix};
use std::f64;
use std::f64::consts::PI;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Exp Sine^2 Kernel
/// k(x_i, x_j) = exp(-2 (sin(pi / periodicity * d(x_i, x_j)) / length_scale) ^ 2)
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ExpSineSquaredKernel {
    length_scale: f64,
    periodicity: f64,
}

impl ExpSineSquaredKernel {
    /// Create a new ExpSineSquaredKernel
    pub fn new(
        length_scale: f64,
        periodicity: f64,
    ) -> Result<Self, KernelError> {
        if length_scale <= 0.0 {
            Err(KernelError::ParameterOutOfBounds {
                name: "length_scale".to_string(),
                given: length_scale,
                bounds: (0.0, f64::INFINITY),
            })
        } else if periodicity <= 0.0 {
            Err(KernelError::ParameterOutOfBounds {
                name: "periodicity".to_string(),
                given: periodicity,
                bounds: (0.0, f64::INFINITY),
            })
        } else {
            Ok(Self {
                length_scale,
                periodicity,
            })
        }
    }

    /// Create a new ExpSineSquaredKernel without checking the parameters
    pub fn new_unchecked(length_scale: f64, periodicity: f64) -> Self {
        Self {
            length_scale,
            periodicity,
        }
    }
}

impl Kernel for ExpSineSquaredKernel {
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
        assert!(x1.ncols() == x2.ncols());
        let metric = EuclideanNorm {};
        let l2 = self.length_scale.powi(2);

        DMatrix::from_fn(x1.nrows(), x2.nrows(), |i, j| {
            let d = metric.metric_distance(&x1.row(i), &x2.row(j));
            let s2 = (PI * d / self.periodicity).sin().powi(2);
            (-2.0 * s2 / l2).exp()
        })
    }
    fn is_stationary(&self) -> bool {
        true
    }

    /// Returns the diagnal of the kernel(x, x)
    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>,
    {
        DVector::repeat(x.len(), 1.0)
    }

    /// Return the corresponding parameter vector
    /// The parameters here are in a log-scale
    fn parameters(&self) -> Vec<f64> {
        vec![self.length_scale.ln(), self.periodicity.ln()]
    }

    /// Create a new kernel of the given type from the provided parameters.
    /// The parameters here are in a log-scale
    fn from_parameters(params: &[f64]) -> Result<Self, KernelError> {
        match params {
            [] => Err(KernelError::MisingParameters(2)),
            [_] => Err(KernelError::MisingParameters(1)),
            [length_scale, periodicity] => {
                Self::new(length_scale.exp(), periodicity.exp())
            }
            _ => Err(KernelError::ExtraniousParameters(params.len() - 1)),
        }
    }

    /// Takes a sequence of parameters and consumes only the ones it needs
    /// to create itself.
    /// The parameters here are in a log-scale
    fn consume_parameters(
        params: &[f64],
    ) -> Result<(Self, &[f64]), KernelError> {
        if params.len() < 2 {
            Err(KernelError::MisingParameters(2))
        } else {
            let (cur, next) = params.split_at(2);
            let ck = Self::from_parameters(cur)?;
            Ok((ck, next))
        }
    }

    /// Covariance and Gradient with the log-scaled hyper-parameters
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
        let mut grad = CovGrad::zeros(n, 2);
        let metric = EuclideanNorm {};
        let l2 = self.length_scale.powi(2);

        // Fill in lower triangular portion and copy it to upper triangular portion
        for i in 0..n {
            for j in 0..i {
                let d = metric.metric_distance(&x.row(i), &x.row(j));
                let arg = PI * d / self.periodicity;

                let sin_arg = arg.sin();
                let sin_arg_2 = sin_arg.powi(2);
                let cos_arg = arg.cos();

                let k = (-2.0 * sin_arg_2 / l2).exp();
                cov[(i, j)] = k;
                cov[(j, i)] = k;

                let dk_dl = 4.0 * sin_arg_2 * k / l2;
                grad[(i, j, 0)] = dk_dl;
                grad[(j, i, 0)] = dk_dl;

                let dk_dp = (4.0 * arg / l2) * cos_arg * sin_arg * k;
                grad[(i, j, 1)] = dk_dp;
                grad[(j, i, 1)] = dk_dp;
            }
            // Diag is always one
            cov[(i, i)] = 1.0;
        }
        Ok((cov, grad))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expsinesquared_kernel_a() -> Result<(), KernelError> {
        let kernel = ExpSineSquaredKernel::new(3.0, 5.0)?;
        assert!(kernel.is_stationary());

        let kernel = ExpSineSquaredKernel::new(1.0, 1.0)?;

        let x: DMatrix<f64> =
            DMatrix::from_row_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y = x.map(|z| z.sin());

        let cov = kernel.covariance(&x, &y);
        let expected_cov = DMatrix::from_row_slice(
            5,
            5,
            &[
                0.38393897, 0.69210748, 0.85381081, 0.6335651, 0.6335651,
                0.38393897, 0.69210748, 0.85381081, 0.6335651, 0.6335651,
                0.38393897, 0.69210748, 0.85381081, 0.6335651, 0.6335651,
                0.38393897, 0.69210748, 0.85381081, 0.6335651, 0.6335651,
                0.38393897, 0.69210748, 0.85381081, 0.6335651, 0.6335651,
            ],
        );
        assert!(cov.relative_eq(&expected_cov, 1E-7, 1E-7));

        let (cov, grad) = kernel.covariance_with_gradient(&x)?;

        let expected_cov = DMatrix::from_row_slice(
            5,
            5,
            &[
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ],
        );

        let expected_grad = CovGrad::new_unchecked(&[
            DMatrix::from_row_slice(
                5,
                5,
                &[
                    0.00000000e+00,
                    5.99903913e-32,
                    2.39961565e-31,
                    5.39913522e-31,
                    1.49975978e-30,
                    5.99903913e-32,
                    0.00000000e+00,
                    5.99903913e-32,
                    2.39961565e-31,
                    9.59846261e-31,
                    2.39961565e-31,
                    5.99903913e-32,
                    0.00000000e+00,
                    5.99903913e-32,
                    5.39913522e-31,
                    5.39913522e-31,
                    2.39961565e-31,
                    5.99903913e-32,
                    0.00000000e+00,
                    2.39961565e-31,
                    1.49975978e-30,
                    9.59846261e-31,
                    5.39913522e-31,
                    2.39961565e-31,
                    0.00000000e+00,
                ],
            ),
            DMatrix::from_row_slice(
                5,
                5,
                &[
                    0.00000000e+00,
                    -1.53893655e-15,
                    -6.15574622e-15,
                    -1.38504290e-14,
                    -3.84734139e-14,
                    -1.53893655e-15,
                    0.00000000e+00,
                    -1.53893655e-15,
                    -6.15574622e-15,
                    -2.46229849e-14,
                    -6.15574622e-15,
                    -1.53893655e-15,
                    0.00000000e+00,
                    -1.53893655e-15,
                    -1.38504290e-14,
                    -1.38504290e-14,
                    -6.15574622e-15,
                    -1.53893655e-15,
                    0.00000000e+00,
                    -6.15574622e-15,
                    -3.84734139e-14,
                    -2.46229849e-14,
                    -1.38504290e-14,
                    -6.15574622e-15,
                    0.00000000e+00,
                ],
            ),
        ]);
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));
        Ok(())
    }

    #[test]
    fn expsinesquared_kernel_b() -> Result<(), KernelError> {
        let x: DMatrix<f64> =
            DMatrix::from_row_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        // Non default variables
        let kernel = ExpSineSquaredKernel::new(5.0, 2.0 * f64::consts::PI)?;
        let (cov, grad) = kernel.covariance_with_gradient(&x)?;
        let expected_cov = DMatrix::from_row_slice(
            5,
            5,
            &[
                1., 0.98178012, 0.94492863, 0.92348594, 0.97175311, 0.98178012,
                1., 0.98178012, 0.94492863, 0.93599444, 0.94492863, 0.98178012,
                1., 0.98178012, 0.92348594, 0.92348594, 0.94492863, 0.98178012,
                1., 0.94492863, 0.97175311, 0.93599444, 0.92348594, 0.94492863,
                1.,
            ],
        );

        let expected_grad = CovGrad::new_unchecked(&[
            DMatrix::from_row_slice(
                5,
                5,
                &[
                    0., 0.03610576, 0.10705262, 0.14701841, 0.05568828,
                    0.03610576, 0., 0.03610576, 0.10705262, 0.1238241,
                    0.10705262, 0.03610576, 0., 0.03610576, 0.14701841,
                    0.14701841, 0.10705262, 0.03610576, 0., 0.10705262,
                    0.05568828, 0.1238241, 0.14701841, 0.10705262, 0.,
                ],
            ),
            DMatrix::from_row_slice(
                5,
                5,
                &[
                    0.,
                    0.03304558,
                    0.06873769,
                    0.01563868,
                    -0.18636753,
                    0.03304558,
                    0.,
                    0.03304558,
                    0.06873769,
                    -0.11333807,
                    0.06873769,
                    0.03304558,
                    0.,
                    0.03304558,
                    0.01563868,
                    0.01563868,
                    0.06873769,
                    0.03304558,
                    0.,
                    0.06873769,
                    -0.18636753,
                    -0.11333807,
                    0.01563868,
                    0.06873769,
                    0.,
                ],
            ),
        ]);
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));

        Ok(())
    }
}
