use super::{CovGrad, CovGradError, Kernel, KernelError};
use nalgebra::{
    base::constraint::{SameNumberOfColumns, ShapeConstraint},
    dvector, EuclideanNorm,
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
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
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
    fn n_parameters(&self) -> usize {
        2
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

    /// Returns the diagonal of the kernel(x, x)
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
    fn parameters(&self) -> DVector<f64> {
        dvector![self.length_scale.ln(), self.periodicity.ln()]
    }

    /// Create a new kernel of the given type from the provided parameters.
    /// The parameters here are in a log-scale
    fn reparameterize(&self, params: &[f64]) -> Result<Self, KernelError> {
        match params {
            [] => Err(KernelError::MissingParameters(2)),
            [_] => Err(KernelError::MissingParameters(1)),
            [length_scale, periodicity] => {
                Self::new(length_scale.exp(), periodicity.exp())
            }
            _ => Err(KernelError::ExtraneousParameters(params.len() - 1)),
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
                0.383_938_97,
                0.692_107_48,
                0.853_810_81,
                0.633_565_1,
                0.633_565_1,
                0.383_938_97,
                0.692_107_48,
                0.853_810_81,
                0.633_565_1,
                0.633_565_1,
                0.383_938_97,
                0.692_107_48,
                0.853_810_81,
                0.633_565_1,
                0.633_565_1,
                0.383_938_97,
                0.692_107_48,
                0.853_810_81,
                0.633_565_1,
                0.633_565_1,
                0.383_938_97,
                0.692_107_48,
                0.853_810_81,
                0.633_565_1,
                0.633_565_1,
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
                    0.000_000_00e+00,
                    5.999_039_13e-32,
                    2.399_615_65e-31,
                    5.399_135_22e-31,
                    1.499_759_78e-30,
                    5.999_039_13e-32,
                    0.000_000_00e+00,
                    5.999_039_13e-32,
                    2.399_615_65e-31,
                    9.598_462_61e-31,
                    2.399_615_65e-31,
                    5.999_039_13e-32,
                    0.000_000_00e+00,
                    5.999_039_13e-32,
                    5.399_135_22e-31,
                    5.399_135_22e-31,
                    2.399_615_65e-31,
                    5.999_039_13e-32,
                    0.000_000_00e+00,
                    2.399_615_65e-31,
                    1.499_759_78e-30,
                    9.598_462_61e-31,
                    5.399_135_22e-31,
                    2.399_615_65e-31,
                    0.000_000_00e+00,
                ],
            ),
            DMatrix::from_row_slice(
                5,
                5,
                &[
                    0.000_000_00e+00,
                    -1.538_936_55e-15,
                    -6.155_746_22e-15,
                    -1.385_042_90e-14,
                    -3.847_341_39e-14,
                    -1.538_936_55e-15,
                    0.000_000_00e+00,
                    -1.538_936_55e-15,
                    -6.155_746_22e-15,
                    -2.462_298_49e-14,
                    -6.155_746_22e-15,
                    -1.538_936_55e-15,
                    0.000_000_00e+00,
                    -1.538_936_55e-15,
                    -1.385_042_90e-14,
                    -1.385_042_90e-14,
                    -6.155_746_22e-15,
                    -1.538_936_55e-15,
                    0.000_000_00e+00,
                    -6.155_746_22e-15,
                    -3.847_341_39e-14,
                    -2.462_298_49e-14,
                    -1.385_042_90e-14,
                    -6.155_746_22e-15,
                    0.000_000_00e+00,
                ],
            ),
        ]);
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));
        Ok(())
    }

    #[test]
    fn expsinesquared_kernel_b() -> Result<(), KernelError> {
        use crate::consts::TWO_PI;
        let x: DMatrix<f64> =
            DMatrix::from_row_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        // Non default variables
        let kernel = ExpSineSquaredKernel::new(5.0, TWO_PI)?;
        let (cov, grad) = kernel.covariance_with_gradient(&x)?;
        let expected_cov = DMatrix::from_row_slice(
            5,
            5,
            &[
                1.,
                0.981_780_12,
                0.944_928_63,
                0.923_485_94,
                0.971_753_11,
                0.981_780_12,
                1.,
                0.981_780_12,
                0.944_928_63,
                0.935_994_44,
                0.944_928_63,
                0.981_780_12,
                1.,
                0.981_780_12,
                0.923_485_94,
                0.923_485_94,
                0.944_928_63,
                0.981_780_12,
                1.,
                0.944_928_63,
                0.971_753_11,
                0.935_994_44,
                0.923_485_94,
                0.944_928_63,
                1.,
            ],
        );

        let expected_grad = CovGrad::new_unchecked(&[
            DMatrix::from_row_slice(
                5,
                5,
                &[
                    0.,
                    0.036_105_76,
                    0.107_052_62,
                    0.147_018_41,
                    0.055_688_28,
                    0.036_105_76,
                    0.,
                    0.036_105_76,
                    0.107_052_62,
                    0.123_824_1,
                    0.107_052_62,
                    0.036_105_76,
                    0.,
                    0.036_105_76,
                    0.147_018_41,
                    0.147_018_41,
                    0.107_052_62,
                    0.036_105_76,
                    0.,
                    0.107_052_62,
                    0.055_688_28,
                    0.123_824_1,
                    0.147_018_41,
                    0.107_052_62,
                    0.,
                ],
            ),
            DMatrix::from_row_slice(
                5,
                5,
                &[
                    0.,
                    0.033_045_58,
                    0.068_737_69,
                    0.015_638_68,
                    -0.186_367_53,
                    0.033_045_58,
                    0.,
                    0.033_045_58,
                    0.068_737_69,
                    -0.113_338_07,
                    0.068_737_69,
                    0.033_045_58,
                    0.,
                    0.033_045_58,
                    0.015_638_68,
                    0.015_638_68,
                    0.068_737_69,
                    0.033_045_58,
                    0.,
                    0.068_737_69,
                    -0.186_367_53,
                    -0.113_338_07,
                    0.015_638_68,
                    0.068_737_69,
                    0.,
                ],
            ),
        ]);
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));

        Ok(())
    }
}
