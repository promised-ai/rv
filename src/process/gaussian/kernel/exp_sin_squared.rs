use super::{CovGrad, Kernel};
use nalgebra::{
    base::constraint::{SameNumberOfColumns, ShapeConstraint},
    EuclideanNorm,
};
use nalgebra::{base::storage::Storage, Norm};
use nalgebra::{DMatrix, DVector, Dim, Matrix};
use std::f64;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Exp Sine^2 Kernel
/// k(x_i, x_j) = exp(-2 (sin(pi / periodicity * d(x_i, x_j)) / length_scale) ^ 2)
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ExpSineSquaredKernel {
    length_scale: f64,
    length_scale_lower_bound: f64,
    length_scale_upper_bound: f64,
    periodicity: f64,
    periodicity_lower_bound: f64,
    periodicity_upper_bound: f64,
}

impl ExpSineSquaredKernel {
    /// Create a new ExpSineSquaredKernel
    pub fn new(periodicity: f64, length_scale: f64) -> Self {
        Self {
            length_scale,
            periodicity,
            length_scale_lower_bound: 1E-5,
            length_scale_upper_bound: 1E5,
            periodicity_lower_bound: 1E-5,
            periodicity_upper_bound: 1E5,
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
        let mut cov = DMatrix::zeros(x1.nrows(), x2.nrows());
        const PI: f64 = std::f64::consts::PI;
        let l2 = self.length_scale.powi(2);
        for i in 0..x1.nrows() {
            for j in 0..x2.nrows() {
                let d = metric.metric_distance(&x1.row(i), &x2.row(j));
                cov[(i, j)] =
                    (-2.0 * (PI * d / self.periodicity).sin().powi(2) / l2)
                        .exp();
            }
        }
        cov
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

    fn parameter_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![self.length_scale_lower_bound, self.periodicity_lower_bound],
            vec![self.length_scale_upper_bound, self.periodicity_upper_bound],
        )
    }
    /// Create a new kernel of the given type from the provided parameters.
    /// The parameters here are in a log-scale
    fn from_parameters(params: &[f64]) -> Self {
        assert_eq!(
            params.len(),
            2,
            "ExpSineSquaredKernel requires two parameters"
        );
        let length_scale = params[0].exp();
        let periodicity = params[1].exp();
        Self::new(length_scale, periodicity)
    }

    /// Takes a sequence of parameters and consumes only the ones it needs
    /// to create itself.
    /// The parameters here are in a log-scale
    fn consume_parameters(params: &[f64]) -> (Self, &[f64]) {
        assert!(
            params.len() >= 2,
            "ExpSineSquaredKernel requires two parameters"
        );
        let (cur, next) = params.split_at(2);
        let ck = Self::from_parameters(cur);
        (ck, next)
    }

    /// Covariance and Gradient with the log-scaled hyper-parameters
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
        let mut cov = DMatrix::zeros(n, n);
        let mut grad = CovGrad::zeros(n, 2);
        let metric = EuclideanNorm {};
        const PI: f64 = std::f64::consts::PI;
        let l2 = self.length_scale.powi(2);
        for i in 0..n {
            for j in 0..i {
                let d = metric.metric_distance(&x.row(i), &x.row(j));
                let arg = PI * d / self.periodicity;
                let sin_arg = arg.sin();
                let cos_arg = arg.cos();
                let k = (-2.0 * arg.sin().powi(2) / l2).exp();
                cov[(i, j)] = k;
                cov[(j, i)] = k;

                let dk_dl = (4.0 / l2) * sin_arg.powi(2) * k;
                grad[(i, j, 0)] = dk_dl;
                grad[(j, i, 0)] = dk_dl;

                let dk_dp = (4.0 * arg / l2) * cos_arg * sin_arg * k;
                grad[(i, j, 1)] = dk_dp;
                grad[(j, i, 1)] = dk_dp;
            }
            cov[(i, i)] = 1.0;
        }
        (cov, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expsinesquared_kernel() {
        let kernel = ExpSineSquaredKernel::new(3.0, 5.0);
        assert!(kernel.is_stationary());

        let kernel = ExpSineSquaredKernel::new(1.0, 1.0);

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

        let (cov, grad) = kernel.covariance_with_gradient(&x);

        let expected_cov = DMatrix::from_row_slice(
            5,
            5,
            &[
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ],
        );

        let expected_grad = CovGrad::new(&[
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
    }
}
