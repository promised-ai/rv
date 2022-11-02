use super::{e2_norm, CovGrad, CovGradError, Kernel, KernelError, E2METRIC};
use nalgebra::base::storage::Storage;
use nalgebra::{
    base::constraint::{SameNumberOfColumns, ShapeConstraint},
    Norm,
};
use nalgebra::{DMatrix, DVector, Dim, Matrix};
use std::f64;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Rational Quadratic Kernel
///
/// # Parameters
/// `scale` -- Length scale
/// `mixture` -- Mixture Scale
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct RationalQuadratic {
    scale: f64,
    mixture: f64,
}

impl RationalQuadratic {
    /// Create a new RationalQuadratic kernel
    pub fn new(scale: f64, mixture: f64) -> Result<Self, KernelError> {
        if scale <= 0.0 {
            Err(KernelError::ParameterOutOfBounds {
                name: "scale".to_string(),
                given: scale,
                bounds: (0.0, f64::INFINITY),
            })
        } else if mixture <= 0.0 {
            Err(KernelError::ParameterOutOfBounds {
                name: "mixture".to_string(),
                given: mixture,
                bounds: (0.0, f64::INFINITY),
            })
        } else {
            Ok(Self { scale, mixture })
        }
    }

    /// Create a new RationalQuadratic without checking values
    pub fn new_unchecked(scale: f64, mixture: f64) -> Self {
        Self { scale, mixture }
    }
}

impl Kernel for RationalQuadratic {
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
        let d = (2.0 * self.scale * self.scale * self.mixture).sqrt();
        DMatrix::from_fn(x1.nrows(), x2.nrows(), |i, j| {
            let t = e2_norm(&x1.row(i), &x2.row(j), d);
            (1.0 + t).powf(-self.mixture)
        })
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
        DVector::repeat(x.len(), 1.0)
    }

    fn parameters(&self) -> Vec<f64> {
        vec![self.scale.ln(), self.mixture.ln()]
    }

    fn reparameterize(&self, params: &[f64]) -> Result<Self, KernelError> {
        match params {
            [] => Err(KernelError::MissingParameters(2)),
            [_] => Err(KernelError::MissingParameters(1)),
            [length_scale, mixture] => {
                Self::new(length_scale.exp(), mixture.exp())
            }
            _ => Err(KernelError::ExtraniousParameters(params.len() - 1)),
        }
    }

    fn consume_parameters<'p>(
        &self,
        params: &'p [f64],
    ) -> Result<(Self, &'p [f64]), KernelError> {
        if params.len() < 2 {
            Err(KernelError::MissingParameters(2))
        } else {
            let (cur, next) = params.split_at(2);
            let ck = Self::reparameterize(self, cur)?;
            Ok((ck, next))
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
        let mut grad = CovGrad::zeros(n, 2);
        let d = 2.0 * self.mixture * self.scale.powi(2);
        for i in 0..n {
            // off-diagnols
            for j in 0..i {
                let d2 = E2METRIC.metric_distance(&x.row(i), &x.row(j));
                let temp = d2 / d;
                let base = 1.0 + temp;
                let k = base.powf(-self.mixture);
                cov[(i, j)] = k;
                cov[(j, i)] = k;

                let dk_dl = d2 * k / (self.scale.powi(2) * base);
                let dk_da = k * base.ln().mul_add(
                    -self.mixture,
                    d2 / (2.0 * self.scale.powi(2) * base),
                );

                grad[(i, j, 0)] = dk_dl;
                grad[(j, i, 0)] = dk_dl;
                grad[(j, i, 1)] = dk_da;
                grad[(i, j, 1)] = dk_da;
            }
            // diag
            cov[(i, i)] = 1.0;
        }
        Ok((cov, grad))
    }
}

#[cfg(test)]
mod tests {
    use crate::test::relative_eq;

    use super::*;

    #[test]
    fn rational_quadratic() -> Result<(), KernelError> {
        let kernel = RationalQuadratic::new(3.0, 5.0)?;
        assert::close(kernel.parameters()[0], 3.0_f64.ln(), 1E-10);
        assert::close(kernel.parameters()[1], 5.0_f64.ln(), 1E-10);
        assert!(relative_eq(
            kernel.parameters(),
            vec![3.0_f64.ln(), 5.0_f64.ln()],
            1E-10,
            1E-10
        ));

        let x = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let y = DMatrix::from_row_slice(2, 2, &[5.0, 7.0, 6.0, 8.0]);

        let cov = kernel.covariance(&x, &y);
        let expected_cov = DMatrix::from_row_slice(
            2,
            2,
            &[
                5_904_900_000.0 / 38_579_489_651.0,
                5_904_900_000.0 / 78_502_725_751.0,
                5_904_900_000.0 / 11_592_740_742.0,
                1_889_568.0 / 6_436_343.0,
            ],
        );
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));

        let (cov, grad) = kernel.covariance_with_gradient(&x)?;

        let expected_cov = DMatrix::from_row_slice(
            2,
            2,
            &[
                1.0,
                184528125.0 / 282475249.0,
                184528125.0 / 282475249.0,
                1.0,
            ],
        );

        let eg_l = 0.53326868;
        let eg_a = -0.01151411;
        let expected_grad = CovGrad::from_row_slices(
            2,
            2,
            &[0.0, eg_l, eg_l, 0.0, 0.0, eg_a, eg_a, 0.0],
        )
        .unwrap();
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));
        Ok(())
    }
}
