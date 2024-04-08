use crate::misc::bessel::bessel_ikv_temme;

use super::{e2_norm, CovGrad, CovGradError, Kernel, KernelError};
use nalgebra::base::constraint::{SameNumberOfColumns, ShapeConstraint};
use nalgebra::base::storage::Storage;
use nalgebra::{dvector, DMatrix, DVector, Dim, Matrix};
use peroxide::prelude::gamma;
use std::f64;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Matérn Kernel
/// The Matérn kernel is given by
///
/// ```math
///   K(\mathbf{x}, \mathbf{x'}) = \frac{2^{1-\nu}}{\Gamma(\nu)} {\left( \sqrt{2 \nu} \frac{d}{l} \right)}^\nu K_\nu (\sqrt{2\nu} \frac{d}{l})
/// ```
/// where $\Gamma$ is the gamma function and $K_\nu$ is the Bessel funciton of the second kind.
///
/// # Parameters
/// * `nu` -
/// * `length_scale` -
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct MaternKernel {
    nu: f64,
    length_scale: f64,
}

impl MaternKernel {
    /// Create a new `MaternKernel` with `nu` and `length_scale`.
    pub fn new(nu: f64, length_scale: f64) -> Result<Self, KernelError> {
        if nu <= 0.0 {
            Err(KernelError::ParameterOutOfBounds {
                name: "nu".to_string(),
                given: nu,
                bounds: (0.0, f64::INFINITY),
            })
        } else if length_scale <= 0.0 {
            Err(KernelError::ParameterOutOfBounds {
                name: "length_scale".to_string(),
                given: length_scale,
                bounds: (0.0, f64::INFINITY),
            })
        } else {
            Ok(Self { nu, length_scale })
        }
    }

    /// Create a new `MaternKernel` with `nu` and `length_scale` without checking inputs.
    pub fn new_unchecked(nu: f64, length_scale: f64) -> Self {
        Self { nu, length_scale }
    }

    fn autocov<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DMatrix<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>,
        //ShapeConstraint: DimEq<C, R>,
    {
        let n = x.nrows();

        let mut dm: DMatrix<f64> = DMatrix::zeros(n, n);
        let c = (1.0 - self.nu).exp2() / gamma(self.nu);
        let sqrt_two_nu = (2.0 * self.nu).sqrt();

        for i in 0..n {
            for j in 0..i {
                let r = e2_norm(&x.row(i), &x.row(j), self.length_scale).sqrt();

                if r < f64::EPSILON {
                    dm[(i, j)] = 1.0;
                } else {
                    let tmp = sqrt_two_nu * r;
                    dm[(i, j)] = c
                        * tmp.powf(self.nu)
                        * bessel_ikv_temme(self.nu, tmp).unwrap().1;
                    dm[(j, i)] = dm[(i, j)];
                }
            }
            dm[(i, i)] = 1.0;
        }

        dm
    }
}

impl Default for MaternKernel {
    fn default() -> Self {
        Self {
            nu: 1.5,
            length_scale: 1.0,
        }
    }
}

impl Kernel for MaternKernel {
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
        let m = x1.nrows();
        let n = x2.nrows();

        let mut dm: DMatrix<f64> = DMatrix::zeros(m, n);
        let c = (1.0 - self.nu).exp2() / gamma(self.nu);
        let sqrt_two_nu = (2.0 * self.nu).sqrt();

        for i in 0..m {
            for j in 0..n {
                let r =
                    e2_norm(&x1.row(i), &x2.row(j), self.length_scale).sqrt();

                if r < f64::EPSILON {
                    dm[(i, j)] = 1.0;
                } else {
                    let tmp = sqrt_two_nu * r;
                    dm[(i, j)] = c
                        * tmp.powf(self.nu)
                        * bessel_ikv_temme(self.nu, tmp).unwrap().1
                }
            }
        }

        dm
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
        dvector![self.nu.ln(), self.length_scale.ln()]
    }

    fn reparameterize(&self, params: &[f64]) -> Result<Self, KernelError> {
        match params {
            [] => Err(KernelError::MissingParameters(2)),
            [_] => Err(KernelError::MissingParameters(1)),
            [ln_nu, ln_length] => Self::new(ln_nu.exp(), ln_length.exp()),
            _ => Err(KernelError::ExtraniousParameters(params.len() - 2)),
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
        // Epsilon for derivative approximation.
        const EPS: f64 = 1e-10;

        // Since there isn't a closed form solution for the derivative for K_nu, we just use an
        // approximation of the gradient.
        let cov = self.autocov(x);

        let mut dv_params = self.parameters();
        dv_params[0] += EPS;
        let cov_pdv = (self.clone())
            .reparameterize((&dv_params).into())
            .expect("Ought to be a valid matern kernel")
            .autocov(x);

        let grad_v = (&cov_pdv - &cov) / EPS;

        let mut dl_params = self.parameters();
        dl_params[1] += EPS;
        let cov_pdl = (self.clone())
            .reparameterize((&dl_params).into())
            .expect("Ought to be a valid matern kernel")
            .autocov(x);

        let grad_l = (&cov_pdl - &cov) / EPS;

        let grad = CovGrad::new(&[grad_v, grad_l])?;

        Ok((cov, grad))
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matern_cov() -> Result<(), KernelError> {
        let x = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let y = DMatrix::from_row_slice(2, 2, &[5.0, 4.0, 3.0, 2.0]);
        let k = MaternKernel::new_unchecked(1.0, 1.0);
        let cov = k.covariance(&x, &y);

        let expected_cov = DMatrix::from_row_slice(
            2,
            2,
            &[0.005_967_69, 0.139_667_47, 0.139_667_47, 0.139_667_47],
        );

        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        Ok(())
    }

    #[test]
    fn matern_gradient() -> Result<(), KernelError> {
        let x = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 2.0, 2.0]);
        let k = MaternKernel::new_unchecked(1.0, 1.0);
        let (cov, grad) = k.covariance_with_gradient(&x)?;
        let expected_cov = DMatrix::from_row_slice(
            2,
            2,
            &[1.0, 0.279_731_76, 0.279_731_76, 1.0],
        );

        let expected_grad = CovGrad::new_unchecked(&[
            DMatrix::from_row_slice(2, 2, &[0.0, 0.0475_717, 0.047_571_7, 0.0]),
            DMatrix::from_row_slice(2, 2, &[0.0, 0.455_575, 0.455_575, 0.0]),
        ]);

        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-4, 1E-4));
        Ok(())
    }
}
