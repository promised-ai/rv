use super::{CovGrad, CovGradError, Kernel, KernelError};
use nalgebra::base::constraint::{SameNumberOfColumns, ShapeConstraint};
use nalgebra::base::storage::Storage;
use nalgebra::{DMatrix, DVector, Dim, Matrix};
use std::f64;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Squared Exponential function (`SEard`) kernel
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
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
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
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
                bounds: (0.0, f64::INFINITY),
            })
        }
    }

    /// Create a new `SEardKernel` without checking parameters
    #[must_use] pub fn new_unchecked(length_scale: DVector<f64>) -> Self {
        Self { length_scale }
    }
}

#[allow(clippy::many_single_char_names)]
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
                for k in 0..c {
                    let term = (a[k] - b[k]) / self.length_scale[k];
                    s += term * term;
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

    fn parameters(&self) -> DVector<f64> {
        DVector::from_iterator(
            self.n_parameters(),
            self.length_scale.iter().map(|x| x.ln()),
        )
    }

    fn reparameterize(&self, params: &[f64]) -> Result<Self, KernelError> {
        use std::cmp::Ordering;
        match params.len().cmp(&self.length_scale.nrows()) {
            Ordering::Equal => {
                let exped: Vec<f64> = params.iter().map(|x| x.exp()).collect();
                Ok(Self::new(DVector::from_row_slice(&exped)).unwrap())
            }
            Ordering::Greater => Err(KernelError::ExtraneousParameters(
                params.len() - self.length_scale.nrows(),
            )),
            Ordering::Less => Err(KernelError::MissingParameters(
                self.length_scale.nrows() - params.len(),
            )),
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

        let cov = DMatrix::identity(n, n);
        let mut grad = CovGrad::zeros(n, self.length_scale.nrows());

        for i in 0..n {
            for j in 0..i {
                // Save covariance
                let mut d2: f64 = 0.0;
                for k in 0..x.ncols() {
                    let a = x.row(i);
                    let b = x.row(j);
                    d2 += a.zip_fold(&b, 0.0_f64, |acc, c, d| {
                        let diff = (c - d) / self.length_scale[k];
                        diff.mul_add(diff, acc)
                    });
                }
                let cov_ij = (-d2 / 2.0_f64).exp();

                for k in 0..x.ncols() {
                    // Compute effect on cov for l_k
                    let a = x.row(i);
                    let b = x.row(j);
                    // M = diag(l_0^-2, l_1^-2, l_2^-2)
                    // cov = exp((a-b)' * M * (a-b))
                    // d/dl_k = ( -2*(a_k - b_k)^2 / l_k^3 ) * cov
                    grad[(i, j, k)] = -2.0 * (a[k] - b[k]).powi(2) * cov_ij
                        / self.length_scale[k].powi(3);
                    grad[(j, i, k)] = grad[(i, j, k)];
                }
            }
        }
        Ok((cov, grad))
    }

    fn n_parameters(&self) -> usize {
        self.length_scale.nrows()
    }
}
