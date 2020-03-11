#[cfg(feature = "serde1")]
use serde_derive::{Deserialize, Serialize};

use crate::dist::MvGaussian;
use crate::misc::lnmv_gamma;
use crate::traits::*;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::f64::consts::LN_2;
use std::fmt;

/// [Inverse Wishart distribution](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution),
/// W<sup>-1</sup>(**Ψ**,ν) over positive definite matrices.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct InvWishart {
    /// p-dimensional inverse scale matrix, **Ψ**
    inv_scale: DMatrix<f64>,
    /// Degrees of freedom, ν > p - 1
    df: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum InvWishartError {
    /// The scale matrix is not square
    ScaleMatrixNotSquare { nrows: usize, ncols: usize },
    /// Degrees of freedom is less than the number of dimension of the scale
    /// matrix.
    DfLessThanDimensions { df: usize, ndims: usize },
}

#[inline]
fn validate_inv_scale(
    inv_scale: &DMatrix<f64>,
    df: usize,
) -> Result<(), InvWishartError> {
    if !inv_scale.is_square() {
        Err(InvWishartError::ScaleMatrixNotSquare {
            nrows: inv_scale.nrows(),
            ncols: inv_scale.ncols(),
        })
    } else if df < inv_scale.nrows() {
        Err(InvWishartError::DfLessThanDimensions {
            df,
            ndims: inv_scale.nrows(),
        })
    } else {
        Ok(())
    }
}

impl InvWishart {
    /// Create an Inverse Wishart distribution, W<sup>-1</sup>(**Ψ**,ν) with
    /// p-by-p inverse scale matrix, **Ψ**, and degrees of freedom, ν > p - 1.
    ///
    /// # Arguments
    /// - inv_scale: p-dimensional inverse scale matrix, **Ψ**
    /// - df: Degrees of freedom, ν > p - 1
    #[inline]
    pub fn new(
        inv_scale: DMatrix<f64>,
        df: usize,
    ) -> Result<Self, InvWishartError> {
        validate_inv_scale(&inv_scale, df)?;
        Ok(InvWishart { inv_scale, df })
    }

    /// Creates a new InvWishart without checking whether the parameters are
    /// valid.
    #[inline]
    pub fn new_unchecked(inv_scale: DMatrix<f64>, df: usize) -> Self {
        InvWishart { inv_scale, df }
    }

    /// Create an Inverse Wishart distribution, W<sup>-1</sup>(**I**<sup>p</sup>,
    /// p)
    #[inline]
    pub fn identity(dims: usize) -> Self {
        InvWishart {
            inv_scale: DMatrix::identity(dims, dims),
            df: dims,
        }
    }

    #[inline]
    pub fn ndims(&self) -> usize {
        self.inv_scale.nrows()
    }

    /// Get a reference to the inverse scale parameter
    #[inline]
    pub fn inv_scale(&self) -> &DMatrix<f64> {
        &self.inv_scale
    }

    /// Get the degrees of freedom
    #[inline]
    pub fn df(&self) -> usize {
        self.df
    }

    /// Set the value of df
    #[inline]
    pub fn set_df(&mut self, df: usize) -> Result<(), InvWishartError> {
        let ndims = self.ndims();
        if df < ndims {
            Err(InvWishartError::DfLessThanDimensions { df, ndims })
        } else {
            self.set_df_unchecked(df);
            Ok(())
        }
    }

    /// Set the value of df without input validation
    #[inline]
    pub fn set_df_unchecked(&mut self, df: usize) {
        self.df = df;
    }

    /// Set inverse scale parameter
    #[inline]
    pub fn set_inv_scale(
        &mut self,
        inv_scale: DMatrix<f64>,
    ) -> Result<(), InvWishartError> {
        validate_inv_scale(&inv_scale, self.df)?;
        self.inv_scale = inv_scale;
        Ok(())
    }

    #[inline]
    pub fn set_inv_scale_unchecked(&mut self, inv_scale: DMatrix<f64>) {
        self.inv_scale = inv_scale;
    }
}

impl Rv<DMatrix<f64>> for InvWishart {
    fn ln_f(&self, x: &DMatrix<f64>) -> f64 {
        let p = self.inv_scale.nrows();
        let pf = p as f64;
        let v = self.df as f64;

        // TODO: cache det_s
        let det_s: f64 = v * 0.5 * self.inv_scale.determinant().ln();
        let det_x: f64 = -(v + pf + 1.0) * 0.5 * x.determinant().ln();

        // TODO: cache denom
        let denom: f64 = v * pf * 0.5 * LN_2 + lnmv_gamma(p, 0.5 * v);
        let numer: f64 =
            -0.5 * (&self.inv_scale * x.clone().try_inverse().unwrap()).trace();

        det_s - denom + det_x + numer
    }

    // XXX: The complexity of this is O(df * dims^2). There is a O(dims^2)
    // algorithm, but it's more complicated to implement, so standby.
    // See https://www.math.wustl.edu/~sawyer/hmhandouts/Wishart.pdf  for more
    fn draw<R: Rng>(&self, mut rng: &mut R) -> DMatrix<f64> {
        let p = self.inv_scale.nrows();
        let scale = self.inv_scale.clone().try_inverse().unwrap();
        let mvg = MvGaussian::new_unchecked(DVector::zeros(p), scale);
        let xs = mvg.sample(self.df, &mut rng);
        let y = xs.iter().fold(DMatrix::<f64>::zeros(p, p), |acc, x| {
            // TODO: faster way to do X * X^T?
            acc + x * x.transpose()
        });
        y.try_inverse().unwrap()
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<DMatrix<f64>> {
        let p = self.inv_scale.nrows();
        let scale = self.inv_scale.clone().try_inverse().unwrap();
        let mvg = MvGaussian::new_unchecked(DVector::zeros(p), scale);
        (0..n)
            .map(|_| {
                let xs = mvg.sample(self.df, &mut rng);
                let y =
                    xs.iter().fold(DMatrix::<f64>::zeros(p, p), |acc, x| {
                        // TODO: faster way to do X * X^T?
                        acc + x * x.transpose()
                    });
                y.try_inverse().unwrap()
            })
            .collect()
    }
}

impl Support<DMatrix<f64>> for InvWishart {
    fn supports(&self, x: &DMatrix<f64>) -> bool {
        x.clone().cholesky().is_some()
    }
}

impl ContinuousDistr<DMatrix<f64>> for InvWishart {}

impl Mean<DMatrix<f64>> for InvWishart {
    fn mean(&self) -> Option<DMatrix<f64>> {
        let p = self.inv_scale.nrows();
        if self.df > p + 1 {
            Some(&self.inv_scale / (self.df - p - 1) as f64)
        } else {
            None
        }
    }
}

impl Mode<DMatrix<f64>> for InvWishart {
    fn mode(&self) -> Option<DMatrix<f64>> {
        let p = self.inv_scale.nrows();
        Some(&self.inv_scale / (self.df + p + 1) as f64)
    }
}

impl std::error::Error for InvWishartError {}

impl fmt::Display for InvWishartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DfLessThanDimensions { df, ndims } => write!(
                f,
                "df, the degrees of freedom must be greater than or \
                    equal to the number of dimensions, but {} < {}",
                df, ndims
            ),
            Self::ScaleMatrixNotSquare { nrows, ncols } => write!(
                f,
                "The scale matrix is not square: {} x {}",
                nrows, ncols
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;

    test_basic_impls!(InvWishart::identity(3));

    #[test]
    fn new_should_reject_df_too_low() {
        let inv_scale = DMatrix::identity(4, 4);
        assert!(InvWishart::new(inv_scale.clone(), 4).is_ok());
        assert!(InvWishart::new(inv_scale.clone(), 5).is_ok());
        assert_eq!(
            InvWishart::new(inv_scale.clone(), 3),
            Err(InvWishartError::DfLessThanDimensions { df: 3, ndims: 4 })
        );
    }

    #[test]
    fn new_should_reject_non_square_scale() {
        let inv_scale = DMatrix::identity(4, 3);
        assert_eq!(
            InvWishart::new(inv_scale, 5),
            Err(InvWishartError::ScaleMatrixNotSquare { nrows: 4, ncols: 3 })
        );
    }

    #[test]
    fn ln_f_standard_ident() {
        let iw = InvWishart::identity(4);
        let x = DMatrix::<f64>::identity(4, 4);
        assert::close(iw.ln_f(&x), -11.430949807317218, TOL)
    }

    #[test]
    fn ln_f_standard_mode() {
        let iw = InvWishart::identity(4);
        let x = DMatrix::<f64>::identity(4, 4) / 9.0;
        assert::close(iw.ln_f(&x), 12.11909258473473, TOL)
    }

    #[test]
    fn ln_f_nonstandard_ident() {
        let slice = vec![
            1.10576891,
            -0.20160336,
            0.09378834,
            -0.19339029,
            -0.20160336,
            0.66794786,
            -0.46020905,
            -0.62806951,
            0.09378834,
            -0.46020905,
            1.15263284,
            0.98443641,
            -0.19339029,
            -0.62806951,
            0.98443641,
            1.21050189,
        ];
        let inv_scale: DMatrix<f64> = DMatrix::from_row_slice(4, 4, &slice);
        let iw = InvWishart::new(inv_scale, 5).unwrap();
        let x = DMatrix::<f64>::identity(4, 4);
        assert::close(iw.ln_f(&x), -18.939673925150899, TOL)
    }

    #[test]
    fn draws_should_be_positive_definite() {
        let mut rng = rand::thread_rng();
        let slice = vec![
            1.10576891,
            -0.20160336,
            0.09378834,
            -0.19339029,
            -0.20160336,
            0.66794786,
            -0.46020905,
            -0.62806951,
            0.09378834,
            -0.46020905,
            1.15263284,
            0.98443641,
            -0.19339029,
            -0.62806951,
            0.98443641,
            1.21050189,
        ];
        let inv_scale: DMatrix<f64> = DMatrix::from_row_slice(4, 4, &slice);
        let iw = InvWishart::new(inv_scale, 5).unwrap();
        for x in iw.sample(100, &mut rng) {
            assert!(x.cholesky().is_some());
        }
    }

    // XXX: I've been using scipy distributionst to check my answers, but it
    // turns out that this case exposes a bug in the scipy implementation, see
    // https://github.com/scipy/scipy/issues/8844. I've tested the
    // individual terms by hand and everything looks right, so I put the
    // rust-computed value (-6.187876016819759) as the target until scipy
    // pushes the corrected code to PyPi.
    #[test]
    fn ln_f_nonstandard_mode() {
        let slice = vec![
            1.10576891,
            -0.20160336,
            0.09378834,
            -0.19339029,
            -0.20160336,
            0.66794786,
            -0.46020905,
            -0.62806951,
            0.09378834,
            -0.46020905,
            1.15263284,
            0.98443641,
            -0.19339029,
            -0.62806951,
            0.98443641,
            1.21050189,
        ];
        let inv_scale: DMatrix<f64> = DMatrix::from_row_slice(4, 4, &slice);
        let x = inv_scale.clone();
        let iw = InvWishart::new(inv_scale, 5).unwrap();
        assert::close(iw.ln_f(&x), -6.187876016819759, TOL)
    }
}
