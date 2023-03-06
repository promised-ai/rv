#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::dist::{InvWishart, MvGaussian};
use crate::impl_display;
use crate::traits::*;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::fmt;

mod mvg_prior;

/// Common conjugate prior on the μ and Σ parameters in the Multivariate
/// Gaussian, Ν(μ, Σ)
///
/// Ν(μ, Σ) ~ NIW(μ<sub>0</sub>, κ<sub>0</sub>, ν, Ψ) implies
/// μ ~ N(μ<sub>0</sub>, Σ/k<sub>0</sub>) and
/// Σ ~ W<sup>-1</sup>(Ψ, ν)
///
/// # Example
///
/// Draw a Multivariate Gaussian from GIW
///
/// ```
/// use nalgebra::{DMatrix, DVector};
/// use rv::prelude::*;
///
/// let mu = DVector::zeros(3);
/// let k = 1.0;
/// let df = 3;
/// let scale = DMatrix::identity(3, 3);
///
/// let niw = NormalInvWishart::new(mu, k, df, scale).unwrap();
///
/// let mut rng = rand::thread_rng();
///
/// let mvg: MvGaussian = niw.draw(&mut rng);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NormalInvWishart {
    /// The mean of μ, μ<sub>0</sub>
    mu: DVector<f64>,
    /// A scale factor on Σ, κ<sub>0</sub>
    k: f64,
    /// The degrees of freedom, ν > |μ| - 1
    df: usize,
    /// The positive-definite scale matrix, Ψ
    scale: DMatrix<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum NormalInvWishartError {
    /// The k parameter is less than or equal to zero
    KTooLow { k: f64 },
    /// The df parameter is less than the number of dimensions
    DfLessThanDimensions { df: usize, ndims: usize },
    /// The scale matrix is not square
    ScaleMatrixNotSquare {
        /// number of row
        nrows: usize,
        /// number of columns
        ncols: usize,
    },
    /// The dimensions of the mu vector and the scale matrix do not align
    MuScaleDimensionMismatch {
        /// Number of dimensions in the mean vector
        n_mu: usize,
        /// Number of dimensions in the scale matrix
        n_scale: usize,
    },
}

fn validate_params(
    mu: &DVector<f64>,
    k: f64,
    df: usize,
    scale: &DMatrix<f64>,
) -> Result<(), NormalInvWishartError> {
    let ndims = mu.len();
    if k <= 0.0 {
        Err(NormalInvWishartError::KTooLow { k })
    } else if df < ndims {
        Err(NormalInvWishartError::DfLessThanDimensions { df, ndims })
    } else if !scale.is_square() {
        Err(NormalInvWishartError::ScaleMatrixNotSquare {
            nrows: scale.nrows(),
            ncols: scale.ncols(),
        })
    } else if ndims != scale.nrows() {
        Err(NormalInvWishartError::MuScaleDimensionMismatch {
            n_mu: ndims,
            n_scale: scale.nrows(),
        })
    } else {
        Ok(())
    }
}

impl NormalInvWishart {
    /// Create a new `NormalInvWishart` distribution
    ///
    /// # Arguments
    /// - mu: The mean of μ, μ<sub>0</sub>
    /// - k: A scale factor on Σ, κ<sub>0</sub>
    /// - df: The degrees of freedom, ν > |μ| - 1
    /// - scale The positive-definite scale matrix, Ψ
    #[inline]
    pub fn new(
        mu: DVector<f64>,
        k: f64,
        df: usize,
        scale: DMatrix<f64>,
    ) -> Result<Self, NormalInvWishartError> {
        validate_params(&mu, k, df, &scale)?;
        Ok(NormalInvWishart { mu, k, df, scale })
    }

    /// Creates a new NormalInvWishart without checking whether the parameters
    /// are valid.
    #[inline]
    pub fn new_unchecked(
        mu: DVector<f64>,
        k: f64,
        df: usize,
        scale: DMatrix<f64>,
    ) -> Self {
        NormalInvWishart { mu, k, df, scale }
    }

    /// Get the number of dimensions
    #[inline]
    pub fn ndims(&self) -> usize {
        self.mu.len()
    }

    /// Get a reference to the mu vector
    #[inline]
    pub fn mu(&self) -> &DVector<f64> {
        &self.mu
    }

    /// Get the k parameter
    #[inline]
    pub fn k(&self) -> f64 {
        self.k
    }

    /// Set the value of k
    #[inline]
    pub fn set_k(&mut self, k: f64) -> Result<(), NormalInvWishartError> {
        if k <= 0.0 {
            Err(NormalInvWishartError::KTooLow { k })
        } else {
            self.k = k;
            Ok(())
        }
    }

    /// Set the value of k without input validation
    #[inline]
    pub fn set_k_unchecked(&mut self, k: f64) {
        self.k = k;
    }

    /// Get the degrees of freedom, df
    #[inline]
    pub fn df(&self) -> usize {
        self.df
    }

    /// Set the value of df
    #[inline]
    pub fn set_df(&mut self, df: usize) -> Result<(), NormalInvWishartError> {
        let ndims = self.ndims();
        if df < ndims {
            Err(NormalInvWishartError::DfLessThanDimensions { df, ndims })
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

    /// Get a reference to the scale matrix
    #[inline]
    pub fn scale(&self) -> &DMatrix<f64> {
        &self.scale
    }

    /// Set the scale parameter
    #[inline]
    pub fn set_scale(
        &mut self,
        scale: DMatrix<f64>,
    ) -> Result<(), NormalInvWishartError> {
        validate_params(&self.mu, self.k, self.df, &scale)?;
        self.scale = scale;
        Ok(())
    }

    #[inline]
    pub fn set_scale_unnchecked(&mut self, scale: DMatrix<f64>) {
        self.scale = scale;
    }

    /// Set the scale parameter
    #[inline]
    pub fn set_mu(
        &mut self,
        mu: DVector<f64>,
    ) -> Result<(), NormalInvWishartError> {
        validate_params(&mu, self.k, self.df, &self.scale)?;
        self.mu = mu;
        Ok(())
    }

    #[inline]
    pub fn set_mu_unchecked(&mut self, mu: DVector<f64>) {
        self.mu = mu;
    }
}

impl From<&NormalInvWishart> for String {
    fn from(niw: &NormalInvWishart) -> String {
        format!(
            "NIW (\n μ: {}\n κ: {}\n ν: {}\n Ψ: {}",
            niw.mu, niw.k, niw.df, niw.scale
        )
    }
}

impl_display!(NormalInvWishart);

// TODO: We might be able to make things faster by storing the InvWishart
// because each time we create it, it clones and validates the parameters.
impl Rv<MvGaussian> for NormalInvWishart {
    fn ln_f(&self, x: &MvGaussian) -> f64 {
        let m = self.mu.clone();
        let sigma = x.cov().clone() / self.k;
        // TODO: cache mvg and iw instead of cloning
        let mvg = MvGaussian::new_unchecked(m, sigma);
        let iw = InvWishart::new_unchecked(self.scale.clone(), self.df);
        mvg.ln_f(x.mu()) + iw.ln_f(x.cov())
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> MvGaussian {
        let iw = InvWishart::new_unchecked(self.scale.clone(), self.df);
        let sigma: DMatrix<f64> = iw.draw(&mut rng);

        let mvg =
            MvGaussian::new_unchecked(self.mu.clone(), sigma.clone() / self.k);
        let mu = mvg.draw(&mut rng);

        MvGaussian::new(mu, sigma).unwrap()
    }
}

impl Support<MvGaussian> for NormalInvWishart {
    fn supports(&self, x: &MvGaussian) -> bool {
        let p = self.mu.len();
        x.mu().len() == p && x.cov().clone().cholesky().is_some()
    }
}

impl ContinuousDistr<MvGaussian> for NormalInvWishart {}

impl std::error::Error for NormalInvWishartError {}

impl fmt::Display for NormalInvWishartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KTooLow { k } => {
                write!(f, "k ({}) must be greater than zero", k)
            }
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
            Self::MuScaleDimensionMismatch { n_mu, n_scale } => write!(
                f,
                "The mu vector (nrows = {}) must have the same \
                    number of entries as the scale matrix has columns/rows \
                    (ndims = {}). ",
                n_mu, n_scale
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disallow_zero_k() {
        let mu = DVector::zeros(2);
        let scale = DMatrix::identity(2, 2);
        let res = NormalInvWishart::new(mu, 0.0, 2, scale);
        if let Err(NormalInvWishartError::KTooLow { .. }) = res {
        } else {
            panic!("wrong error");
        }
    }

    #[test]
    fn disallow_negative_k() {
        let mu = DVector::zeros(2);
        let scale = DMatrix::identity(2, 2);
        let res = NormalInvWishart::new(mu, -1.0, 2, scale);
        if let Err(NormalInvWishartError::KTooLow { .. }) = res {
        } else {
            panic!("wrong error");
        }
    }

    #[test]
    fn disallow_df_less_than_n_dims() {
        let mu = DVector::zeros(2);
        let scale = DMatrix::identity(2, 2);
        let res = NormalInvWishart::new(mu, 1.0, 1, scale);
        if let Err(NormalInvWishartError::DfLessThanDimensions {
            df: 1,
            ndims: 2,
        }) = res
        {
        } else {
            panic!("wrong error");
        }
    }

    #[test]
    fn disallow_mu_and_sigma_different_dims() {
        let mu = DVector::zeros(2);
        let scale = DMatrix::identity(3, 3);
        let res = NormalInvWishart::new(mu, 1.0, 4, scale);
        if let Err(NormalInvWishartError::MuScaleDimensionMismatch {
            n_mu: 2,
            n_scale: 3,
        }) = res
        {
        } else {
            panic!("wrong error");
        }
    }

    #[test]
    fn disallow_non_scale_square() {
        let mu = DVector::zeros(2);
        let scale = DMatrix::identity(2, 3);
        let res = NormalInvWishart::new(mu, 1.0, 3, scale);
        if let Err(NormalInvWishartError::ScaleMatrixNotSquare {
            nrows: 2,
            ncols: 3,
        }) = res
        {
        } else {
            panic!("wrong error");
        }
    }
}
