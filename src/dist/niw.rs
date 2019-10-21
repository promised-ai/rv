#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::dist::{InvWishart, MvGaussian};
use crate::impl_display;
use crate::traits::*;
use getset::Setters;
use nalgebra::{DMatrix, DVector};
use rand::Rng;

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
#[derive(Debug, Clone, PartialEq, PartialOrd, Setters)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NormalInvWishart {
    /// The mean of μ, μ<sub>0</sub>
    mu: DVector<f64>,
    /// A scale factor on Σ, κ<sub>0</sub>
    #[set = "pub"]
    k: f64,
    /// The degrees of freedom, ν > |μ| - 1
    #[set = "pub"]
    df: usize,
    /// The positive-definite scale matrix, Ψ
    scale: DMatrix<f64>,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum Error {
    /// The k parameter is less than or equal to zero
    KTooLowError,
    /// The df parameter is less than the number of dimensions
    DfLowerThanDimensionsError,
    /// The scale matrix is not square
    ScaleMatrixNotSquareError,
    /// The dimensions of the mu vector and the scale matrix do not align
    MuScaleDimensionMismatch,
}

fn validate_params<'a>(
    mu: &DVector<f64>,
    k: f64,
    df: usize,
    scale: &DMatrix<f64>,
) -> Result<(), Error> {
    let dims = mu.len();
    if k <= 0.0 {
        Err(Error::KTooLowError)
    } else if df < dims {
        Err(Error::DfLowerThanDimensionsError)
    } else if !scale.is_square() {
        Err(Error::ScaleMatrixNotSquareError)
    } else if dims != scale.nrows() {
        Err(Error::MuScaleDimensionMismatch)
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
    pub fn new(
        mu: DVector<f64>,
        k: f64,
        df: usize,
        scale: DMatrix<f64>,
    ) -> Result<Self, Error> {
        validate_params(&mu, k, df, &scale)?;
        Ok(NormalInvWishart { mu, k, df, scale })
    }

    /// Creates a new NormalInvWishart without checking whether the parameters
    /// are valid.
    pub fn new_unchecked(
        mu: DVector<f64>,
        k: f64,
        df: usize,
        scale: DMatrix<f64>,
    ) -> Self {
        NormalInvWishart { mu, k, df, scale }
    }

    /// Get the number of dimensions
    pub fn dims(&self) -> usize {
        self.mu.len()
    }

    /// Get a reference to the mu vector
    pub fn mu(&self) -> &DVector<f64> {
        &self.mu
    }

    /// Get the k parameter
    pub fn k(&self) -> f64 {
        self.k
    }

    /// Get the degrees of freedom, df
    pub fn df(&self) -> usize {
        self.df
    }

    /// Get a reference to the scale matrix
    pub fn scale(&self) -> &DMatrix<f64> {
        &self.scale
    }

    /// Set the scale parameter
    pub fn set_scale(&mut self, scale: DMatrix<f64>) -> Result<(), Error> {
        validate_params(&self.mu, self.k, self.df, &scale)?;
        self.scale = scale;
        Ok(())
    }

    pub fn set_scale_unnchecked(&mut self, scale: DMatrix<f64>) {
        self.scale = scale;
    }

    /// Set the scale parameter
    pub fn set_mu(&mut self, mu: DVector<f64>) -> Result<(), Error> {
        validate_params(&mu, self.k, self.df, &self.scale)?;
        self.mu = mu;
        Ok(())
    }

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
        let sigma = x.cov().to_owned() / self.k;
        let mvg = MvGaussian::new(m, sigma).unwrap();
        let iw = InvWishart::new(self.scale.clone(), self.df).unwrap();
        mvg.ln_f(x.mu()) + iw.ln_f(x.cov())
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> MvGaussian {
        let iw = InvWishart::new(self.scale.clone(), self.df).unwrap();
        let sigma = iw.draw(&mut rng);

        let mvg =
            MvGaussian::new(self.mu.clone(), sigma.clone() / self.k).unwrap();
        let mu = mvg.draw(&mut rng);

        MvGaussian::new(mu, sigma).unwrap()
    }
}

impl Support<MvGaussian> for NormalInvWishart {
    fn supports(&self, x: &MvGaussian) -> bool {
        let p = self.mu.len();
        x.mu().len() == p && x.cov().to_owned().cholesky().is_some()
    }
}

impl ContinuousDistr<MvGaussian> for NormalInvWishart {}
