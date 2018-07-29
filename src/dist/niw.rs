extern crate nalgebra;
extern crate rand;

use self::nalgebra::{DMatrix, DVector};
use self::rand::Rng;
use dist::{InvWishart, MvGaussian};
use std::io;
use traits::*;

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
/// # extern crate rv;
/// extern crate rand;
/// extern crate nalgebra;
///
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
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NormalInvWishart {
    /// The mean of μ, μ<sub>0</sub>
    pub mu: DVector<f64>,
    /// A scale factor on Σ, κ<sub>0</sub>
    pub k: f64,
    /// The degrees of freedom, ν > |μ| - 1
    pub df: usize,
    /// The positive-definite scale matrix, Ψ
    pub scale: DMatrix<f64>,
}

impl NormalInvWishart {
    /// Create a new `NormalInvWishart` distribution
    pub fn new(
        mu: DVector<f64>,
        k: f64,
        df: usize,
        scale: DMatrix<f64>,
    ) -> io::Result<Self> {
        let dims = mu.len();
        let err = if k <= 0.0 {
            Some("k must be > 0.0")
        } else if df < dims {
            Some("df must be >= the number of dimensions")
        } else if !scale.is_square() {
            Some("scale must be square")
        } else if dims != scale.nrows() {
            Some("dimensions in mu and scale must match")
        } else if scale.clone().cholesky().is_none() {
            Some("scale is not positive definite")
        } else {
            None
        };

        match err {
            Some(msg) => Err(io::Error::new(io::ErrorKind::InvalidInput, msg)),
            None => Ok(NormalInvWishart { mu, k, df, scale }),
        }
    }
}

// TODO: We might be able to make things faster by storing the InvWishart
// because each time we create it, it clones and validates the parameters.
impl Rv<MvGaussian> for NormalInvWishart {
    fn ln_f(&self, x: &MvGaussian) -> f64 {
        let m = self.mu.clone();
        let sigma = x.cov.clone() / self.k;
        let mvg = MvGaussian::new(m, sigma).unwrap();
        let iw = InvWishart::new(self.scale.clone(), self.df).unwrap();
        mvg.ln_f(&x.mu) + iw.ln_f(&x.cov)
    }

    fn ln_normalizer(&self) -> f64 {
        0.0
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
    fn contains(&self, x: &MvGaussian) -> bool {
        let p = self.mu.len();
        x.mu.len() == p && x.cov.clone().cholesky().is_some()
    }
}

impl ContinuousDistr<MvGaussian> for NormalInvWishart {}
