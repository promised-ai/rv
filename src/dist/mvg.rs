#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::data::MvGaussianSuffStat;
use crate::impl_display;
use crate::traits::*;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::f64::consts::{E, PI};

/// [Multivariate Gaussian/Normal Distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution),
/// 𝒩(μ, Σ).
///
/// # Example
///
/// Generate a Wishart random 3x3 matrix **Σ** ~ W<sub>ν</sub>(S)
///
/// ```
/// use nalgebra::{DMatrix, DVector};
/// use rv::prelude::*;
///
/// let k = 3;   // number of dimensions
/// let df = 6;  // The degrees of freedom is an unsigned int > k
///
/// // The scale matrix of the wishar distribution
/// let scale_mat: DMatrix<f64> = DMatrix::identity(k, k);
///
/// // The draw procedure outlined in the appendices of "Bayesian Data
/// // Analysis" by Andrew Gelman and colleagues.
/// let mut rng = rand::thread_rng();
///
/// // 1. Create a multivariate normal with zero mean and covariance matrix S
/// let mvg = MvGaussian::new(DVector::zeros(k), scale_mat).unwrap();
///
/// // 2. Draw ν (df) vectors {x_1, ..., x_ν}
/// let xs = mvg.sample(df, &mut rng);
///
/// // 3. Compute the sum Σ xx'
/// let mat = xs.iter().fold(DMatrix::<f64>::zeros(k, k), |acc, x| {
///     acc +x*x.transpose()
/// });
///
/// // Check that the matrix is square and has the right size
/// assert_eq!(mat.nrows(), k);
/// assert_eq!(mat.ncols(), k);
///
/// // check that the matrix is positive definite by attempting a Cholesky
/// // decomposition.
/// assert!(mat.cholesky().is_some());
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct MvGaussian {
    // Mean vector
    mu: DVector<f64>,
    // Covariance matrix
    cov: DMatrix<f64>,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum Error {
    /// The mu and cov parameters have incompatible dimensions
    MuCovDimensionMismatchError,
    /// The cov matrix is not square
    CovNotSquareError,
    /// Cov is not a positive semi-definite matrix
    CovNotPositiveSemiDefiniteError,
    /// Requested dimension is too low
    ZeroDimensionError,
}

impl MvGaussian {
    /// Create a new multivariate Gaussian distribution
    ///
    /// # Arguments
    /// - mu: k-length mean vector
    /// - cov: k-by-k positive-definite covariance matrix
    pub fn new(mu: DVector<f64>, cov: DMatrix<f64>) -> Result<Self, Error> {
        if cov.nrows() != cov.ncols() {
            Err(Error::CovNotSquareError)
        } else if mu.len() != cov.nrows() {
            Err(Error::MuCovDimensionMismatchError)
        } else {
            Ok(MvGaussian { mu, cov })
        }
    }

    /// Creates a new MvGaussian without checking whether the parameters are
    /// valid.
    pub fn new_unchecked(mu: DVector<f64>, cov: DMatrix<f64>) -> Self {
        MvGaussian { mu, cov }
    }

    /// Create a standard Gaussian distribution with zero mean and identiry
    /// covariance matrix.
    pub fn standard(dims: usize) -> Result<Self, Error> {
        if dims == 0 {
            Err(Error::ZeroDimensionError)
        } else {
            let mu = DVector::zeros(dims);
            let cov = DMatrix::identity(dims, dims);
            Ok(MvGaussian { mu, cov })
        }
    }

    /// Get the number of dimensions
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::MvGaussian;
    /// let mvg = MvGaussian::standard(4).unwrap();
    /// assert_eq!(mvg.dims(), 4);
    /// ```
    pub fn dims(&self) -> usize {
        self.mu.len()
    }

    /// Get a reference to the mean
    pub fn mu(&self) -> &DVector<f64> {
        &self.mu
    }

    /// Get a reference to the covariance
    pub fn cov(&self) -> &DMatrix<f64> {
        &self.cov
    }

    /// Set the mean
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::prelude::*;
    /// # use nalgebra::{DVector, DMatrix};
    /// let mut mvg = MvGaussian::standard(3).unwrap();
    /// let x = DVector::<f64>::zeros(3);
    ///
    /// assert::close(mvg.ln_f(&x), -2.756815599614018, 1E-8);
    ///
    /// let cov_vals = vec![
    ///     1.01742788,
    ///     0.36586652,
    ///     -0.65620486,
    ///     0.36586652,
    ///     1.00564553,
    ///     -0.42597261,
    ///     -0.65620486,
    ///     -0.42597261,
    ///     1.27247972,
    /// ];
    /// let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
    /// let mu = DVector::<f64>::from_column_slice(&vec![0.5, 3.1, -6.2]);
    ///
    /// mvg.set_mu(mu).unwrap();
    /// mvg.set_cov(cov).unwrap();
    ///
    /// assert::close(mvg.ln_f(&x), -24.602370253215661, 1E-8);
    /// ```
    pub fn set_mu(&mut self, mu: DVector<f64>) -> Result<(), Error> {
        if mu.len() != self.cov.nrows() {
            Err(Error::MuCovDimensionMismatchError)
        } else {
            self.mu = mu;
            Ok(())
        }
    }

    pub fn set_mu_unchecked(&mut self, mu: DVector<f64>) {
        self.mu = mu;
    }

    /// Set the covariance matrix
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::prelude::*;
    /// # use nalgebra::{DVector, DMatrix};
    /// let mut mvg = MvGaussian::standard(3).unwrap();
    /// let x = DVector::<f64>::zeros(3);
    ///
    /// assert::close(mvg.ln_f(&x), -2.756815599614018, 1E-8);
    ///
    /// let cov_vals = vec![
    ///     1.01742788,
    ///     0.36586652,
    ///     -0.65620486,
    ///     0.36586652,
    ///     1.00564553,
    ///     -0.42597261,
    ///     -0.65620486,
    ///     -0.42597261,
    ///     1.27247972,
    /// ];
    /// let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
    /// let mu = DVector::<f64>::from_column_slice(&vec![0.5, 3.1, -6.2]);
    ///
    /// mvg.set_mu(mu).unwrap();
    /// mvg.set_cov(cov).unwrap();
    ///
    /// assert::close(mvg.ln_f(&x), -24.602370253215661, 1E-8);
    /// ```
    pub fn set_cov(&mut self, cov: DMatrix<f64>) -> Result<(), Error> {
        if self.mu.len() != cov.nrows() {
            Err(Error::MuCovDimensionMismatchError)
        } else if cov.nrows() != cov.ncols() {
            Err(Error::CovNotSquareError)
        } else {
            self.cov = cov;
            Ok(())
        }
    }

    pub fn set_cov_unchecked(&mut self, cov: DMatrix<f64>) {
        self.cov = cov;
    }
}

impl From<&MvGaussian> for String {
    fn from(mvg: &MvGaussian) -> String {
        format!("Nₖ({})\n  μ: {}\n  σ: {})", mvg.dims(), mvg.mu, mvg.cov)
    }
}

impl_display!(MvGaussian);

impl Rv<DVector<f64>> for MvGaussian {
    fn ln_f(&self, x: &DVector<f64>) -> f64 {
        let diff = x - &self.mu;
        let det: f64 = (2.0 * PI * &self.cov).determinant();
        let inv = self
            .cov
            .clone()
            .try_inverse()
            .expect("Failed to invert cov");
        let term: f64 = (0.5 * diff.transpose() * inv * diff)[0];
        -0.5 * det.ln() - term
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> DVector<f64> {
        let dims = self.mu.len();
        let norm = rand_distr::StandardNormal;
        let vals: Vec<f64> = (0..dims).map(|_| rng.sample(norm)).collect();

        let a: DMatrix<f64> = self
            .cov
            .clone()
            .cholesky()
            .expect("Cholesky decomp failed")
            .unpack();
        let z: DVector<f64> = DVector::from_column_slice(&vals);

        self.mu.clone() + a * z
    }
}

impl Support<DVector<f64>> for MvGaussian {
    fn supports(&self, x: &DVector<f64>) -> bool {
        x.len() == self.mu.len()
    }
}

impl ContinuousDistr<DVector<f64>> for MvGaussian {}

impl Mean<DVector<f64>> for MvGaussian {
    fn mean(&self) -> Option<DVector<f64>> {
        Some(self.mu.clone())
    }
}

impl Mode<DVector<f64>> for MvGaussian {
    fn mode(&self) -> Option<DVector<f64>> {
        Some(self.mu.clone())
    }
}

impl Variance<DMatrix<f64>> for MvGaussian {
    fn variance(&self) -> Option<DMatrix<f64>> {
        Some(self.cov.clone())
    }
}

impl Entropy for MvGaussian {
    fn entropy(&self) -> f64 {
        0.5 * (2.0 * PI * E * &self.cov).determinant().ln()
    }
}
impl HasSuffStat<DVector<f64>> for MvGaussian {
    type Stat = MvGaussianSuffStat;
    fn empty_suffstat(&self) -> Self::Stat {
        MvGaussianSuffStat::new(self.mu.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Gaussian;
    use crate::misc::{ks_test, mardia};

    const TOL: f64 = 1E-12;
    const NTRIES: usize = 5;
    const KS_PVAL: f64 = 0.2;
    const MARDIA_PVAL: f64 = 0.2;

    #[test]
    fn new() {
        let mu = DVector::zeros(3);
        let cov = DMatrix::identity(3, 3);
        assert!(MvGaussian::new(mu, cov).is_ok());
    }

    #[test]
    fn new_should_reject_cov_too_big() {
        let mu = DVector::zeros(3);
        let cov = DMatrix::identity(4, 4);
        let mvg = MvGaussian::new(mu, cov);

        assert_eq!(mvg, Err(Error::MuCovDimensionMismatchError))
    }

    #[test]
    fn new_should_reject_cov_too_small() {
        let mu = DVector::zeros(3);
        let cov = DMatrix::identity(2, 2);
        let mvg = MvGaussian::new(mu, cov);
        assert_eq!(mvg, Err(Error::MuCovDimensionMismatchError))
    }

    #[test]
    fn new_should_reject_cov_not_square() {
        let mu = DVector::zeros(3);
        let cov = DMatrix::identity(3, 2);
        let mvg = MvGaussian::new(mu, cov);
        assert_eq!(mvg, Err(Error::CovNotSquareError));
    }

    #[test]
    fn ln_f_standard_x_zeros() {
        let mvg = MvGaussian::standard(3).unwrap();
        let x = DVector::<f64>::zeros(3);
        assert::close(mvg.ln_f(&x), -2.756815599614018, TOL);
    }

    #[test]
    fn ln_f_standard_x_nonzeros() {
        let mvg = MvGaussian::standard(3).unwrap();
        let x = DVector::<f64>::from_column_slice(&vec![0.5, 3.1, -6.2]);
        assert::close(mvg.ln_f(&x), -26.906815599614021, TOL);
    }

    #[test]
    fn ln_f_nonstandard_zeros() {
        let cov_vals = vec![
            1.01742788,
            0.36586652,
            -0.65620486,
            0.36586652,
            1.00564553,
            -0.42597261,
            -0.65620486,
            -0.42597261,
            1.27247972,
        ];
        let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
        let mu = DVector::<f64>::from_column_slice(&vec![0.5, 3.1, -6.2]);
        let mvg = MvGaussian::new(mu, cov).unwrap();
        let x = DVector::<f64>::zeros(3);
        assert::close(mvg.ln_f(&x), -24.602370253215661, TOL);
    }

    #[test]
    fn ln_f_nonstandard_nonzeros() {
        let cov_vals = vec![
            1.01742788,
            0.36586652,
            -0.65620486,
            0.36586652,
            1.00564553,
            -0.42597261,
            -0.65620486,
            -0.42597261,
            1.27247972,
        ];
        let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
        let mu = DVector::<f64>::from_column_slice(&vec![0.5, 3.1, -6.2]);
        let mvg = MvGaussian::new(mu, cov).unwrap();
        let x = DVector::<f64>::from_column_slice(&vec![0.5, 3.1, -6.2]);
        assert::close(mvg.ln_f(&x), -2.5915350538112296, TOL);
    }

    #[test]
    fn sample_returns_proper_number_of_draws() {
        let cov_vals = vec![
            1.01742788,
            0.36586652,
            -0.65620486,
            0.36586652,
            1.00564553,
            -0.42597261,
            -0.65620486,
            -0.42597261,
            1.27247972,
        ];
        let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
        let mu = DVector::<f64>::from_column_slice(&vec![0.5, 3.1, -6.2]);
        let mvg = MvGaussian::new(mu, cov).unwrap();

        let mut rng = rand::thread_rng();

        let xs = mvg.sample(103, &mut rng);

        assert_eq!(xs.len(), 103);
    }

    #[test]
    fn standard_entropy() {
        let mvg = MvGaussian::standard(3).unwrap();
        assert::close(mvg.entropy(), 4.2568155996140185, TOL);
    }

    #[test]
    fn nonstandard_entropy() {
        let cov_vals = vec![
            1.01742788,
            0.36586652,
            -0.65620486,
            0.36586652,
            1.00564553,
            -0.42597261,
            -0.65620486,
            -0.42597261,
            1.27247972,
        ];
        let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
        let mu = DVector::<f64>::from_column_slice(&vec![0.5, 3.1, -6.2]);
        let mvg = MvGaussian::new(mu, cov).unwrap();
        assert::close(mvg.entropy(), 4.0915350538112305, TOL);
    }

    #[test]
    fn standard_draw_marginals() {
        let mut rng = rand::thread_rng();
        let mvg = MvGaussian::standard(2).unwrap();

        let g = Gaussian::standard();
        let cdf = |x: f64| g.cdf(&x);

        let passed = (0..NTRIES).fold(false, |acc, _| {
            if acc {
                acc
            } else {
                let xys = mvg.sample(500, &mut rng);
                let xs: Vec<f64> = xys.iter().map(|xy| xy[0].clone()).collect();
                let ys: Vec<f64> = xys.iter().map(|xy| xy[1].clone()).collect();

                let (_, px) = ks_test(&xs, cdf);
                let (_, py) = ks_test(&ys, cdf);
                px > KS_PVAL && py > KS_PVAL
            }
        });

        assert!(passed);
    }

    #[test]
    fn standard_draw_mardia() {
        let mut rng = rand::thread_rng();
        let mvg = MvGaussian::standard(4).unwrap();

        let passed = (0..NTRIES).fold(false, |acc, _| {
            if acc {
                acc
            } else {
                let xys = mvg.sample(500, &mut rng);
                let (pa, pb) = mardia(&xys);
                pa > MARDIA_PVAL && pb > MARDIA_PVAL
            }
        });

        assert!(passed);
    }

    #[test]
    fn nonstandard_draw_mardia() {
        let mut rng = rand::thread_rng();
        let cov_vals = vec![
            1.01742788,
            0.36586652,
            -0.65620486,
            0.36586652,
            1.00564553,
            -0.42597261,
            -0.65620486,
            -0.42597261,
            1.27247972,
        ];
        let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
        let mu = DVector::<f64>::from_column_slice(&vec![0.5, 3.1, -6.2]);
        let mvg = MvGaussian::new(mu, cov).unwrap();

        let passed = (0..NTRIES).fold(false, |acc, _| {
            if acc {
                acc
            } else {
                let xys = mvg.sample(500, &mut rng);
                let (pa, pb) = mardia(&xys);
                pa > MARDIA_PVAL && pb > MARDIA_PVAL
            }
        });

        assert!(passed);
    }
}
