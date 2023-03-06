#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::consts::HALF_LN_2PI_E;
use crate::consts::LN_2PI;
use crate::data::MvGaussianSuffStat;
use crate::impl_display;
use crate::traits::*;
use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, Dyn};
use once_cell::sync::OnceCell;
use rand::Rng;
use std::fmt;

/// Cache for MvGaussian Internals
#[derive(Clone, Debug)]
struct MvgCache {
    /// Covariant Matrix Cholesky Decomposition
    pub cov_chol: Cholesky<f64, Dyn>,
    /// Inverse of Covariance Matrix
    pub cov_inv: DMatrix<f64>,
}

impl MvgCache {
    pub fn from_cov(cov: &DMatrix<f64>) -> Result<Self, MvGaussianError> {
        match cov.clone().cholesky() {
            None => Err(MvGaussianError::CovNotPositiveSemiDefinite),
            Some(cov_chol) => {
                let cov_inv = cov_chol.inverse();
                Ok(MvgCache { cov_chol, cov_inv })
            }
        }
    }

    #[inline]
    pub fn from_chol(cov_chol: Cholesky<f64, Dyn>) -> Self {
        let cov_inv = cov_chol.inverse();
        MvgCache { cov_chol, cov_inv }
    }

    #[inline]
    pub fn cov(&self) -> DMatrix<f64> {
        let l = self.cov_chol.l();
        &l * &l.transpose()
    }
}

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
/// let mat = xs
///     .iter()
///     .fold(DMatrix::<f64>::zeros(k, k), |acc, x: &DVector<f64>| {
///         acc +x*x.transpose()
///     });
///
/// // Check that the matrix is square and has the right size
/// assert_eq!(mat.nrows(), k);
/// assert_eq!(mat.ncols(), k);
///
/// // check that the matrix is positive definite by attempting a Cholesky
/// // decomposition.
/// assert!(mat.cholesky().is_some());
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct MvGaussian {
    // Mean vector
    mu: DVector<f64>,
    // Covariance Matrix
    cov: DMatrix<f64>,
    // Cached values for computations
    #[cfg_attr(
        feature = "serde1",
        serde(skip, default = "default_cache_none")
    )]
    cache: OnceCell<MvgCache>,
}

#[allow(dead_code)]
#[cfg(feature = "serde1")]
fn default_cache_none() -> OnceCell<MvgCache> {
    OnceCell::new()
}

impl PartialEq for MvGaussian {
    fn eq(&self, other: &MvGaussian) -> bool {
        self.mu == other.mu && self.cov == other.cov
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum MvGaussianError {
    /// The mu and cov parameters have incompatible dimensions
    MuCovDimensionMismatch {
        /// Length of the mu vector
        n_mu: usize,
        /// Number of dimensions of the covariance matrix
        n_cov: usize,
    },
    /// The cov matrix is not square
    CovNotSquare {
        /// Number of rows in the covariance matrix
        nrows: usize,
        /// Number of columns in the covariance matrix
        ncols: usize,
    },
    /// Cov is not a positive semi-definite matrix
    CovNotPositiveSemiDefinite,
    /// Requested dimension is too low
    ZeroDimension,
}

impl MvGaussian {
    /// Create a new multivariate Gaussian distribution
    ///
    /// # Arguments
    /// - mu: k-length mean vector
    /// - cov: k-by-k positive-definite covariance matrix
    pub fn new(
        mu: DVector<f64>,
        cov: DMatrix<f64>,
    ) -> Result<Self, MvGaussianError> {
        let cov_rows = cov.nrows();
        let cov_cols = cov.ncols();
        if cov_rows != cov_cols {
            Err(MvGaussianError::CovNotSquare {
                nrows: cov_rows,
                ncols: cov_cols,
            })
        } else if mu.len() != cov_rows {
            Err(MvGaussianError::MuCovDimensionMismatch {
                n_mu: mu.len(),
                n_cov: cov_rows,
            })
        } else {
            let cache = OnceCell::from(MvgCache::from_cov(&cov)?);
            Ok(MvGaussian { mu, cov, cache })
        }
    }

    /// Create a new multivariate Gaussian distribution from Cholesky factorized Dov
    ///
    /// # Arguments
    /// - mu: k-length mean vector
    /// - cov_chol: Choleksy decomposition of k-by-k positive-definite covariance matrix
    /// ```rust
    /// use nalgebra::{DMatrix, DVector};
    /// use rv::prelude::*;
    ///
    /// let mu = DVector::zeros(3);
    /// let cov = DMatrix::from_row_slice(3, 3, &[
    ///     2.0, 1.0, 0.0,
    ///     1.0, 2.0, 0.0,
    ///     0.0, 0.0, 1.0
    /// ]);
    ///
    /// let chol = cov.clone().cholesky().unwrap();
    /// let mvg_r = MvGaussian::new_cholesky(mu, chol);
    ///
    /// assert!(mvg_r.is_ok());
    /// let mvg = mvg_r.unwrap();
    /// assert!(cov.relative_eq(mvg.cov(), 1E-8, 1E-8));
    /// ```
    pub fn new_cholesky(
        mu: DVector<f64>,
        cov_chol: Cholesky<f64, Dyn>,
    ) -> Result<Self, MvGaussianError> {
        let l = cov_chol.l();
        let cov = &l * &l.transpose();
        if mu.len() != cov.nrows() {
            Err(MvGaussianError::MuCovDimensionMismatch {
                n_mu: mu.len(),
                n_cov: cov.nrows(),
            })
        } else {
            let cache = OnceCell::from(MvgCache::from_chol(cov_chol));
            Ok(MvGaussian { mu, cov, cache })
        }
    }

    /// Creates a new MvGaussian from mean and covariance without checking
    /// whether the parameters are valid.
    #[inline]
    pub fn new_unchecked(mu: DVector<f64>, cov: DMatrix<f64>) -> Self {
        let cache = OnceCell::from(MvgCache::from_cov(&cov).unwrap());
        MvGaussian { mu, cov, cache }
    }

    /// Creates a new MvGaussian from mean and covariance's Cholesky factorization
    /// without checking whether the parameters are valid.
    #[inline]
    pub fn new_cholesky_unchecked(
        mu: DVector<f64>,
        cov_chol: Cholesky<f64, Dyn>,
    ) -> Self {
        let cache = OnceCell::from(MvgCache::from_chol(cov_chol));
        let cov = cache.get().unwrap().cov();
        MvGaussian { mu, cov, cache }
    }

    /// Create a standard Gaussian distribution with zero mean and identity
    /// covariance matrix.
    #[inline]
    pub fn standard(dims: usize) -> Result<Self, MvGaussianError> {
        if dims == 0 {
            Err(MvGaussianError::ZeroDimension)
        } else {
            let mu = DVector::zeros(dims);
            let cov = DMatrix::identity(dims, dims);
            let cov_chol = cov.clone().cholesky().unwrap();
            let cache = OnceCell::from(MvgCache::from_chol(cov_chol));
            Ok(MvGaussian { mu, cov, cache })
        }
    }

    /// Get the number of dimensions
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::MvGaussian;
    /// let mvg = MvGaussian::standard(4).unwrap();
    /// assert_eq!(mvg.ndims(), 4);
    /// ```
    #[inline]
    pub fn ndims(&self) -> usize {
        self.mu.len()
    }

    /// Get a reference to the mean
    #[inline]
    pub fn mu(&self) -> &DVector<f64> {
        &self.mu
    }

    /// Get a reference to the covariance
    #[inline]
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
    #[inline]
    pub fn set_mu(&mut self, mu: DVector<f64>) -> Result<(), MvGaussianError> {
        if mu.len() != self.cov.nrows() {
            Err(MvGaussianError::MuCovDimensionMismatch {
                n_mu: mu.len(),
                n_cov: self.cov.nrows(),
            })
        } else {
            self.mu = mu;
            Ok(())
        }
    }

    #[inline]
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
    pub fn set_cov(
        &mut self,
        cov: DMatrix<f64>,
    ) -> Result<(), MvGaussianError> {
        let cov_rows = cov.nrows();
        if self.mu.len() != cov_rows {
            Err(MvGaussianError::MuCovDimensionMismatch {
                n_mu: self.mu.len(),
                n_cov: cov.nrows(),
            })
        } else if cov_rows != cov.ncols() {
            Err(MvGaussianError::CovNotSquare {
                nrows: cov_rows,
                ncols: cov.ncols(),
            })
        } else {
            let cache = MvgCache::from_cov(&cov)?;
            self.cov = cov;
            self.cache = OnceCell::new();
            self.cache.set(cache).unwrap();
            Ok(())
        }
    }

    /// Set the covariance matrix without input validation
    #[inline]
    pub fn set_cov_unchecked(&mut self, cov: DMatrix<f64>) {
        let cache = MvgCache::from_cov(&cov).unwrap();
        self.cov = cov;
        self.cache = OnceCell::from(cache);
    }

    #[inline]
    fn cache(&self) -> &MvgCache {
        self.cache
            .get_or_try_init(|| MvgCache::from_cov(&self.cov))
            .unwrap()
    }
}

impl From<&MvGaussian> for String {
    fn from(mvg: &MvGaussian) -> String {
        format!("Nₖ({})\n  μ: {}\n  σ: {})", mvg.ndims(), mvg.mu, mvg.cov)
    }
}

impl_display!(MvGaussian);

impl Rv<DVector<f64>> for MvGaussian {
    fn ln_f(&self, x: &DVector<f64>) -> f64 {
        let diff = x - &self.mu;
        let det_sqrt: f64 = self
            .cache()
            .cov_chol
            .l_dirty()
            .diagonal()
            .row_iter()
            .fold(1.0, |acc, y| acc * y[0]);

        let det = det_sqrt * det_sqrt;
        let inv = &(self.cache().cov_inv);
        let term: f64 = (diff.transpose() * inv * &diff)[0];
        -0.5 * (det.ln() + (diff.nrows() as f64).mul_add(LN_2PI, term))
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> DVector<f64> {
        let dims = self.mu.len();
        let norm = rand_distr::StandardNormal;
        let vals: Vec<f64> = (0..dims).map(|_| rng.sample(norm)).collect();

        let a = self.cache().cov_chol.l_dirty();
        let z: DVector<f64> = DVector::from_column_slice(&vals);

        DVector::from_fn(dims, |i, _| {
            let mut out: f64 = self.mu[i];
            for j in 0..=i {
                out += a[(i, j)] * z[j];
            }
            out
        })
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
        let det_sqrt: f64 = self
            .cache()
            .cov_chol
            .l_dirty()
            .diagonal()
            .row_iter()
            .fold(1.0, |acc, x| acc * x[0]);
        let det = det_sqrt * det_sqrt;
        det.ln()
            .mul_add(0.5, HALF_LN_2PI_E * (self.cov.nrows() as f64))
    }
}
impl HasSuffStat<DVector<f64>> for MvGaussian {
    type Stat = MvGaussianSuffStat;
    fn empty_suffstat(&self) -> Self::Stat {
        MvGaussianSuffStat::new(self.mu.len())
    }
}

impl std::error::Error for MvGaussianError {}

impl fmt::Display for MvGaussianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "requested dimension is too low"),
            Self::CovNotPositiveSemiDefinite => {
                write!(f, "covariance is not positive semi-definite")
            }
            Self::MuCovDimensionMismatch { n_mu, n_cov } => write!(
                f,
                "mean vector and covariance matrix do not align. mu is {} \
                    dimensions but cov is {} dimensions",
                n_mu, n_cov
            ),
            Self::CovNotSquare { nrows, ncols } => write!(
                f,
                "covariance matrix is not square ({} x {})",
                nrows, ncols
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Gaussian;
    use crate::misc::{ks_test, mardia};
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;
    const NTRIES: usize = 5;
    const KS_PVAL: f64 = 0.2;
    const MARDIA_PVAL: f64 = 0.2;

    test_basic_impls!(MvGaussian::standard(3).unwrap(), DVector::zeros(3));

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

        assert_eq!(
            mvg,
            Err(MvGaussianError::MuCovDimensionMismatch { n_mu: 3, n_cov: 4 })
        )
    }

    #[test]
    fn new_should_reject_cov_too_small() {
        let mu = DVector::zeros(3);
        let cov = DMatrix::identity(2, 2);
        let mvg = MvGaussian::new(mu, cov);
        assert_eq!(
            mvg,
            Err(MvGaussianError::MuCovDimensionMismatch { n_mu: 3, n_cov: 2 })
        )
    }

    #[test]
    fn new_should_reject_cov_not_square() {
        let mu = DVector::zeros(3);
        let cov = DMatrix::identity(3, 2);
        let mvg = MvGaussian::new(mu, cov);
        assert_eq!(
            mvg,
            Err(MvGaussianError::CovNotSquare { nrows: 3, ncols: 2 })
        );
    }

    #[test]
    fn ln_f_standard_x_zeros() {
        let mvg = MvGaussian::standard(3).unwrap();
        let x = DVector::<f64>::zeros(3);
        assert::close(mvg.ln_f(&x), -2.756_815_599_614_018, TOL);
    }

    #[test]
    fn ln_f_standard_x_nonzeros() {
        let mvg = MvGaussian::standard(3).unwrap();
        let x = DVector::<f64>::from_column_slice(&[0.5, 3.1, -6.2]);
        assert::close(mvg.ln_f(&x), -26.906_815_599_614_02, TOL);
    }

    #[test]
    fn ln_f_nonstandard_zeros() {
        let cov_vals = vec![
            1.017_427_88,
            0.365_866_52,
            -0.656_204_86,
            0.365_866_52,
            1.005_645_53,
            -0.425_972_61,
            -0.656_204_86,
            -0.425_972_61,
            1.272_479_72,
        ];
        let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
        let mu = DVector::<f64>::from_column_slice(&[0.5, 3.1, -6.2]);
        let mvg = MvGaussian::new(mu, cov).unwrap();
        let x = DVector::<f64>::zeros(3);
        assert::close(mvg.ln_f(&x), -24.602_370_253_215_66, TOL);
    }

    #[test]
    fn ln_f_nonstandard_nonzeros() {
        let cov_vals = vec![
            1.017_427_88,
            0.365_866_52,
            -0.656_204_86,
            0.365_866_52,
            1.005_645_53,
            -0.425_972_61,
            -0.656_204_86,
            -0.425_972_61,
            1.272_479_72,
        ];
        let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
        let mu = DVector::<f64>::from_column_slice(&[0.5, 3.1, -6.2]);
        let mvg = MvGaussian::new(mu, cov).unwrap();
        let x = DVector::<f64>::from_column_slice(&[0.5, 3.1, -6.2]);
        assert::close(mvg.ln_f(&x), -2.591_535_053_811_229_6, TOL);
    }

    #[test]
    fn sample_returns_proper_number_of_draws() {
        let cov_vals = vec![
            1.017_427_88,
            0.365_866_52,
            -0.656_204_86,
            0.365_866_52,
            1.005_645_53,
            -0.425_972_61,
            -0.656_204_86,
            -0.425_972_61,
            1.272_479_72,
        ];
        let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
        let mu = DVector::<f64>::from_column_slice(&[0.5, 3.1, -6.2]);
        let mvg = MvGaussian::new(mu, cov).unwrap();

        let mut rng = rand::thread_rng();

        let xs: Vec<DVector<f64>> = mvg.sample(103, &mut rng);

        assert_eq!(xs.len(), 103);
    }

    #[test]
    fn standard_entropy() {
        let mvg = MvGaussian::standard(3).unwrap();
        assert::close(mvg.entropy(), 4.256_815_599_614_018_5, TOL);
    }

    #[test]
    fn nonstandard_entropy() {
        let cov_vals = vec![
            1.017_427_88,
            0.365_866_52,
            -0.656_204_86,
            0.365_866_52,
            1.005_645_53,
            -0.425_972_61,
            -0.656_204_86,
            -0.425_972_61,
            1.272_479_72,
        ];
        let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
        let mu = DVector::<f64>::from_column_slice(&[0.5, 3.1, -6.2]);
        let mvg = MvGaussian::new(mu, cov).unwrap();
        assert::close(mvg.entropy(), 4.091_535_053_811_230_5, TOL);
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
                let xs: Vec<f64> =
                    xys.iter().map(|xy: &DVector<f64>| xy[0]).collect();
                let ys: Vec<f64> =
                    xys.iter().map(|xy: &DVector<f64>| xy[1]).collect();

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
            1.017_427_88,
            0.365_866_52,
            -0.656_204_86,
            0.365_866_52,
            1.005_645_53,
            -0.425_972_61,
            -0.656_204_86,
            -0.425_972_61,
            1.272_479_72,
        ];
        let cov: DMatrix<f64> = DMatrix::from_row_slice(3, 3, &cov_vals);
        let mu = DVector::<f64>::from_column_slice(&[0.5, 3.1, -6.2]);
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
