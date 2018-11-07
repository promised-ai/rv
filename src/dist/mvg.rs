extern crate nalgebra;
extern crate rand;

use std::f64::consts::{E, PI};

use self::nalgebra::{DMatrix, DVector};
use self::rand::Rng;

use data::MvGaussianSuffStat;
use result;
use traits::*;

/// [Multivariate Gaussian/Normal Distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution),
/// 𝒩(μ, Σ).
///
/// # Example
///
/// Generate a Wishart random 3x3 matrix **Σ** ~ W<sub>ν</sub>(S)
///
/// ```
/// # extern crate rv;
/// extern crate rand;
/// extern crate nalgebra;
///
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
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct MvGaussian {
    // Mean vector
    pub mu: DVector<f64>,
    // Covariance matrix
    pub cov: DMatrix<f64>,
}

impl MvGaussian {
    pub fn new(mu: DVector<f64>, cov: DMatrix<f64>) -> result::Result<Self> {
        let cov_square = cov.nrows() == cov.ncols();
        let dims_match = mu.len() == cov.nrows();

        if !dims_match {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "Number of dimensions in μ and Σ must match";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        } else if !cov_square {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let err = result::Error::new(err_kind, "Σ must be square");
            Err(err)
        } else {
            Ok(MvGaussian { mu, cov })
        }
    }

    pub fn standard(dims: usize) -> result::Result<Self> {
        if dims < 1 {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let err = result::Error::new(err_kind, "ndims must be >= 1");
            Err(err)
        } else {
            let mu = DVector::zeros(dims);
            let cov = DMatrix::identity(dims, dims);
            MvGaussian::new(mu, cov)
        }
    }
}

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
        let norm = rand::distributions::Normal::new(0.0, 1.0);
        let vals: Vec<f64> = (0..dims).map(|_| rng.sample(norm)).collect();

        let a: DMatrix<f64> = self
            .cov
            .clone()
            .cholesky()
            .expect("Cholesky decomp failed")
            .unpack();
        let z: DVector<f64> = DVector::from_column_slice(dims, &vals);

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
    extern crate assert;
    use super::*;
    use dist::Gaussian;
    use misc::{ks_test, mardia};

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

        match mvg {
            Err(err) => {
                let msg = err.description();
                assert!(msg.contains("dimensions"));
            }
            Ok(..) => panic!("Should've failed"),
        }
    }

    #[test]
    fn new_should_reject_cov_too_small() {
        let mu = DVector::zeros(3);
        let cov = DMatrix::identity(2, 2);
        let mvg = MvGaussian::new(mu, cov);

        match mvg {
            Err(err) => {
                let msg = err.description();
                assert!(msg.contains("dimensions"));
            }
            Ok(..) => panic!("Should've failed"),
        }
    }

    #[test]
    fn new_should_reject_cov_not_square() {
        let mu = DVector::zeros(3);
        let cov = DMatrix::identity(3, 2);
        let mvg = MvGaussian::new(mu, cov);

        match mvg {
            Err(err) => {
                let msg = err.description();
                assert!(msg.contains("square"));
            }
            Ok(..) => panic!("Should've failed"),
        }
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
        let x = DVector::<f64>::from_column_slice(3, &vec![0.5, 3.1, -6.2]);
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
        let mu = DVector::<f64>::from_column_slice(3, &vec![0.5, 3.1, -6.2]);
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
        let mu = DVector::<f64>::from_column_slice(3, &vec![0.5, 3.1, -6.2]);
        let mvg = MvGaussian::new(mu, cov).unwrap();
        let x = DVector::<f64>::from_column_slice(3, &vec![0.5, 3.1, -6.2]);
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
        let mu = DVector::<f64>::from_column_slice(3, &vec![0.5, 3.1, -6.2]);
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
        let mu = DVector::<f64>::from_column_slice(3, &vec![0.5, 3.1, -6.2]);
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
        let mu = DVector::<f64>::from_column_slice(3, &vec![0.5, 3.1, -6.2]);
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
