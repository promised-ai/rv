//! Gaussian Processes

use argmin::solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS};
use log::warn;
use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, Dynamic};
use rand::Rng;

use once_cell::sync::OnceCell;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::consts::HALF_LN_2PI;
use crate::dist::MvGaussian;
use crate::traits::*;

pub mod kernel;
use kernel::*;

mod noise_model;
pub use self::noise_model::*;

use super::{RandomProcess, RandomProcessMle};

#[inline]
fn outer_product_self(col: &DVector<f64>) -> DMatrix<f64> {
    let row = DMatrix::from_row_slice(1, col.nrows(), col.as_slice());
    col * row
}

/// Errors from GaussianProcess
#[derive(Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum GaussianProcessError {
    /// The kernel is not returning a positive-definite matrix. Try adding a small, constant noise parameter as y_train_sigma.
    NotPositiveSemiDefinite,
    /// Optimization Error
    OptimizerError,
    /// Extranious Parameters
    ExtraniousParameters(Vec<f64>),
}

impl std::error::Error for GaussianProcessError {}
impl std::fmt::Display for GaussianProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct GaussianProcess<K>
where
    K: Kernel,
{
    /// Cholesky Decomposition of K
    k_chol: Cholesky<f64, Dynamic>,
    /// Dual coefficients of training data in kernel space.
    alpha: DVector<f64>,
    /// Covariance Kernel
    pub kernel: K,
    /// x values used in training
    x_train: DMatrix<f64>,
    /// y values used in training
    y_train: DVector<f64>,
    /// Inverse covariance matrix
    k_inv: DMatrix<f64>,
    /// Noise Model
    pub noise_model: NoiseModel,
}

impl<K> GaussianProcess<K>
where
    K: Kernel,
{
    /// Train a Gaussian Process on the given data points
    ///
    /// # Arguments
    /// * `kernel` - Kernel to use to determine covariance
    /// * `x_train` - Values to use for input into `f`
    /// * `y_train` - Known values for `f(x)`
    /// * `noise_model` - Noise model to use for fitting
    pub fn train(
        kernel: K,
        x_train: DMatrix<f64>,
        y_train: DVector<f64>,
        noise_model: NoiseModel,
    ) -> Result<Self, GaussianProcessError> {
        let k = noise_model
            .add_noise_to_kernel(&kernel.covariance(&x_train, &x_train));

        // Decompose K into Cholesky lower lower triangular matrix
        let k_chol = match Cholesky::new(k) {
            Some(ch) => Ok(ch),
            None => Err(GaussianProcessError::NotPositiveSemiDefinite),
        }?;

        let k_inv = k_chol.inverse();
        let alpha = k_chol.solve(&y_train);

        Ok(GaussianProcess {
            k_chol,
            alpha,
            kernel,
            x_train,
            y_train,
            k_inv,
            noise_model,
        })
    }

    /// Return the inverse of K.
    pub fn k_inv(&self) -> &DMatrix<f64> {
        &self.k_inv
    }

    /// Return the Cholesky decomposition of K
    pub fn k_chol(&self) -> &Cholesky<f64, Dynamic> {
        &(self.k_chol)
    }

    /// Return the kernel being used in this GP
    pub fn kernel(&self) -> &K {
        &(self.kernel)
    }
}

impl<K> RandomProcess<f64> for GaussianProcess<K>
where
    K: Kernel,
{
    type Index = Vec<f64>;
    type Parameter = Vec<f64>;
    type SampleFunction = GaussianProcessPrediction<K>;
    type ParameterError = GaussianProcessError;

    fn sample_function(
        &self,
        indicies: &[Self::Index],
    ) -> Self::SampleFunction {
        let n = indicies.len();
        let m = indicies.get(0).map(|i| i.len()).unwrap_or(0);

        let indicies: DMatrix<f64> = DMatrix::from_iterator(
            n,
            m,
            indicies.iter().map(|i| i.iter().cloned()).flatten(),
        );
        let k_trans = self.kernel.covariance(&indicies, &self.x_train);
        let y_mean = &k_trans * &self.alpha;
        GaussianProcessPrediction {
            gp: self.clone(),
            y_mean,
            k_trans,
            xs: indicies,
            cov: OnceCell::new(),
            dist: OnceCell::new(),
        }
    }

    fn ln_m(&self) -> f64 {
        let k_chol = self.k_chol();
        let dlog_sum = k_chol.l_dirty().diagonal().map(|x| x.ln()).sum();
        let n: f64 = self.x_train.nrows() as f64;
        let alpha = k_chol.solve(&self.y_train);
        -0.5 * self.y_train.dot(&alpha) - dlog_sum - n * HALF_LN_2PI
    }

    fn ln_m_with_parameters(
        &self,
        parameter: Self::Parameter,
    ) -> Option<(f64, Self::Parameter)> {
        let kernel = K::from_parameters(&parameter);

        // GPML Equation 2.30
        let (k, k_grad) = kernel.covariance_with_gradient(&self.x_train);
        let k = self.noise_model.add_noise_to_kernel(&k);

        let m = k.nrows();
        // TODO: try to symmetricize the matrix
        let maybe_k_chol = Cholesky::new(k.clone());

        if maybe_k_chol.is_none() {
            warn!(
                "failed to find chol of k = {}, with parameters = {:?}",
                k, parameter
            );
        }

        let k_chol = maybe_k_chol?;
        let alpha = k_chol.solve(&self.y_train);
        let dlog_sum = k_chol.l_dirty().diagonal().map(|x| x.ln()).sum();
        let n: f64 = self.x_train.nrows() as f64;

        let ln_m = -0.5 * self.y_train.dot(&alpha) - dlog_sum - n * HALF_LN_2PI;

        // GPML Equation 5.9
        let aat_kinv = &outer_product_self(&alpha) - &k_chol.inverse();
        let grad_ln_m: Vec<f64> = (0..parameter.len())
            .map(|i| {
                let theta_i_grad = &k_grad[i];
                let mut sum = 0.0;
                for j in 0..m {
                    sum += (aat_kinv.row(j) * theta_i_grad.column(j))[0];
                }
                0.5 * sum
            })
            .collect();
        Some((ln_m, grad_ln_m))
    }

    fn parameters(&self) -> Self::Parameter {
        self.kernel().parameters()
    }

    fn set_parameters(
        &mut self,
        parameters: Self::Parameter,
    ) -> Result<(), GaussianProcessError> {
        let (kernel, leftovers) = K::consume_parameters(&parameters);

        let k = self.noise_model.add_noise_to_kernel(
            &kernel.covariance(&self.x_train, &self.x_train),
        );

        // Decompose K into Cholesky lower lower triangular matrix
        let k_chol = match Cholesky::new(k) {
            Some(ch) => Ok(ch),
            None => Err(GaussianProcessError::NotPositiveSemiDefinite),
        }?;

        let alpha = k_chol.solve(&self.y_train);
        if !leftovers.is_empty() {
            return Err(GaussianProcessError::ExtraniousParameters(
                leftovers.to_vec(),
            ));
        }

        self.kernel = kernel;
        self.k_chol = k_chol;
        self.alpha = alpha;
        Ok(())
    }
}

impl<K> RandomProcessMle<f64> for GaussianProcess<K>
where
    K: Kernel,
{
    type Solver = LBFGS<
        MoreThuenteLineSearch<Self::Parameter, f64>,
        Self::Parameter,
        f64,
    >;

    fn generate_solver() -> Self::Solver {
        let linesearch = MoreThuenteLineSearch::new();
        LBFGS::new(linesearch, 50)
    }

    fn random_params<R: Rng>(&self, rng: &mut R) -> Self::Parameter {
        let (lower, upper) = self.kernel.parameter_bounds();
        lower
            .into_iter()
            .zip(upper.into_iter())
            .map(|(l, u)| rng.gen_range(l.ln(), u.ln()))
            .collect()
    }
}

/// Structure for making GP preditions
pub struct GaussianProcessPrediction<K>
where
    K: Kernel,
{
    /// Parent GP
    gp: GaussianProcess<K>,
    /// Mean of y values
    y_mean: DVector<f64>,
    /// Intermediate matrix
    k_trans: DMatrix<f64>,
    /// Values to predict `f(x)` against.
    xs: DMatrix<f64>,
    /// Covariance matrix
    cov: OnceCell<DMatrix<f64>>,
    /// Output Distribution
    dist: OnceCell<MvGaussian>,
}

impl<K> GaussianProcessPrediction<K>
where
    K: Kernel,
{
    /// Return the covariance of the posterior
    pub fn cov(&self) -> &DMatrix<f64> {
        self.cov.get_or_init(|| {
            let v = self.gp.k_chol().solve(&(self.k_trans.transpose()));
            let kernel = self.gp.kernel();
            &kernel.covariance(&self.xs, &self.xs) - &(self.k_trans) * &v
        })
    }

    /// Return the standard deviation of posterior.
    pub fn std(&self) -> DVector<f64> {
        let kernel = self.gp.kernel();
        let k_inv = self.gp.k_inv();
        let k_ti = &(self.k_trans) * k_inv;

        let mut y_var: DVector<f64> = kernel.diag(&self.xs);
        for i in 0..y_var.nrows() {
            y_var[i] -= (0..k_inv.ncols())
                .map(|j| k_ti[(i, j)] * self.k_trans[(i, j)])
                .sum::<f64>();
        }
        y_var.map(|e| e.sqrt())
    }

    /// Return the MV Gaussian distribution which shows the predicted values
    pub fn dist(&self) -> &MvGaussian {
        let mean = self.mean().unwrap();
        let cov = (self.cov()).clone();
        self.dist
            .get_or_init(|| MvGaussian::new(mean, cov).unwrap())
    }

    /// Draw a single value from the corresponding MV Gaussian
    pub fn draw<RNG: Rng>(&self, rng: &mut RNG) -> DVector<f64> {
        self.dist().draw(rng)
    }

    /// Return a number of samples from the MV Gaussian
    pub fn sample<RNG: Rng>(
        &self,
        size: usize,
        rng: &mut RNG,
    ) -> Vec<DVector<f64>> {
        self.dist().sample(size, rng)
    }
}

impl<K> Rv<DVector<f64>> for GaussianProcessPrediction<K>
where
    K: Kernel,
{
    fn ln_f(&self, x: &DVector<f64>) -> f64 {
        self.dist().ln_f(x)
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> DVector<f64> {
        self.dist().draw(rng)
    }
}

impl<K> Mean<DVector<f64>> for GaussianProcessPrediction<K>
where
    K: Kernel,
{
    fn mean(&self) -> Option<DVector<f64>> {
        Some(self.y_mean.clone())
    }
}

impl<K> Variance<DVector<f64>> for GaussianProcessPrediction<K>
where
    K: Kernel,
{
    fn variance(&self) -> Option<DVector<f64>> {
        let kernel = self.gp.kernel();
        let k_inv = self.gp.k_inv();
        let k_ti = &(self.k_trans) * k_inv;

        let mut y_var: DVector<f64> = kernel.diag(&self.xs);
        for i in 0..y_var.nrows() {
            y_var[i] -= (0..k_inv.ncols())
                .map(|j| k_ti[(i, j)] * self.k_trans[(i, j)])
                .sum::<f64>();
        }
        Some(y_var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::relative_eq;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn arange(start: f64, stop: f64, step_size: f64) -> Vec<f64> {
        let size = ((stop - start) / step_size).floor() as usize;
        (0..size).map(|i| start + (i as f64) * step_size).collect()
    }

    #[test]
    fn simple() {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = RBFKernel::new(1.0);
        let gp = GaussianProcess::train(
            kernel,
            x_train,
            y_train,
            NoiseModel::default(),
        )
        .unwrap();

        let xs: Vec<Vec<f64>> = arange(-5.0, 5.0, 1.0)
            .into_iter()
            .map(|x| vec![x])
            .collect();
        let pred = gp.sample_function(&xs);

        let expected_mean: DMatrix<f64> = DMatrix::from_column_slice(
            10,
            1,
            &[
                0.61409752,
                0.7568025,
                -0.14112001,
                -0.90929743,
                -0.84147098,
                0.08533365,
                0.84147098,
                0.5639856,
                0.12742202,
                0.01047683,
            ],
        );

        let mean = pred.mean().expect("Should be able to compute the mean");
        assert!(mean.relative_eq(&expected_mean, 1E-8, 1E-8));

        let expected_cov = DMatrix::from_row_slice(
            10,
            10,
            &[
                5.09625632e-01,
                0.00000000e+00,
                5.55111512e-17,
                6.76542156e-17,
                3.16587034e-17,
                3.44967276e-02,
                3.52051377e-19,
                -7.75055224e-03,
                -2.00292507e-03,
                -1.67618574e-04,
                -1.11022302e-16,
                9.99999972e-09,
                1.11022302e-16,
                1.38777878e-16,
                6.93889390e-17,
                1.70761842e-17,
                -6.92025918e-19,
                -2.07291131e-18,
                -5.05982846e-19,
                -4.19922650e-20,
                -1.11022302e-16,
                -1.11022302e-16,
                9.99999994e-09,
                0.00000000e+00,
                -5.55111512e-17,
                -5.03069808e-17,
                -1.05709712e-17,
                7.37765697e-19,
                3.91795751e-19,
                3.47275204e-20,
                -6.76542156e-17,
                -2.77555756e-17,
                3.33066907e-16,
                1.00000004e-08,
                1.11022302e-16,
                0.00000000e+00,
                1.56125113e-17,
                1.71303943e-17,
                4.15003793e-18,
                3.44537269e-19,
                -1.31730564e-17,
                1.04083409e-17,
                0.00000000e+00,
                -1.11022302e-16,
                9.99999994e-09,
                0.00000000e+00,
                -2.77555756e-17,
                -2.08166817e-17,
                -4.98732999e-18,
                -4.15469661e-19,
                3.44967276e-02,
                7.67615138e-17,
                7.80625564e-17,
                0.00000000e+00,
                0.00000000e+00,
                2.66312702e-01,
                0.00000000e+00,
                -1.77597042e-01,
                -5.69934156e-02,
                -5.23533037e-03,
                -2.62952445e-18,
                -1.96935160e-18,
                -3.41523684e-18,
                -3.46944695e-18,
                0.00000000e+00,
                0.00000000e+00,
                9.99999994e-09,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                -7.75055224e-03,
                -9.40329325e-18,
                -3.76972013e-18,
                9.37834879e-18,
                1.38777878e-17,
                -1.77597042e-01,
                1.11022302e-16,
                6.23591981e-01,
                5.22272453e-01,
                1.28415894e-01,
                -2.00292507e-03,
                -2.53816432e-18,
                -2.37371719e-18,
                4.43845264e-18,
                -2.43945489e-18,
                -5.69934156e-02,
                0.00000000e+00,
                5.22272453e-01,
                9.81130576e-01,
                6.04980983e-01,
                -1.67618574e-04,
                -3.01246445e-19,
                -2.26631991e-19,
                -6.04801377e-20,
                -2.02017358e-19,
                -5.23533037e-03,
                0.00000000e+00,
                1.28415894e-01,
                6.04980983e-01,
                9.99872740e-01,
            ],
        );

        let cov = pred.cov();
        assert!(cov.relative_eq(&expected_cov, 1E-7, 1E-7))
    }

    #[test]
    fn log_marginal_a() {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = RBFKernel::new(1.0) * ConstantKernel::new(1.0);
        let parameters = kernel.parameters();
        assert!(relative_eq(&parameters, &vec![0.0, 0.0], 1E-9, 1E-9));

        let expected_ln_m = -5.029140040847684;
        let expected_grad = vec![2.06828541, -1.19111032];

        let gp = GaussianProcess::train(
            kernel,
            x_train,
            y_train,
            NoiseModel::default(),
        )
        .unwrap();
        // Without Gradient
        assert::close(gp.ln_m(), expected_ln_m, 1E-7);

        // With Gradient
        let (ln_m, grad_ln_m) = gp.ln_m_with_parameters(parameters).unwrap();
        assert::close(ln_m, expected_ln_m, 1E-7);
        assert!(relative_eq(grad_ln_m, expected_grad, 1E-7, 1E-7));
    }

    #[test]
    fn log_marginal_b() {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = RBFKernel::new(1.9948914742700008)
            * ConstantKernel::new(1.221163421070665);
        let parameters = kernel.parameters();
        assert!(relative_eq(
            &parameters,
            &vec![0.69058965, 0.19980403],
            1E-7,
            1E-7
        ));

        let expected_ln_m = -3.414870095916796;
        let expected_grad = vec![0.0, 0.0];

        let gp = GaussianProcess::train(
            kernel,
            x_train,
            y_train,
            NoiseModel::default(),
        )
        .unwrap();
        // Without Gradient
        let ln_m = gp.ln_m();
        assert::close(ln_m, expected_ln_m, 1E-7);

        // With Gradient
        let (ln_m, grad_ln_m) = gp.ln_m_with_parameters(parameters).unwrap();
        assert::close(ln_m, expected_ln_m, 1E-7);
        assert!(relative_eq(grad_ln_m, expected_grad, 1E-6, 1E-6));
    }

    #[test]
    fn optimize_gp_1_param() {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = RBFKernel::new(1.0);
        let noise_model = NoiseModel::default();

        let gp = GaussianProcess::train(kernel, x_train, y_train, noise_model)
            .unwrap();

        let mut rng = SmallRng::seed_from_u64(0xABCD);
        let gp = gp.optimize(100, 10, &mut rng).expect("Failed to optimize");
        let opt_params = gp.kernel().parameters();

        assert!(relative_eq(opt_params, vec![0.65785421], 1E-5, 1E-5));
        assert::close(gp.ln_m(), -3.444937833462115, 1E-7);
        assert::close(
            gp.ln_m(),
            gp.ln_m_with_parameters(gp.kernel().parameters()).unwrap().0,
            1E-7,
        );
    }

    #[test]
    fn optimize_gp_2_param() {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = ConstantKernel::new(1.0) * RBFKernel::new(1.0);
        let noise_model = NoiseModel::default();

        let gp = GaussianProcess::train(kernel, x_train, y_train, noise_model)
            .unwrap();

        let mut rng = SmallRng::seed_from_u64(0xABCD);
        let gp = gp.optimize(100, 10, &mut rng).expect("Failed to optimize");
        let opt_params = gp.kernel().parameters();

        assert!(relative_eq(
            opt_params,
            vec![0.19980403, 0.69058965],
            1E-5,
            1E-5
        ));

        assert::close(gp.ln_m(), -3.414870095916796, 1E-7);
        assert::close(
            gp.ln_m(),
            gp.ln_m_with_parameters(gp.kernel().parameters()).unwrap().0,
            1E-7,
        );
    }
}
