//! Gaussian Processes

use argmin::solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS};
use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, Dynamic};
use once_cell::sync::OnceCell;
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::dist::MvGaussian;
use crate::{consts::HALF_LN_2PI, traits::Mean, traits::Rv, traits::Variance};

pub mod kernel;
use kernel::{Kernel, KernelError};

mod noise_model;
pub use self::noise_model::NoiseModel;

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
    /// Error from the kernel function
    KernelError(KernelError),
    /// The given noise model does not match the training data
    MisshapenNoiseModel(String),
}

impl std::error::Error for GaussianProcessError {}
impl std::fmt::Display for GaussianProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPositiveSemiDefinite => {
                writeln!(f, "Covariance matrix is not semi-positive definite")
            }
            Self::MisshapenNoiseModel(msg) => {
                writeln!(f, "Noise model error: {}", msg)
            }
            Self::KernelError(e) => writeln!(f, "Error from kernel: {}", e),
        }
    }
}

impl From<KernelError> for GaussianProcessError {
    fn from(e: KernelError) -> Self {
        Self::KernelError(e)
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
            .add_noise_to_kernel(&kernel.covariance(&x_train, &x_train))
            .map_err(GaussianProcessError::MisshapenNoiseModel)?;

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
    type Param = Vec<f64>;
    type SampleFunction = GaussianProcessPrediction<K>;
    type Error = GaussianProcessError;

    fn sample_function(
        &self,
        indicies: &[Self::Index],
    ) -> Self::SampleFunction {
        let n = indicies.len();
        let m = indicies.get(0).map(|i| i.len()).unwrap_or(0);

        let indicies: DMatrix<f64> = DMatrix::from_iterator(
            n,
            m,
            indicies.iter().flat_map(|i| i.iter().cloned()),
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

    fn ln_m_with_params(
        &self,
        parameter: Self::Param,
    ) -> Result<(f64, Self::Param), GaussianProcessError> {
        let kernel = self
            .kernel
            .reparameterize(&parameter)
            .map_err(GaussianProcessError::KernelError)?;

        // GPML Equation 2.30
        let (k, k_grad) = kernel
            .covariance_with_gradient(&self.x_train)
            .map_err(|e| GaussianProcessError::KernelError(e.into()))?;
        let k = self.noise_model.add_noise_to_kernel(&k).unwrap(); // if we got here, the noise model will be okay

        let m = k.nrows();
        // TODO: try to symmetricize the matrix
        let maybe_k_chol = Cholesky::new(k.clone());

        if maybe_k_chol.is_none() {
            eprintln!(
                "failed to find chol of k = {}, with parameters = {:?}",
                k, parameter
            );
        }

        let k_chol = maybe_k_chol
            .ok_or(GaussianProcessError::NotPositiveSemiDefinite)?;
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
        Ok((ln_m, grad_ln_m))
    }

    fn parameters(&self) -> Self::Param {
        self.kernel().parameters()
    }

    fn set_parameters(
        self,
        parameters: Self::Param,
    ) -> Result<Self, GaussianProcessError> {
        let (kernel, leftovers) = self
            .kernel
            .consume_parameters(&parameters)
            .map_err(GaussianProcessError::KernelError)?;
        if !leftovers.is_empty() {
            return Err(GaussianProcessError::KernelError(
                KernelError::ExtraniousParameters(leftovers.len()),
            ));
        }

        Self::train(kernel, self.x_train, self.y_train, self.noise_model)
    }
}

impl<K> RandomProcessMle<f64> for GaussianProcess<K>
where
    K: Kernel,
{
    type Solver =
        LBFGS<MoreThuenteLineSearch<Self::Param, f64>, Self::Param, f64>;

    fn generate_solver() -> Self::Solver {
        let linesearch = MoreThuenteLineSearch::new();
        LBFGS::new(linesearch, 10)
    }

    fn random_params<R: Rng>(&self, rng: &mut R) -> Self::Param {
        let n = self.parameters().len();
        (0..n).map(|_| rng.gen_range(-5.0..5.0)).collect()
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
        let mean = self.y_mean.clone();
        let cov = (self.cov()).clone();
        self.dist
            .get_or_init(|| MvGaussian::new_unchecked(mean, cov))
    }

    /// Draw a single value from the corresponding MV Gaussian
    pub fn draw<RNG: Rng>(&self, rng: &mut RNG) -> DVector<f64> {
        self.dist().draw(rng)
    }

    /// Return a number of samples from the MV Gaussian
    pub fn sample<R: Rng>(
        &self,
        size: usize,
        rng: &mut R,
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
    use self::kernel::{ConstantKernel, ProductKernel, RBFKernel};
    use super::*;
    use crate::test::relative_eq;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    fn arange(start: f64, stop: f64, step_size: f64) -> Vec<f64> {
        let size = ((stop - start) / step_size).floor() as usize;
        (0..size).map(|i| start + (i as f64) * step_size).collect()
    }

    #[test]
    fn simple() {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = RBFKernel::default();
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

        let kernel = RBFKernel::default() * ConstantKernel::default();
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
        let (ln_m, grad_ln_m) = gp.ln_m_with_params(parameters).unwrap();
        assert::close(ln_m, expected_ln_m, 1E-7);
        assert!(relative_eq(grad_ln_m, expected_grad, 1E-7, 1E-7));
    }

    #[test]
    fn log_marginal_b() -> Result<(), KernelError> {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = RBFKernel::new(1.9948914742700008)?
            * ConstantKernel::new(1.221163421070665)?;
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
        let (ln_m, grad_ln_m) = gp.ln_m_with_params(parameters).unwrap();
        assert::close(ln_m, expected_ln_m, 1E-7);
        assert!(relative_eq(grad_ln_m, expected_grad, 1E-6, 1E-6));
        Ok(())
    }

    #[test]
    fn optimize_gp_1_param() {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = RBFKernel::default();
        let noise_model = NoiseModel::default();

        let gp = GaussianProcess::train(kernel, x_train, y_train, noise_model)
            .unwrap();

        let mut rng = Xoshiro256Plus::seed_from_u64(0xABCD);
        let gp = gp.optimize(100, 10, &mut rng).expect("Failed to optimize");
        let opt_params = gp.kernel().parameters();

        assert!(relative_eq(opt_params, vec![0.65785421], 1E-5, 1E-5));
        assert::close(gp.ln_m(), -3.444937833462115, 1E-7);
        assert::close(
            gp.ln_m(),
            gp.ln_m_with_params(gp.kernel().parameters()).unwrap().0,
            1E-7,
        );
    }

    #[test]
    fn optimize_gp_2_param() {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = ConstantKernel::default() * RBFKernel::default();
        let noise_model = NoiseModel::default();

        let gp = GaussianProcess::train(kernel, x_train, y_train, noise_model)
            .unwrap();

        let mut rng = Xoshiro256Plus::seed_from_u64(0xABCD);
        let gp = gp.optimize(200, 30, &mut rng).expect("Failed to optimize");
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
            gp.ln_m_with_params(gp.kernel().parameters()).unwrap().0,
            1E-7,
        );
    }

    #[test]
    fn no_noise_k_chol() -> Result<(), KernelError> {
        let xs: DMatrix<f64> =
            DMatrix::from_column_slice(6, 1, &[1., 3., 5., 6., 7., 8.]);
        let ys: DVector<f64> = xs.map(|x| x * x.sin()).column(0).into();

        let kernel: ProductKernel<ConstantKernel, RBFKernel> =
            (ConstantKernel::new_unchecked(1.0)
                * RBFKernel::new_unchecked(1.0))
            .reparameterize(&[3.09975267, 0.51633823])?;
        let gp =
            GaussianProcess::train(kernel, xs, ys, NoiseModel::Uniform(0.0))
                .expect("Should produce GP");
        let expected_k_chol: DMatrix<f64> = DMatrix::from_row_slice(
            6,
            6,
            &[
                4.71088758e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                2.31120928e+00,
                4.10496936e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                2.72928489e-01,
                2.49869155e+00,
                3.98428317e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                5.49801688e-02,
                1.05810706e+00,
                3.99430301e+00,
                2.26172320e+00,
                0.00000000e+00,
                0.00000000e+00,
                7.75767414e-03,
                3.08846597e-01,
                2.53847856e+00,
                3.58428088e+00,
                1.67513357e+00,
                0.00000000e+00,
                7.66699649e-04,
                6.26639003e-02,
                1.08269933e+00,
                2.87253128e+00,
                3.28904854e+00,
                1.39535672e+00,
            ],
        );

        assert!(gp.k_chol().l().relative_eq(&expected_k_chol, 1E-8, 1E-8));
        Ok(())
    }

    #[test]
    fn noisy_k_chol() -> Result<(), KernelError> {
        let xs: DMatrix<f64> =
            DMatrix::from_column_slice(6, 1, &[1., 3., 5., 6., 7., 8.]);
        let ys: DVector<f64> = xs.map(|x| x * x.sin()).column(0).into();
        let dy = DVector::from_row_slice(&[
            0.917022, 1.22032449, 0.50011437, 0.80233257, 0.64675589,
            0.59233859,
        ]);

        let ys = &ys + &dy;

        let kernel: ProductKernel<ConstantKernel, RBFKernel> =
            (ConstantKernel::new_unchecked(1.0)
                * RBFKernel::new_unchecked(1.0))
            .reparameterize(&[2.88672093, -0.03332773])?;
        let gp = GaussianProcess::train(
            kernel,
            xs,
            ys,
            NoiseModel::PerPoint(dy.map(|x| x * x)),
        )
        .expect("Should produce GP");
        let expected_k_chol: DMatrix<f64> = DMatrix::from_row_slice(
            6,
            6,
            &[
                4.33305138e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                4.88016869e-01,
                4.38011830e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                7.99944659e-04,
                4.82683717e-01,
                4.23692519e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                6.51671570e-06,
                3.33549665e-02,
                2.47659940e+00,
                3.52753247e+00,
                0.00000000e+00,
                0.00000000e+00,
                1.82292356e-08,
                7.91346757e-04,
                4.98998709e-01,
                2.62886878e+00,
                3.34555626e+00,
                0.00000000e+00,
                1.75097116e-11,
                6.44668785e-06,
                3.44822627e-02,
                5.75247207e-01,
                2.68410080e+00,
                3.27853235e+00,
            ],
        );

        assert!(gp.k_chol().l().relative_eq(&expected_k_chol, 1E-7, 1E-7));
        Ok(())
    }
}
