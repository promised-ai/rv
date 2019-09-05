//! Gaussian Processes

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, Dynamic};
use rand::Rng;

use log::{debug, info};

use once_cell::sync::OnceCell;

use crate::consts::HALF_LN_2PI;
use crate::dist::Gamma;
use crate::dist::MvGaussian;
use crate::result;
use crate::traits::Rv;

pub mod kernel;
use kernel::Kernel;

use optim::bfgs::{bfgs, outer_product_self, BFGSParams};
use optim::OptimizeError;

impl From<OptimizeError> for result::Error {
    fn from(optimize_error: OptimizeError) -> result::Error {
        result::Error::new(
            result::ErrorKind::InvalidParameterError,
            &format!("Optimization Failed: {:?}", optimize_error),
        )
    }
}

/// Model of noise to use in Gaussian Process
pub enum NoiseModel<'a> {
    /// The same noise is applied to all values
    Uniform(f64),
    /// Different noise values are applied to each y-value
    PerPoint(&'a DVector<f64>),
}

impl<'a> Default for NoiseModel<'a> {
    fn default() -> Self {
        NoiseModel::Uniform(1E-8)
    }
}

impl<'a> NoiseModel<'a> {
    /// Enact the given noise model onto the given covariance matrix
    pub fn add_noise_to_kernel(&self, cov: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            NoiseModel::Uniform(noise) => {
                let diag = DVector::from_element(cov.nrows(), noise.powi(2));
                cov + &DMatrix::from_diagonal(&diag)
            }
            NoiseModel::PerPoint(sigma) => {
                assert_eq!(
                    cov.nrows(),
                    sigma.nrows(),
                    "Per point noise must be the same size as y_train"
                );
                let s = sigma.map(|e| e * e);
                cov + &DMatrix::from_diagonal(&s)
            }
        }
    }
}

/// Parameters for running GaussianProcess
pub struct GaussianProcessParams<'a> {
    /// Type of noise
    noise_model: NoiseModel<'a>,
    /// Optimization parameters
    bfgs_params: BFGSParams,
}

impl<'a> GaussianProcessParams<'a> {
    pub fn with_bfgs_params(self, bfgs_params: BFGSParams) -> Self {
        Self {
            bfgs_params,
            ..self
        }
    }

    pub fn with_noise_model(self, noise_model: NoiseModel<'a>) -> Self {
        Self {
            noise_model,
            ..self
        }
    }
}

impl<'a> Default for GaussianProcessParams<'a> {
    fn default() -> Self {
        Self {
            noise_model: NoiseModel::default(),
            bfgs_params: BFGSParams::default(),
        }
    }
}

pub struct GaussianProcess<'a, K>
where
    K: Kernel,
{
    /// Cholesky Decomposition of K
    k_chol: Cholesky<f64, Dynamic>,
    /// Dual coefficients of training data in kernel space.
    alpha: DVector<f64>,
    /// Covariance Kernel
    kernel: K,
    /// x values used in training
    x_train: &'a DMatrix<f64>,
    /// y values used in training
    y_train: &'a DVector<f64>,
    /// Inverse covariance matrix
    k_inv: OnceCell<DMatrix<f64>>,
    /// Given parameters
    params: GaussianProcessParams<'a>,
}

impl<'a, K> GaussianProcess<'a, K>
where
    K: Kernel,
{
    /// Train a Gaussian Process on the given data points
    ///
    /// # Arguments
    /// * `kernel` - Kernel to use to determine covariance
    /// * `x_train` - Values to use for input into `f`
    /// * `y_train` - Known values for `f(x)`
    /// * `params` - GaussianProcessParams to use. Can just use `GaussianProcessParams::default()`.
    ///
    pub fn train(
        kernel: K,
        x_train: &'a DMatrix<f64>,
        y_train: &'a DVector<f64>,
        params: GaussianProcessParams<'a>,
    ) -> result::Result<Self> {
        let k = params
            .noise_model
            .add_noise_to_kernel(&kernel.covariance_with_gradient(x_train).0);

        // Decompose K into Cholesky lower lower triangular matrix
        let k_chol = match Cholesky::new(k) {
            Some(ch) => Ok(ch),
            None => Err(
                result::Error::new(
                    result::ErrorKind::InvalidParameterError,
                    "The kernel is not returning a positive-definite matrix. Try adding a increasing the impact of the noise model"
                )
            ),
        }?;

        let alpha = k_chol.solve(y_train);

        Ok(GaussianProcess {
            k_chol,
            alpha,
            kernel,
            x_train,
            y_train,
            k_inv: OnceCell::new(),
            params,
        })
    }

    /// Return the inverse of K.
    pub fn k_inv(&self) -> &DMatrix<f64> {
        self.k_inv.get_or_init(|| self.k_chol.inverse())
    }

    /// Return the Cholesky decomposition of K
    pub fn k_chol(&self) -> &Cholesky<f64, Dynamic> {
        &(self.k_chol)
    }

    /// Return the kernel being used in this GP
    pub fn kernel(&self) -> &K {
        &(self.kernel)
    }

    /// Return the log marginal likelihood
    pub fn ln_m(&self) -> f64 {
        let k_chol = self.k_chol();
        let dlog_sum = k_chol.l_dirty().diagonal().map(|x| x.ln()).sum();
        let n: f64 = self.x_train.nrows() as f64;
        let alpha = k_chol.solve(self.y_train);
        -0.5 * self.y_train.dot(&alpha) - dlog_sum - n * HALF_LN_2PI
    }

    /// Log-marginal likelihood
    pub fn ln_m_with_parameters(
        &self,
        theta: &DVector<f64>,
    ) -> Option<(f64, DVector<f64>)> {
        let kernel = K::from_parameters(theta);

        // GPML Equation 2.30
        let (k, k_grad) = kernel.covariance_with_gradient(self.x_train);
        let k = self.params.noise_model.add_noise_to_kernel(&k);

        let m = k.nrows();
        let k_chol = Cholesky::new(k)?;
        let alpha = k_chol.solve(self.y_train);
        let dlog_sum = k_chol.l_dirty().diagonal().map(|x| x.ln()).sum();
        let n: f64 = self.x_train.nrows() as f64;

        let ln_m = -0.5 * self.y_train.dot(&alpha) - dlog_sum - n * HALF_LN_2PI;

        // GPML Equation 5.9
        let aat_kinv = &outer_product_self(&alpha) - &k_chol.inverse();
        let mut grad_ln_m = DVector::zeros(theta.len());
        for i in 0..theta.len() {
            let theta_i_grad = &k_grad[i];
            let mut sum = 0.0;
            for j in 0..m {
                sum += (aat_kinv.row(j) * theta_i_grad.column(j))[0];
            }
            grad_ln_m[i] = 0.5 * sum;
        }
        Some((ln_m, grad_ln_m))
    }

    /// Optimize kernel parameters s.t. ln_m is maximized
    pub fn optimize<RNG: Rng>(
        self,
        n_tries: usize,
        rng: &mut RNG,
    ) -> result::Result<Self> {
        let objective = |th: &DVector<f64>| {
            let (ln_m, grad_ln_m) = self.ln_m_with_parameters(th).unwrap_or((
                std::f64::NEG_INFINITY,
                DVector::zeros(th.nrows()),
            ));
            let grad_ln_m = grad_ln_m.map(|x| -x);
            debug!(
                "objective: theta = {}, ln_m = {}, grad_ln_m = {}",
                th, -ln_m, grad_ln_m
            );
            (-ln_m, grad_ln_m)
        };

        // Try optimization on given parameters
        let (mut best_theta, mut best_ln_m) = match bfgs(
            self.kernel.parameters(),
            &self.params.bfgs_params,
            objective,
        ) {
            Ok(opt_theta) => {
                let opt_ln_m = self
                    .ln_m_with_parameters(&opt_theta)
                    .map(|x| x.0)
                    .unwrap_or(std::f64::NEG_INFINITY);
                (opt_theta, opt_ln_m)
            }
            Err(e) => return Err(e.into()),
        };

        // Draw random numbers from gamma(2, 1) dist as parameters for the kernel
        let dist = Gamma::new(2.0, 1.0).unwrap();
        for _n_try in 0..n_tries {
            let theta: DVector<f64> = DVector::from_column_slice(
                &dist.sample(best_theta.nrows(), rng),
            );
            let theta = theta.map(|x| x.ln());
            match bfgs(theta, &self.params.bfgs_params, objective) {
                Ok(opt_theta) => {
                    let opt_ln_m = self
                        .ln_m_with_parameters(&opt_theta)
                        .map(|x| x.0)
                        .unwrap_or(std::f64::NEG_INFINITY);
                    if opt_ln_m > best_ln_m {
                        best_ln_m = opt_ln_m;
                        best_theta = opt_theta;
                    }
                }
                Err(_) => {}
            }
        }

        let new_kernel = K::from_parameters(&best_theta);
        let new_gp = GaussianProcess::train(
            new_kernel,
            self.x_train,
            self.y_train,
            self.params,
        );
        new_gp
    }

    /// Return a `GaussianProcessPrediction` to preform prediction on.
    pub fn predict<'b>(
        &'b mut self,
        xs: &'b DMatrix<f64>,
    ) -> GaussianProcessPrediction<'b, K>
    where
        'b: 'a,
    {
        let k_trans = self.kernel.covariance(xs, self.x_train);
        let y_mean = &k_trans * &self.alpha;
        GaussianProcessPrediction {
            gp: self,
            y_mean,
            k_trans,
            xs,
            cov: OnceCell::new(),
            dist: OnceCell::new(),
        }
    }
}

/// Structure for making GP preditions
pub struct GaussianProcessPrediction<'a, K>
where
    K: Kernel,
{
    /// Parent GP
    gp: &'a mut GaussianProcess<'a, K>,
    /// Mean of y values
    y_mean: DVector<f64>,
    /// Intermediate matrix
    k_trans: DMatrix<f64>,
    /// Values to predict `f(x)` against.
    xs: &'a DMatrix<f64>,
    /// Covariance matrix
    cov: OnceCell<DMatrix<f64>>,
    /// Output Distribution
    dist: OnceCell<MvGaussian>,
}

impl<'a, K> GaussianProcessPrediction<'a, K>
where
    K: Kernel,
{
    /// Return the mean of the posterior
    pub fn mean(&self) -> &DVector<f64> {
        &(self.y_mean)
    }

    /// Return the covariance of the posterior
    pub fn cov(&self) -> &DMatrix<f64> {
        self.cov.get_or_init(|| {
            let v = self.gp.k_chol().solve(&(self.k_trans.transpose()));
            let kernel = self.gp.kernel();
            &kernel.covariance(self.xs, self.xs) - &(self.k_trans) * &v
        })
    }

    /// Return the standard deviation of posterior.
    pub fn std(&self) -> DVector<f64> {
        let kernel = self.gp.kernel();
        let k_inv = self.gp.k_inv();
        let k_ti = &(self.k_trans) * k_inv;

        let mut y_var: DVector<f64> = kernel.diag(self.xs);
        for i in 0..y_var.nrows() {
            y_var[i] -= (0..k_inv.ncols()).map(|j| k_ti[(i, j)] * self.k_trans[(i, j)]).sum::<f64>();
        }
        y_var.map(|e| e.sqrt())
    }

    /// Return the MV Gaussian distribution which shows the predicted values
    pub fn dist(&self) -> &MvGaussian {
        let mean = (self.mean()).clone();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::gaussian::kernel::*;
    use env_logger;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn logging_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn arange(start: f64, stop: f64, step_size: f64) -> DMatrix<f64> {
        let size = ((stop - start) / step_size).floor() as usize;
        let it = (0..size).map(|i| start + (i as f64) * step_size);
        DMatrix::from_iterator(size, 1, it)
    }

    #[test]
    fn simple() {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = RBFKernel::new(1.0);
        let mut gp = GaussianProcess::train(
            kernel,
            &x_train,
            &y_train,
            GaussianProcessParams::default(),
        )
        .unwrap();

        let xs: DMatrix<f64> = arange(-5.0, 5.0, 1.0);
        let mut pred = gp.predict(&xs);

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

        let mean = pred.mean();
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
    fn log_marginal() {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = RBFKernel::new(1.0);
        let parameters = kernel.parameters();

        let expected_ln_m = -5.029140040847684;
        let expected_grad = DVector::from_column_slice(&[2.06828541]);

        let mut gp = GaussianProcess::train(
            kernel,
            &x_train,
            &y_train,
            GaussianProcessParams::default(),
        )
        .unwrap();
        // Without Gradient
        assert::close(gp.ln_m(), expected_ln_m, 1E-7);

        // With Gradient
        let (ln_m, grad_ln_m) = gp
            .ln_m_with_parameters(&parameters)
            .expect("Should be Some");
        assert::close(ln_m, expected_ln_m, 1E-7);
        assert!(grad_ln_m.relative_eq(&expected_grad, 1E-7, 1E-7));
    }

    #[test]
    fn optimize_gp_1_param() {
        logging_init();

        let mut rng = StdRng::seed_from_u64(0x1234);
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = RBFKernel::new(1.0);
        let gp_params = GaussianProcessParams::default()
            .with_bfgs_params(BFGSParams::default().with_accuracy(1E-5));

        let gp = GaussianProcess::train(kernel, &x_train, &y_train, gp_params)
            .unwrap();

        let mut gp = gp.optimize(10, &mut rng).expect("Failed to optimize");
        let opt_params = gp.kernel().parameters();

        assert!(opt_params.relative_eq(
            &DVector::from_column_slice(&[0.6578541991730281]),
            1E-7,
            1E-7
        ));
        assert::close(gp.ln_m(), -3.4449378334620895, 1E-7);
        assert::close(
            gp.ln_m(),
            gp.ln_m_with_parameters(&gp.kernel().parameters())
                .unwrap()
                .0,
            1E-7,
        );
    }

    #[test]
    fn optimize_gp_2_param() {
        logging_init();

        let mut rng = StdRng::seed_from_u64(0x1234);
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel =
            ProductKernel::new(ConstantKernel::new(1.0), RBFKernel::new(1.0));
        let gp_params = GaussianProcessParams::default()
            .with_bfgs_params(BFGSParams::default().with_accuracy(1E-5));

        let gp = GaussianProcess::train(kernel, &x_train, &y_train, gp_params)
            .unwrap();

        let mut gp = gp.optimize(10, &mut rng).expect("Failed to optimize");
        let opt_params = gp.kernel().parameters();
        println!("Found Opt Params = {}", opt_params);
        println!("Found ln_m = {}", gp.ln_m());

        assert!(opt_params.relative_eq(
            &DVector::from_column_slice(&[0.19980395, 0.69058964]),
            1E-5,
            1E-5
        ));
        assert::close(gp.ln_m(), -3.414870095916784, 1E-7);
        assert::close(
            gp.ln_m(),
            gp.ln_m_with_parameters(&gp.kernel().parameters())
                .unwrap()
                .0,
            1E-7,
        );
    }
}
