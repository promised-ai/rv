//! Gaussian Processes

use argmin::solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS};
use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, Dyn};
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
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
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
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct GaussianProcess<K>
where
    K: Kernel,
{
    /// Cholesky Decomposition of K
    k_chol: Cholesky<f64, Dyn>,
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
    pub fn k_chol(&self) -> &Cholesky<f64, Dyn> {
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
    type Index = DVector<f64>;
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
        n.mul_add(
            -HALF_LN_2PI,
            (-0.5_f64).mul_add(self.y_train.dot(&alpha), -dlog_sum),
        )
    }

    fn ln_m_with_params(
        &self,
        parameter: &DVector<f64>,
    ) -> Result<(f64, DVector<f64>), GaussianProcessError> {
        let kernel = self
            .kernel
            .reparameterize(&parameter.iter().copied().collect::<Vec<f64>>())
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

        let ln_m = n.mul_add(
            -HALF_LN_2PI,
            (-0.5_f64).mul_add(self.y_train.dot(&alpha), -dlog_sum),
        );

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
        let grad_ln_m = DVector::from(grad_ln_m);

        Ok((ln_m, grad_ln_m))
    }

    fn parameters(&self) -> DVector<f64> {
        let kernel = self.kernel();
        kernel.parameters()
    }

    fn set_parameters(
        self,
        parameters: &DVector<f64>,
    ) -> Result<Self, GaussianProcessError> {
        let (kernel, leftovers) = self
            .kernel
            .consume_parameters(parameters.iter().copied())
            .map_err(GaussianProcessError::KernelError)?;
        let leftovers: Vec<f64> = leftovers.collect();
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
    type Solver = LBFGS<
        MoreThuenteLineSearch<DVector<f64>, DVector<f64>, f64>,
        DVector<f64>,
        DVector<f64>,
        f64,
    >;

    fn generate_solver() -> Self::Solver {
        let linesearch = MoreThuenteLineSearch::new();
        LBFGS::new(linesearch, 10)
    }

    fn random_params<R: Rng>(&self, rng: &mut R) -> DVector<f64> {
        let n = self.parameters().len();
        DVector::from_iterator(n, (0..n).map(|_| rng.gen_range(-5.0..5.0)))
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
    use nalgebra::dvector;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    fn arange(start: f64, stop: f64, step_size: f64) -> Vec<f64> {
        let size = ((stop - start) / step_size).floor() as usize;
        (0..size)
            .map(|i| (i as f64).mul_add(step_size, start))
            .collect()
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

        let xs: Vec<DVector<f64>> = arange(-5.0, 5.0, 1.0)
            .into_iter()
            .map(|x| dvector![x])
            .collect();
        let pred = gp.sample_function(xs.as_slice());

        let expected_mean: DMatrix<f64> = DMatrix::from_column_slice(
            10,
            1,
            &[
                0.614_097_52,
                0.756_802_5,
                -0.141_120_01,
                -0.909_297_43,
                -0.841_470_98,
                0.085_333_65,
                0.841_470_98,
                0.563_985_6,
                0.127_422_02,
                0.010_476_83,
            ],
        );

        let mean = pred.mean().expect("Should be able to compute the mean");
        assert!(mean.relative_eq(&expected_mean, 1E-8, 1E-8));

        let expected_cov = DMatrix::from_row_slice(
            10,
            10,
            &[
                5.096_256_32e-01,
                0.000_000_00e+00,
                5.551_115_12e-17,
                6.765_421_56e-17,
                3.165_870_34e-17,
                3.449_672_76e-02,
                3.520_513_77e-19,
                -7.750_552_24e-03,
                -2.002_925_07e-03,
                -1.676_185_74e-04,
                -1.110_223_02e-16,
                9.999_999_72e-09,
                1.110_223_02e-16,
                1.387_778_78e-16,
                6.938_893_90e-17,
                1.707_618_42e-17,
                -6.920_259_18e-19,
                -2.072_911_31e-18,
                -5.059_828_46e-19,
                -4.199_226_50e-20,
                -1.110_223_02e-16,
                -1.110_223_02e-16,
                9.999_999_94e-09,
                0.000_000_00e+00,
                -5.551_115_12e-17,
                -5.030_698_08e-17,
                -1.057_097_12e-17,
                7.377_656_97e-19,
                3.917_957_51e-19,
                3.472_752_04e-20,
                -6.765_421_56e-17,
                -2.775_557_56e-17,
                3.330_669_07e-16,
                1.000_000_04e-08,
                1.110_223_02e-16,
                0.000_000_00e+00,
                1.561_251_13e-17,
                1.713_039_43e-17,
                4.150_037_93e-18,
                3.445_372_69e-19,
                -1.317_305_64e-17,
                1.040_834_09e-17,
                0.000_000_00e+00,
                -1.110_223_02e-16,
                9.999_999_94e-09,
                0.000_000_00e+00,
                -2.775_557_56e-17,
                -2.081_668_17e-17,
                -4.987_329_99e-18,
                -4.154_696_61e-19,
                3.449_672_76e-02,
                7.676_151_38e-17,
                7.806_255_64e-17,
                0.000_000_00e+00,
                0.000_000_00e+00,
                2.663_127_02e-01,
                0.000_000_00e+00,
                -1.775_970_42e-01,
                -5.699_341_56e-02,
                -5.235_330_37e-03,
                -2.629_524_45e-18,
                -1.969_351_60e-18,
                -3.415_236_84e-18,
                -3.469_446_95e-18,
                0.000_000_00e+00,
                0.000_000_00e+00,
                9.999_999_94e-09,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                -7.750_552_24e-03,
                -9.403_293_25e-18,
                -3.769_720_13e-18,
                9.378_348_79e-18,
                1.387_778_78e-17,
                -1.775_970_42e-01,
                1.110_223_02e-16,
                6.235_919_81e-01,
                5.222_724_53e-01,
                1.284_158_94e-01,
                -2.002_925_07e-03,
                -2.538_164_32e-18,
                -2.373_717_19e-18,
                4.438_452_64e-18,
                -2.439_454_89e-18,
                -5.699_341_56e-02,
                0.000_000_00e+00,
                5.222_724_53e-01,
                9.811_305_76e-01,
                6.049_809_83e-01,
                -1.676_185_74e-04,
                -3.012_464_45e-19,
                -2.266_319_91e-19,
                -6.048_013_77e-20,
                -2.020_173_58e-19,
                -5.235_330_37e-03,
                0.000_000_00e+00,
                1.284_158_94e-01,
                6.049_809_83e-01,
                9.998_727_40e-01,
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
        assert!(&parameters.relative_eq(&dvector![0.0, 0.0], 1E-9, 1E-9));

        let expected_ln_m = -5.029_140_040_847_684;
        let expected_grad = dvector![2.068_285_41, -1.191_110_32];

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
        let (ln_m, grad_ln_m) = gp.ln_m_with_params(&parameters).unwrap();
        assert::close(ln_m, expected_ln_m, 1E-7);
        assert!(grad_ln_m.relative_eq(&expected_grad, 1E-7, 1E-7));
    }

    #[test]
    fn log_marginal_b() -> Result<(), KernelError> {
        let x_train: DMatrix<f64> =
            DMatrix::from_column_slice(5, 1, &[-4.0, -3.0, -2.0, -1.0, 1.0]);
        let y_train: DVector<f64> = x_train.map(|x| x.sin()).column(0).into();

        let kernel = RBFKernel::new(1.994_891_474_270_000_8)?
            * ConstantKernel::new(1.221_163_421_070_665)?;
        let parameters = kernel.parameters();
        assert!(relative_eq(
            &parameters,
            &dvector![0.690_589_65, 0.199_804_03],
            1E-7,
            1E-7
        ));

        let expected_ln_m = -3.414_870_095_916_796;
        let expected_grad = dvector![0.0, 0.0];

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
        let (ln_m, grad_ln_m) = gp.ln_m_with_params(&parameters).unwrap();
        assert::close(ln_m, expected_ln_m, 1E-7);
        assert!(grad_ln_m.relative_eq(&expected_grad, 1E-6, 1E-6));
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

        assert!(opt_params.relative_eq(&dvector![0.657_854_21], 1E-5, 1E-5));
        assert::close(gp.ln_m(), -3.444_937_833_462_115, 1E-7);
        assert::close(
            gp.ln_m(),
            gp.ln_m_with_params(&gp.kernel().parameters()).unwrap().0,
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

        assert!(opt_params.relative_eq(
            &dvector![0.199_804_03, 0.690_589_65],
            1E-5,
            1E-5
        ));

        assert::close(gp.ln_m(), -3.414_870_095_916_796, 1E-7);
        assert::close(
            gp.ln_m(),
            gp.ln_m_with_params(&gp.kernel().parameters()).unwrap().0,
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
            .reparameterize(&[3.099_752_67, 0.516_338_23])?;
        let gp =
            GaussianProcess::train(kernel, xs, ys, NoiseModel::Uniform(0.0))
                .expect("Should produce GP");
        let expected_k_chol: DMatrix<f64> = DMatrix::from_row_slice(
            6,
            6,
            &[
                4.710_887_58e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                2.311_209_28e+00,
                4.104_969_36e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                2.729_284_89e-01,
                2.498_691_55e+00,
                3.984_283_17e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                5.498_016_88e-02,
                1.058_107_06e+00,
                3.994_303_01e+00,
                2.261_723_20e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                7.757_674_14e-03,
                3.088_465_97e-01,
                2.538_478_56e+00,
                3.584_280_88e+00,
                1.675_133_57e+00,
                0.000_000_00e+00,
                7.666_996_49e-04,
                6.266_390_03e-02,
                1.082_699_33e+00,
                2.872_531_28e+00,
                3.289_048_54e+00,
                1.395_356_72e+00,
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
            0.917_022,
            1.220_324_49,
            0.500_114_37,
            0.802_332_57,
            0.646_755_89,
            0.592_338_59,
        ]);

        let ys = &ys + &dy;

        let kernel: ProductKernel<ConstantKernel, RBFKernel> =
            (ConstantKernel::new_unchecked(1.0)
                * RBFKernel::new_unchecked(1.0))
            .reparameterize(&[2.886_720_93, -0.033_327_73])?;
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
                4.333_051_38e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                4.880_168_69e-01,
                4.380_118_30e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                7.999_446_59e-04,
                4.826_837_17e-01,
                4.236_925_19e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                6.516_715_70e-06,
                3.335_496_65e-02,
                2.476_599_40e+00,
                3.527_532_47e+00,
                0.000_000_00e+00,
                0.000_000_00e+00,
                1.822_923_56e-08,
                7.913_467_57e-04,
                4.989_987_09e-01,
                2.628_868_78e+00,
                3.345_556_26e+00,
                0.000_000_00e+00,
                1.750_971_16e-11,
                6.446_687_85e-06,
                3.448_226_27e-02,
                5.752_472_07e-01,
                2.684_100_80e+00,
                3.278_532_35e+00,
            ],
        );

        assert!(gp.k_chol().l().relative_eq(&expected_k_chol, 1E-7, 1E-7));
        Ok(())
    }
}
