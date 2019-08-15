//! Gaussian Processes

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, Dynamic};
use rand::Rng;

use once_cell::sync::OnceCell;

use crate::dist::MvGaussian;
use crate::result;
use crate::traits::Rv;

pub mod kernel;
use kernel::*;

use optim::bfgs;

pub struct GaussianProcess<'a, K>
where
    K: Kernel,
{
    /// Cholesky Decomposition of K
    l: Cholesky<f64, Dynamic>,
    /// Dual coefficients of training data in kernel space.
    alpha: DVector<f64>,
    kernel: &'a K,
    x_train: &'a DMatrix<f64>,
    y_train: &'a DVector<f64>,
    k_inv: OnceCell<DMatrix<f64>>,
}

impl<'a, K> GaussianProcess<'a, K>
where
    K: Kernel,
{
    pub fn train(
        kernel: &'a K,
        x_train: &'a DMatrix<f64>,
        y_train: &'a DVector<f64>,
        y_train_sigma: Option<&DVector<f64>>,
    ) -> result::Result<Self> {
        let eps: f64 = 1E-10;
        let k: DMatrix<f64> = {
            let k_no_noise = kernel.covariance(x_train, x_train);
            match y_train_sigma {
                Some(sigma) => {
                    let s = sigma.map(|e| e * e);
                    k_no_noise + &DMatrix::from_diagonal(&s)
                }
                None => {
                    k_no_noise
                        + eps
                            * &DMatrix::identity(
                                x_train.nrows(),
                                x_train.nrows(),
                            )
                }
            }
        };

        // Decompose K into Cholesky lower lower triangular matrix
        let l = match Cholesky::new(k) {
            Some(ch) => Ok(ch),
            None => Err(
                result::Error::new(
                    result::ErrorKind::InvalidParameterError,
                    "The kernel is not returning a positive-definite matrix. Try adding a small, constant noise parameter as y_train_sigma."
                )
            ),
        }?;

        let alpha = l.solve(y_train);

        Ok(GaussianProcess {
            l,
            alpha,
            kernel,
            x_train,
            y_train,
            k_inv: OnceCell::new(),
        })
    }

    /// Return the inverse of K.
    pub fn k_inv(&mut self) -> &DMatrix<f64> {
        self.k_inv.get_or_init(|| self.l.inverse())
    }

    /// Return the Cholesky decomposition of K
    pub fn l(&self) -> &Cholesky<f64, Dynamic> {
        &(self.l)
    }

    /// Return the kernel being used in this GP
    pub fn kernel(&self) -> &K {
        &(self.kernel)
    }

    /*
    /// Return the log marginal likelihood
    pub fn ln_m(&self) -> f64 {
        let k = self.kernel(self.x_train);
    }
    */

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

pub struct GaussianProcessPrediction<'a, K>
where
    K: Kernel,
{
    gp: &'a mut GaussianProcess<'a, K>,
    y_mean: DVector<f64>,
    k_trans: DMatrix<f64>,
    xs: &'a DMatrix<f64>,
    cov: OnceCell<DMatrix<f64>>,
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
    pub fn cov(&mut self) -> &DMatrix<f64> {
        self.cov.get_or_init(|| {
            let v = self.gp.l().solve(&(self.k_trans.transpose()));
            let kernel = self.gp.kernel();
            &kernel.covariance(self.xs, self.xs) - &(self.k_trans) * &v
        })
    }

    /// Return the standard deviation of posterior.
    pub fn std(&mut self) -> DVector<f64> {
        let kernel = self.gp.kernel();
        let mut y_var: DVector<f64> = kernel.diag(self.xs);
        let k_inv = self.gp.k_inv();
        let k_ti = &(self.k_trans) * k_inv;

        for i in 0..y_var.len() {
            y_var[i] -= k_ti.row(i).iter().sum::<f64>();
        }

        y_var.map(|e| e.sqrt())
    }

    pub fn dist(&mut self) -> &MvGaussian {
        let mean = (self.mean()).clone();
        let cov = (self.cov()).clone();
        self.dist
            .get_or_init(|| MvGaussian::new(mean, cov).unwrap())
    }

    pub fn draw<RNG: Rng>(&mut self, rng: &mut RNG) -> DVector<f64> {
        self.dist().draw(rng)
    }

    pub fn sample<RNG: Rng>(
        &mut self,
        size: usize,
        rng: &mut RNG,
    ) -> Vec<DVector<f64>> {
        self.dist().sample(size, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::gaussian::kernel::RBFKernel;

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

        let kernel = RBFKernel::new(1.0, 1.0);
        let mut gp =
            GaussianProcess::train(&kernel, &x_train, &y_train, None).unwrap();

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
}
