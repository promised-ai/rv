//! Broyden–Fletcher–Goldfarb–Shanno algorithm
//! Optimizes a twice differentiable function

use nalgebra;
use nalgebra::base::EuclideanNorm;
use nalgebra::base::Norm;
use nalgebra::{DMatrix, DVector};
use crate::line_search::{WolfeParams, wolfe_search};
use crate::OptimizeError;
use log::debug;

#[inline]
pub fn outer_product_self(col: &DVector<f64>) -> DMatrix<f64> {
    let row = DMatrix::from_row_slice(1, col.nrows(), col.as_slice());
    col * row
}

/// Parameters for the BFGS Optimizer
pub struct BFGSParams {
    /// Maximum number of iterations to run
    pub max_iter: usize,
    /// Parameters given to the Wolfe line search algorithm.
    pub wolfe_params: WolfeParams,
    /// Exit accuracy
    pub accuracy: f64,
}

impl Default for BFGSParams {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            wolfe_params: WolfeParams::default(),
            accuracy: 1E-7,
        }
    }
}

impl BFGSParams {
    pub fn with_accuracy(self, accuracy: f64) -> Self {
        Self {
            accuracy,
            ..self
        }
    }

    pub fn with_max_iter(self, max_iter: usize) -> Self {
        Self {
            max_iter,
            ..self
        }
    }

    pub fn with_wofle_params(self, wolfe_params: WolfeParams) -> Self {
        Self {
            wolfe_params,
            ..self
        }
    }
}

pub fn bfgs<F>(
    x0: DVector<f64>,
    params: &BFGSParams,
    f: F,
) -> Result<DVector<f64>, OptimizeError>
where
    F: Fn(&DVector<f64>) -> (f64, DVector<f64>),
{
    let mut b_inv = DMatrix::identity(x0.nrows(), x0.nrows());

    let mut x = x0;
    let mut g_x = f(&x).1;
    let metric = EuclideanNorm {};
    for i in 0..params.max_iter {
        let search_dir = -1.0 * &b_inv * &g_x;
        debug!("bfgs: i = {}, x = {}, search_dir = {}, b_inv = {}, g_x = {}", i, x, search_dir, b_inv, g_x);

        let epsilon = wolfe_search(&params.wolfe_params, |e| {
            let offset = &search_dir * e;
            let (f_e, g_e) = f(&(&x + &offset));
            let d = g_e.dot(&search_dir);
            (f_e, d)
        })?;

        x += epsilon * &search_dir;

        debug!("Wolfe Update: x = {}", x);

        let g_x_last = g_x.clone();
        g_x = f(&x).1;

        // Check stopping condition
        if metric.norm(&g_x) < params.accuracy {
            return Ok(x);
        }

        let y: DVector<f64> = &g_x - &g_x_last;
        let s: DVector<f64> = epsilon * &search_dir;
        let sst: DMatrix<f64> = outer_product_self(&s);
        let sty: f64 = s.dot(&y);
        let yt_bi_y: f64 = y.dot(&(&b_inv * &y));

        let add = ((sty + yt_bi_y) * &sst) / (sty * sty);
        let sub =
            (&b_inv * &y * &s.transpose() + &s * &y.transpose() * &b_inv) / sty;

        b_inv += &add - &sub;
    };
    Err(OptimizeError::MaxIterationReached)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfgs_x_cubed() {
        let res = bfgs(DVector::zeros(1), &BFGSParams::default(), |v| {
            let x = v[0];
            let y = -(x-1.0).powi(3)- (x-1.0).powi(2);
            let dy_dx = -3.0 * x.powi(2) + 4.0 * x - 1.0;
            let g = DVector::from_column_slice(&[dy_dx]);
            (y, g)
        });
        
        assert!(res.is_ok());
        assert::close(res.unwrap()[0], 1.0 / 3.0, 1E-5);
    }

    #[test]
    fn bfgs_rosenbrock() {
        let x0: DVector<f64> = DVector::zeros(2);
        let f = |x: &DVector<f64>| {
            let y = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
            let gx = -400.0 * (x[1] - x[0].powi(2)) * x[0] - 2.0 * (1.0 - x[0]);
            let gy =  200.0 * (x[1] - x[0].powi(2));
            (y, DVector::from_column_slice(&[gx, gy]))
        };

        let xmin = bfgs(x0, &BFGSParams::default(), f);
        assert!(xmin.is_ok());
        let expected = DVector::from_column_slice(&[1.0, 1.0]);
        assert!(xmin.unwrap().relative_eq(&expected, 1E-5, 1E-5));
    }
}
