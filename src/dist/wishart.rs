extern crate nalgebra;
extern crate rand;

use std::f64::consts::LN_2;

use self::nalgebra::{DMatrix, DVector};
use self::rand::Rng;

use dist::MvGaussian;
use misc::lnmv_gamma;
use result;
use traits::*;

/// [Inverse Wishart distribution](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution),
/// W<sup>-1</sup>(**Ψ**,ν) over positive definite matrices.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct InvWishart {
    /// p-dimensional inverse scale matrix, **Ψ**
    pub inv_scale: DMatrix<f64>,
    // Degrees of freedom, ν > p - 1
    pub df: usize,
}

impl InvWishart {
    /// Create an Inverse Wishart distribution, W<sup>-1</sup>(**Ψ**,ν) with
    /// p-by-p inverse scale matrix, **Ψ**, and degrees of freedom, ν > p - 1.
    pub fn new(inv_scale: DMatrix<f64>, df: usize) -> result::Result<Self> {
        let err: Option<&str> = if !inv_scale.is_square() {
            Some("scale matrix not square")
        } else if (df as usize) < inv_scale.nrows() {
            Some("df too low, must be >= ndims")
        } else if inv_scale.clone().cholesky().is_none() {
            // TODO: no clone
            Some("scale matrix not positive definite")
        } else {
            None
        };

        match err {
            Some(msg) => Err(result::Error::new(
                result::ErrorKind::InvalidParameter,
                msg,
            )),
            None => Ok(InvWishart { inv_scale, df }),
        }
    }

    /// Create an Inverse Wishart distribution, W<sup>-1</sup>(**I**<sup>p</sup>,
    /// p)
    pub fn identity(dims: usize) -> Self {
        InvWishart {
            inv_scale: DMatrix::identity(dims, dims),
            df: dims,
        }
    }
}

impl Rv<DMatrix<f64>> for InvWishart {
    fn ln_f(&self, x: &DMatrix<f64>) -> f64 {
        let p = self.inv_scale.nrows();
        let pf = p as f64;
        let v = self.df as f64;

        let det_s: f64 = v * 0.5 * self.inv_scale.determinant().ln();
        let det_x: f64 = -(v + pf + 1.0) * 0.5 * x.determinant().ln();

        let denom: f64 = v * pf * 0.5 * LN_2 + lnmv_gamma(p, 0.5 * v);
        let numer: f64 =
            -0.5 * (&self.inv_scale * x.clone().try_inverse().unwrap()).trace();

        det_s - denom + det_x + numer
    }

    // XXX: The complexity of this is O(df * dims^2). There is a O(dims^2)
    // algorithm, but it's more complicated to implement, so standby.
    // See https://www.math.wustl.edu/~sawyer/hmhandouts/Wishart.pdf  for more
    fn draw<R: Rng>(&self, mut rng: &mut R) -> DMatrix<f64> {
        let p = self.inv_scale.nrows();
        let scale = self.inv_scale.clone().try_inverse().unwrap();
        let mvg = MvGaussian::new(DVector::zeros(p), scale).unwrap();
        let xs = mvg.sample(self.df, &mut rng);
        let y = xs.iter().fold(DMatrix::<f64>::zeros(p, p), |acc, x| {
            acc + x * x.transpose()
        });
        y.try_inverse().unwrap()
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<DMatrix<f64>> {
        let p = self.inv_scale.nrows();
        let scale = self.inv_scale.clone().try_inverse().unwrap();
        let mvg = MvGaussian::new(DVector::zeros(p), scale).unwrap();
        (0..n)
            .map(|_| {
                let xs = mvg.sample(self.df, &mut rng);
                let y =
                    xs.iter().fold(DMatrix::<f64>::zeros(p, p), |acc, x| {
                        acc + x * x.transpose()
                    });
                y.try_inverse().unwrap()
            })
            .collect()
    }
}

impl Support<DMatrix<f64>> for InvWishart {
    fn supports(&self, x: &DMatrix<f64>) -> bool {
        x.clone().cholesky().is_some()
    }
}

impl ContinuousDistr<DMatrix<f64>> for InvWishart {}

impl Mean<DMatrix<f64>> for InvWishart {
    fn mean(&self) -> Option<DMatrix<f64>> {
        let p = self.inv_scale.nrows();
        if self.df > p + 1 {
            Some(&self.inv_scale / (self.df - p - 1) as f64)
        } else {
            None
        }
    }
}

impl Mode<DMatrix<f64>> for InvWishart {
    fn mode(&self) -> Option<DMatrix<f64>> {
        let p = self.inv_scale.nrows();
        Some(&self.inv_scale / (self.df + p + 1) as f64)
    }
}

#[cfg(test)]
mod tests {
    extern crate assert;
    use super::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn new_should_reject_df_too_low() {
        let inv_scale = DMatrix::identity(4, 4);
        assert!(InvWishart::new(inv_scale.clone(), 4).is_ok());
        assert!(InvWishart::new(inv_scale.clone(), 5).is_ok());
        match InvWishart::new(inv_scale.clone(), 3) {
            Err(err) => {
                let msg = err.description();
                assert!(msg.contains("df too low"));
            }
            Ok(..) => panic!("Should've failed"),
        }
    }

    #[test]
    fn new_should_reject_non_square_scale() {
        let inv_scale = DMatrix::identity(4, 3);
        match InvWishart::new(inv_scale, 5) {
            Err(err) => {
                let msg = err.description();
                assert!(msg.contains("square"));
            }
            Ok(..) => panic!("Should've failed"),
        }
    }

    #[test]
    fn ln_f_standard_ident() {
        let iw = InvWishart::identity(4);
        let x = DMatrix::<f64>::identity(4, 4);
        assert::close(iw.ln_f(&x), -11.430949807317218, TOL)
    }

    #[test]
    fn ln_f_standard_mode() {
        let iw = InvWishart::identity(4);
        let x = DMatrix::<f64>::identity(4, 4) / 9.0;
        assert::close(iw.ln_f(&x), 12.11909258473473, TOL)
    }

    #[test]
    fn ln_f_nonstandard_ident() {
        let slice = vec![
            1.10576891,
            -0.20160336,
            0.09378834,
            -0.19339029,
            -0.20160336,
            0.66794786,
            -0.46020905,
            -0.62806951,
            0.09378834,
            -0.46020905,
            1.15263284,
            0.98443641,
            -0.19339029,
            -0.62806951,
            0.98443641,
            1.21050189,
        ];
        let inv_scale: DMatrix<f64> = DMatrix::from_row_slice(4, 4, &slice);
        let iw = InvWishart::new(inv_scale, 5).unwrap();
        let x = DMatrix::<f64>::identity(4, 4);
        assert::close(iw.ln_f(&x), -18.939673925150899, TOL)
    }

    #[test]
    fn draws_should_be_positive_definite() {
        let mut rng = rand::thread_rng();
        let slice = vec![
            1.10576891,
            -0.20160336,
            0.09378834,
            -0.19339029,
            -0.20160336,
            0.66794786,
            -0.46020905,
            -0.62806951,
            0.09378834,
            -0.46020905,
            1.15263284,
            0.98443641,
            -0.19339029,
            -0.62806951,
            0.98443641,
            1.21050189,
        ];
        let inv_scale: DMatrix<f64> = DMatrix::from_row_slice(4, 4, &slice);
        let iw = InvWishart::new(inv_scale, 5).unwrap();
        for x in iw.sample(100, &mut rng) {
            assert!(x.cholesky().is_some());
        }
    }

    // XXX: I've been using scipy distributionst to check my answers, but it
    // turns out that this case exposes a bug in the scipy implementation, see
    // https://github.com/scipy/scipy/issues/8844. I've tested the
    // individual terms by hand and everything looks right, so I put the
    // rust-computed value (-6.187876016819759) as the target until scipy
    // pushes the corrected code to PyPi.
    #[test]
    fn ln_f_nonstandard_mode() {
        let slice = vec![
            1.10576891,
            -0.20160336,
            0.09378834,
            -0.19339029,
            -0.20160336,
            0.66794786,
            -0.46020905,
            -0.62806951,
            0.09378834,
            -0.46020905,
            1.15263284,
            0.98443641,
            -0.19339029,
            -0.62806951,
            0.98443641,
            1.21050189,
        ];
        let inv_scale: DMatrix<f64> = DMatrix::from_row_slice(4, 4, &slice);
        let x = inv_scale.clone();
        let iw = InvWishart::new(inv_scale, 5).unwrap();
        assert::close(iw.ln_f(&x), -6.187876016819759, TOL)
    }
}
