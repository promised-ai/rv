use crate::consts::LN_2PI;
use crate::data::{DataOrSuffStat, MvGaussianSuffStat};
use crate::dist::{MvGaussian, NormalInvWishart};
use crate::misc::lnmv_gamma;
use crate::traits::{ConjugatePrior, SuffStat};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::{LN_2, PI};

type MvgData<'a> = DataOrSuffStat<'a, DVector<f64>, MvGaussian>;

macro_rules! extract_stat_then {
    ($dims: expr, $x: ident, $func: expr) => {{
        match $x {
            DataOrSuffStat::SuffStat(ref stat) => $func(&stat),
            DataOrSuffStat::Data(xs) => {
                let mut stat = MvGaussianSuffStat::new($dims);
                stat.observe_many(&xs);
                $func(&stat)
            }
            DataOrSuffStat::None => {
                let stat = MvGaussianSuffStat::new($dims);
                $func(&stat)
            }
        }
    }};
}

fn ln_z(k: f64, df: usize, scale: &DMatrix<f64>) -> f64 {
    let d = scale.nrows();
    let p = d as f64;
    let v2 = (df as f64) / 2.0;
    (v2 * p) * LN_2 + lnmv_gamma(d, v2) + (p / 2.0) * (2.0 * PI / k).ln()
        - v2 * scale.clone().determinant().ln()
}

impl ConjugatePrior<DVector<f64>, MvGaussian> for NormalInvWishart {
    type Posterior = NormalInvWishart;

    fn posterior<S>(&self, x: &S) -> NormalInvWishart
    where
        S: Into<MvGaussianSuffStat>,
    {
        let stat: MvGaussianSuffStat = (*x).into();

        if stat.n() == 0 {
            return self.clone();
        }

        let nf = stat.n() as f64;

        let xbar = stat.sum_x() / stat.n() as f64;
        let diff = &xbar - self.mu();
        let s = stat.sum_x_sq() - nf * (&xbar * &xbar.transpose());

        let kn = self.k() + stat.n() as f64;
        let vn = self.df() + stat.n();
        let mn = (self.k() * self.mu() + stat.sum_x()) / kn;
        let sn = self.scale()
            + s
            + (self.k() * stat.n() as f64) / kn * &diff * &diff.transpose();

        NormalInvWishart::new(mn, kn, vn, sn)
            .expect("Invalid posterior parameters")
    }

    fn ln_m<S>(&self, x: &S) -> f64
    where
        S: Into<MvGaussianSuffStat>,
    {
        let stat: MvGaussianSuffStat = (*x).into();
        let post = self.posterior(&stat);
        let z0 = ln_z(self.k(), self.df(), self.scale());
        let zn = ln_z(post.k(), post.df(), post.scale());

        let nd: f64 = (self.dims() as f64) * (stat.n() as f64);

        zn - z0 - nd / 2.0 * LN_2PI
    }

    fn ln_pp<S>(&self, y: &DVector<f64>, x: &S) -> f64
    where
        S: Into<MvGaussianSuffStat>,
    {
        let stat: MvGaussianSuffStat = (*x).into();
        let mut y_stat = MvGaussianSuffStat::new(self.dims());
        y_stat.observe(&y);

        let post = self.posterior(&stat);
        let pred = post.posterior(&y_stat);

        let zn = ln_z(post.k(), post.df(), post.scale());
        let zm = ln_z(pred.k(), pred.df(), pred.scale());

        let d: f64 = self.dims() as f64;

        zm - zn - d / 2.0 * LN_2PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-12;

    fn obs_fxtr() -> MvGaussianSuffStat {
        let x0v = vec![3.57839693972576, 0.725404224946106];
        let x1v = vec![2.76943702988488, -0.0630548731896562];
        let x2v = vec![-1.34988694015652, 0.714742903826096];
        let x3v = vec![3.03492346633185, -0.204966058299775];

        let x0 = DVector::<f64>::from_column_slice(2, &x0v);
        let x1 = DVector::<f64>::from_column_slice(2, &x1v);
        let x2 = DVector::<f64>::from_column_slice(2, &x2v);
        let x3 = DVector::<f64>::from_column_slice(2, &x3v);

        let mut stat = MvGaussianSuffStat::new(1);

        stat.observe(&x0);
        stat.observe(&x1);
        stat.observe(&x2);
        stat.observe(&x3);

        stat
    }

    #[test]
    fn ln_z_identity() {
        let z1 = ln_z(1.0, 2, &DMatrix::identity(2, 2));
        assert::close(z1, 4.3689013133786361, TOL);
    }

    #[test]
    fn ln_m_identity() {
        let niw = NormalInvWishart::new(
            DVector::zeros(2),
            1.0,
            2,
            DMatrix::identity(2, 2),
        )
        .unwrap();
        let obs = obs_fxtr();
        let data: MvgData = DataOrSuffStat::SuffStat(&obs);

        let pp = niw.ln_m(&data);

        assert::close(pp, -16.3923777220275, TOL);
    }
}
