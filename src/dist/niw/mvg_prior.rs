use crate::consts::LN_2PI;
use crate::data::{DataOrSuffStat, MvGaussianSuffStat};
use crate::dist::{MvGaussian, NormalInvWishart};
use crate::misc::lnmv_gamma;
use crate::traits::{ConjugatePrior, SuffStat};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::{LN_2, PI};

type MvgData<'a> = DataOrSuffStat<'a, DVector<f64>, MvGaussian>;

macro_rules! extract_stat_then {
    ($ndims: expr, $x: ident, $func: expr) => {{
        match $x {
            DataOrSuffStat::SuffStat(ref stat) => $func(&stat),
            DataOrSuffStat::Data(xs) => {
                let mut stat = MvGaussianSuffStat::new($ndims);
                stat.observe_many(&xs);
                $func(&stat)
            }
            DataOrSuffStat::None => {
                let stat = MvGaussianSuffStat::new($ndims);
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
    type Posterior = Self;
    type LnMCache = f64;
    type LnPpCache = (Self, f64);

    fn posterior(&self, x: &MvgData) -> NormalInvWishart {
        if x.n() == 0 {
            return self.clone();
        }

        let nf = x.n() as f64;
        extract_stat_then!(self.ndims(), x, |stat: &MvGaussianSuffStat| {
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
        })
    }

    #[inline]
    fn ln_m_cache(&self) -> f64 {
        ln_z(self.k(), self.df(), self.scale())
    }

    fn ln_m_with_cache(&self, cache: &Self::LnMCache, x: &MvgData) -> f64 {
        let z0 = cache;
        let post = self.posterior(&x);
        let zn = ln_z(post.k(), post.df(), post.scale());
        let nd: f64 = (self.ndims() as f64) * (x.n() as f64);

        zn - z0 - nd / 2.0 * LN_2PI
    }

    #[inline]
    fn ln_pp_cache(&self, x: &MvgData) -> Self::LnPpCache {
        let post = self.posterior(&x);
        let zn = ln_z(post.k(), post.df(), post.scale());
        (post, zn)
    }

    fn ln_pp_with_cache(
        &self,
        cache: &Self::LnPpCache,
        y: &DVector<f64>,
    ) -> f64 {
        let post = &cache.0;
        let zn = cache.1;

        let mut y_stat = MvGaussianSuffStat::new(self.ndims());
        y_stat.observe(&y);
        let y_packed = DataOrSuffStat::SuffStat(&y_stat);

        let pred = post.posterior(&y_packed);

        let zm = ln_z(pred.k(), pred.df(), pred.scale());

        let d: f64 = self.ndims() as f64;

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

        let x0 = DVector::<f64>::from_column_slice(&x0v);
        let x1 = DVector::<f64>::from_column_slice(&x1v);
        let x2 = DVector::<f64>::from_column_slice(&x2v);
        let x3 = DVector::<f64>::from_column_slice(&x3v);

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
