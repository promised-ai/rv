extern crate nalgebra;

use self::nalgebra::{DMatrix, DVector};
use consts::LN_2PI;
use data::{DataOrSuffStat, MvGaussianSuffStat};
use dist::{GaussianInvWishart, MvGaussian};
use misc::lnmv_gamma;
use std::f64::consts::{LN_2, PI};
use traits::{ConjugatePrior, SuffStat};

type MvgData<'a> = DataOrSuffStat<'a, DVector<f64>, MvGaussian>;

fn extract_stat(xs: &MvgData) -> MvGaussianSuffStat {
    match xs {
        DataOrSuffStat::Data(data) => {
            let dims = data[0].len();
            let mut stat = MvGaussianSuffStat::new(dims);
            stat.observe_many(&data);
            stat
        }
        DataOrSuffStat::SuffStat(ref stat) => (*stat).clone(),
    }
}

fn ln_z(k: f64, df: usize, scale: &DMatrix<f64>) -> f64 {
    let d = scale.nrows();
    let p = d as f64;
    let v2 = (df as f64) / 2.0;
    (v2 * p) * LN_2 + lnmv_gamma(d, v2) + (p / 2.0) * (2.0 * PI / k).ln()
        - v2 * scale.clone().determinant().ln()
}

impl ConjugatePrior<DVector<f64>, MvGaussian> for GaussianInvWishart {
    type Posterior = GaussianInvWishart;

    fn posterior(&self, x: &MvgData) -> GaussianInvWishart {
        let stat = extract_stat(&x);

        let xbar = &stat.sum_x / stat.n as f64;
        let diff = xbar - &self.mu;
        let s = &stat.sum_x_sq; // FIXME: wrong

        let kn = self.k + stat.n as f64;
        let vn = self.df + stat.n;
        let mn = (self.k * &self.mu + &stat.sum_x) / kn;
        let sn = &self.scale
            + s
            + (self.k * stat.n as f64) / kn * &diff * &diff.transpose();

        GaussianInvWishart::new(mn, kn, vn, sn).unwrap()
    }

    fn ln_m(&self, x: &MvgData) -> f64 {
        let post = self.posterior(&x);
        let z0 = ln_z(self.k, self.df, &self.scale);
        let zn = ln_z(post.k, post.df, &post.scale);

        let nd: f64 = (self.mu.len() as f64) * (x.n() as f64);

        zn - z0 - nd / 2.0 * LN_2PI
    }

    fn ln_pp(&self, y: &DVector<f64>, x: &MvgData) -> f64 {
        let dims = self.mu.len();

        let mut y_stat = MvGaussianSuffStat::new(dims);
        y_stat.observe(&y);
        let y_packed = DataOrSuffStat::SuffStat(&y_stat);

        let post = self.posterior(&x);
        let pred = post.posterior(&y_packed);

        let zn = ln_z(post.k, post.df, &post.scale);
        let zm = ln_z(pred.k, pred.df, &pred.scale);

        let d: f64 = self.mu.len() as f64;

        zm - zn - d / 2.0 * LN_2PI
    }
}
