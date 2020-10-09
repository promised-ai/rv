use std::f64::consts::LN_2;

use special::Gamma as SGamma;

use crate::consts::*;
use crate::data::{DataOrSuffStat, GaussianSuffStat};
use crate::dist::{Gaussian, NormalGamma};
use crate::traits::*;

macro_rules! extract_stat_then {
    ($x: ident, $func: expr) => {{
        match $x {
            DataOrSuffStat::SuffStat(ref stat) => $func(&stat),
            DataOrSuffStat::Data(xs) => {
                let mut stat = GaussianSuffStat::new();
                stat.observe_many(&xs);
                $func(&stat)
            }
            DataOrSuffStat::None => {
                let stat = GaussianSuffStat::new();
                $func(&stat)
            }
        }
    }};
}

fn ln_z(r: f64, s: f64, v: f64) -> f64 {
    // This is what is should be in clearer, normal, operations
    // (v + 1.0) / 2.0 * LN_2 + HALF_LN_PI - 0.5 * r.ln() - (v / 2.0) * s.ln()
    //     + (v / 2.0).ln_gamma().0
    // ... and here is what is is when we use mul_add to reduce rounding errors
    let half_v = 0.5 * v;
    (half_v + 0.5).mul_add(LN_2, HALF_LN_PI)
        - 0.5f64.mul_add(r.ln(), half_v.mul_add(s.ln(), -half_v.ln_gamma().0))
}

fn posterior_from_stat(
    ng: &NormalGamma,
    stat: &GaussianSuffStat,
) -> NormalGamma {
    let nf = stat.n() as f64;
    let r = ng.r() + nf;
    let v = ng.v() + nf;
    let m = ng.m().mul_add(ng.r(), stat.sum_x()) / r;
    let s = ng.s()
        + stat.sum_x_sq()
        + ng.r().mul_add(ng.m().powi(2), -r * m.powi(2));
    NormalGamma::new(m, r, s, v).expect("Invalid posterior params.")
}

impl ConjugatePrior<f64, Gaussian> for NormalGamma {
    type Posterior = Self;
    fn posterior(&self, x: &DataOrSuffStat<f64, Gaussian>) -> Self {
        extract_stat_then!(x, |stat: &GaussianSuffStat| {
            posterior_from_stat(&self, &stat)
        })
    }

    fn ln_m(&self, x: &DataOrSuffStat<f64, Gaussian>) -> f64 {
        extract_stat_then!(x, |stat: &GaussianSuffStat| {
            let post = posterior_from_stat(&self, &stat);
            let lnz_0 = ln_z(self.r(), self.s(), self.v());
            let lnz_n = ln_z(post.r(), post.s(), post.v());
            (-(stat.n() as f64)).mul_add(HALF_LN_2PI, lnz_n) - lnz_0
        })
    }

    fn ln_pp(&self, y: &f64, x: &DataOrSuffStat<f64, Gaussian>) -> f64 {
        let mut stat = extract_stat(&x);
        let post_n = posterior_from_stat(&self, &stat);
        stat.observe(y);
        let post_m = posterior_from_stat(&self, &stat);

        let lnz_n = ln_z(post_n.r(), post_n.s(), post_n.v());
        let lnz_m = ln_z(post_m.r(), post_m.s(), post_m.v());

        -HALF_LN_2PI + lnz_m - lnz_n
    }
}

fn extract_stat(x: &DataOrSuffStat<f64, Gaussian>) -> GaussianSuffStat {
    match x {
        DataOrSuffStat::SuffStat(ref s) => (*s).clone(),
        DataOrSuffStat::Data(xs) => {
            let mut stat = GaussianSuffStat::new();
            xs.iter().for_each(|y| stat.observe(y));
            stat
        }
        DataOrSuffStat::None => GaussianSuffStat::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn ln_z_all_ones() {
        let z = ln_z(1.0, 1.0, 1.0);
        assert::close(z, 1.83787706640935, TOL);
    }

    #[test]
    fn ln_z_not_all_ones() {
        let z = ln_z(1.2, 0.4, 5.2);
        assert::close(z, 5.36972819068534, TOL);
    }

    #[test]
    fn ln_marginal_likelihood_vec_data() {
        let ng = NormalGamma::new(2.1, 1.2, 1.3, 1.4).unwrap();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x = DataOrSuffStat::Data(&data);
        let m = ng.ln_m(&x);
        assert::close(m, -7.69707018344038, TOL);
    }

    #[test]
    fn ln_marginal_likelihood_suffstat() {
        let ng = NormalGamma::new(2.1, 1.2, 1.3, 1.4).unwrap();
        let mut stat = GaussianSuffStat::new();
        stat.observe(&1.0);
        stat.observe(&2.0);
        stat.observe(&3.0);
        stat.observe(&4.0);
        let x = DataOrSuffStat::SuffStat(&stat);
        let m = ng.ln_m(&x);
        assert::close(m, -7.69707018344038, TOL);
    }

    #[test]
    fn ln_marginal_likelihood_suffstat_forgotten() {
        let ng = NormalGamma::new(2.1, 1.2, 1.3, 1.4).unwrap();
        let mut stat = GaussianSuffStat::new();
        stat.observe(&1.0);
        stat.observe(&2.0);
        stat.observe(&3.0);
        stat.observe(&4.0);
        stat.observe(&5.0);
        stat.forget(&5.0);
        let x = DataOrSuffStat::SuffStat(&stat);
        let m = ng.ln_m(&x);
        assert::close(m, -7.69707018344038, TOL);
    }

    #[test]
    fn posterior_predictive_positive_value() {
        let ng = NormalGamma::new(2.1, 1.2, 1.3, 1.4).unwrap();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x = DataOrSuffStat::Data(&data);
        let pp = ng.ln_pp(&3.0, &x);
        assert::close(pp, -1.28438638499611, TOL);
    }

    #[test]
    fn posterior_predictive_negative_value() {
        let ng = NormalGamma::new(2.1, 1.2, 1.3, 1.4).unwrap();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x = DataOrSuffStat::Data(&data);
        let pp = ng.ln_pp(&-3.0, &x);
        assert::close(pp, -6.1637698862186, TOL);
    }
}
