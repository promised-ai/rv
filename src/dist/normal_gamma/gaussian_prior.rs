use std::collections::BTreeMap;
use std::f64::consts::LN_2;

use crate::consts::*;
use crate::data::{
    extract_stat, extract_stat_then, DataOrSuffStat, GaussianSuffStat,
};
use crate::dist::{Gaussian, NormalGamma};
use crate::gaussian_prior_geweke_testable;
use crate::misc::ln_gammafn;
use crate::suffstat_traits::*;
use crate::test::GewekeTestable;
use crate::traits::*;

#[inline]
fn ln_z(r: f64, s: f64, v: f64) -> f64 {
    // This is what is should be in clearer, normal, operations
    // (v + 1.0) / 2.0 * LN_2 + HALF_LN_PI - 0.5 * r.ln() - (v / 2.0) * s.ln()
    //     + ln_gammafn(v / 2.0).0
    // ... and here is what is is when we use mul_add to reduce rounding errors
    let half_v = 0.5 * v;
    (half_v + 0.5).mul_add(LN_2, HALF_LN_PI)
        - 0.5_f64.mul_add(r.ln(), half_v.mul_add(s.ln(), -ln_gammafn(half_v)))
}

fn posterior_from_stat(
    ng: &NormalGamma,
    stat: &GaussianSuffStat,
) -> NormalGamma {
    let nf = stat.n() as f64;
    let r = ng.r() + nf;
    let v = ng.v() + nf;
    let m = ng.m().mul_add(ng.r(), stat.sum_x()) / r;
    let s =
        ng.s() + stat.sum_x_sq() + ng.r().mul_add(ng.m() * ng.m(), -r * m * m);
    NormalGamma::new(m, r, s, v).expect("Invalid posterior params.")
}

impl ConjugatePrior<f64, Gaussian> for NormalGamma {
    type Posterior = Self;
    type LnMCache = f64;
    type LnPpCache = (GaussianSuffStat, f64);

    fn posterior(&self, x: &DataOrSuffStat<f64, Gaussian>) -> Self {
        extract_stat_then(x, GaussianSuffStat::new, |stat: GaussianSuffStat| {
            posterior_from_stat(self, &stat)
        })
    }

    #[inline]
    fn ln_m_cache(&self) -> Self::LnMCache {
        ln_z(self.r(), self.s, self.v)
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::LnMCache,
        x: &DataOrSuffStat<f64, Gaussian>,
    ) -> f64 {
        extract_stat_then(x, GaussianSuffStat::new, |stat: GaussianSuffStat| {
            let post = posterior_from_stat(self, &stat);
            let lnz_n = ln_z(post.r, post.s, post.v);
            (-(stat.n() as f64)).mul_add(HALF_LN_2PI, lnz_n) - cache
        })
    }

    #[inline]
    fn ln_pp_cache(
        &self,
        x: &DataOrSuffStat<f64, Gaussian>,
    ) -> Self::LnPpCache {
        let stat = extract_stat(x, GaussianSuffStat::new);
        let post_n = posterior_from_stat(self, &stat);
        let lnz_n = ln_z(post_n.r, post_n.s, post_n.v);
        (stat, lnz_n)
    }

    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &f64) -> f64 {
        let mut stat = cache.0.clone();
        let lnz_n = cache.1;

        stat.observe(y);
        let post_m = posterior_from_stat(self, &stat);

        let lnz_m = ln_z(post_m.r(), post_m.s(), post_m.v());

        -HALF_LN_2PI + lnz_m - lnz_n
    }
}

gaussian_prior_geweke_testable!(NormalGamma, Gaussian);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GaussianData;

    const TOL: f64 = 1E-12;

    #[test]
    fn geweke() {
        use crate::test::GewekeTester;

        let mut rng = rand::thread_rng();
        let pr = NormalGamma::new(0.1, 1.2, 0.5, 1.8).unwrap();
        let n_passes = (0..5)
            .map(|_| {
                let mut tester = GewekeTester::new(pr.clone(), 20);
                tester.run_chains(5_000, 20, &mut rng);
                u8::from(tester.eval(0.025).is_ok())
            })
            .sum::<u8>();
        assert!(n_passes > 1);
    }

    #[test]
    fn ln_z_all_ones() {
        let z = ln_z(1.0, 1.0, 1.0);
        assert::close(z, 1.837_877_066_409_35, TOL);
    }

    #[test]
    fn ln_z_not_all_ones() {
        let z = ln_z(1.2, 0.4, 5.2);
        assert::close(z, 5.369_728_190_685_34, TOL);
    }

    #[test]
    fn ln_marginal_likelihood_vec_data() {
        let ng = NormalGamma::new(2.1, 1.2, 1.3, 1.4).unwrap();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x = GaussianData::<f64>::Data(&data);
        let m = ng.ln_m(&x);
        assert::close(m, -7.697_070_183_440_38, TOL);
    }

    #[test]
    fn ln_marginal_likelihood_suffstat() {
        let ng = NormalGamma::new(2.1, 1.2, 1.3, 1.4).unwrap();
        let mut stat = GaussianSuffStat::new();
        stat.observe(&1.0);
        stat.observe(&2.0);
        stat.observe(&3.0);
        stat.observe(&4.0);
        let x = GaussianData::<f64>::SuffStat(&stat);
        let m = ng.ln_m(&x);
        assert::close(m, -7.697_070_183_440_38, TOL);
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
        let x = GaussianData::<f64>::SuffStat(&stat);
        let m = ng.ln_m(&x);
        assert::close(m, -7.697_070_183_440_38, TOL);
    }

    #[test]
    fn posterior_predictive_positive_value() {
        let ng = NormalGamma::new(2.1, 1.2, 1.3, 1.4).unwrap();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x = GaussianData::<f64>::Data(&data);
        let pp = ng.ln_pp(&3.0, &x);
        assert::close(pp, -1.284_386_384_996_11, TOL);
    }

    #[test]
    fn posterior_predictive_negative_value() {
        let ng = NormalGamma::new(2.1, 1.2, 1.3, 1.4).unwrap();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x = GaussianData::<f64>::Data(&data);
        let pp = ng.ln_pp(&-3.0, &x);
        assert::close(pp, -6.163_769_886_218_6, TOL);
    }

    #[test]
    fn ln_m_vs_monte_carlo() {
        use crate::misc::logsumexp;

        let n_samples = 8_000_000;
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (m, r, s, v) = (0.0, 1.2, 2.3, 3.4);
        let ng = NormalGamma::new(m, r, s, v).unwrap();
        let ln_m = ng.ln_m(&DataOrSuffStat::<f64, Gaussian>::from(&xs));

        let mc_est = {
            let ln_fs: Vec<f64> = ng
                .sample_stream(&mut rand::thread_rng())
                .take(n_samples)
                .map(|gauss: Gaussian| {
                    xs.iter().map(|x| gauss.ln_f(x)).sum::<f64>()
                })
                .collect();
            logsumexp(&ln_fs) - (n_samples as f64).ln()
        };
        // high error tolerance. MC estimation is not the most accurate...
        assert::close(ln_m, mc_est, 1e-2);
    }

    #[test]
    fn ln_m_vs_importance() {
        use crate::misc::logsumexp;

        let n_samples = 2_000_000;
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (m, r, s, v) = (1.0, 2.2, 3.3, 4.4);
        let ng = NormalGamma::new(m, r, s, v).unwrap();
        let ln_m = ng.ln_m(&DataOrSuffStat::<f64, Gaussian>::from(&xs));
        let post = ng.posterior(&DataOrSuffStat::<f64, Gaussian>::from(&xs));

        let mc_est = {
            let mut rng = rand::thread_rng();
            // let pr_p = Gamma::new(1.6, 2.2).unwrap();
            // let pr_m = Gaussian::new(1.0, 2.0).unwrap();
            let ln_fs: Vec<f64> = (0..n_samples)
                .map(|_| {
                    // let mu: f64 = pr_m.draw(&mut rng);
                    // let prec: f64 = pr_p.draw(&mut rng);
                    // let gauss = Gaussian::new(mu, prec.sqrt().recip()).unwrap();
                    let gauss: Gaussian = post.draw(&mut rng);
                    let ln_f = xs.iter().map(|x| gauss.ln_f(x)).sum::<f64>();

                    // ln_f + ng.ln_f(&gauss) - pr_m.ln_f(&mu) - pr_p.ln_f(&prec)
                    ln_f + ng.ln_f(&gauss) - post.ln_f(&gauss)
                })
                .collect();
            logsumexp(&ln_fs) - (n_samples as f64).ln()
        };
        // high error tolerance. MC estimation is not the most accurate...
        assert::close(ln_m, mc_est, 1e-2);
    }
}
