use std::collections::BTreeMap;

use special::Gamma as _;

use crate::consts::HALF_LN_2PI;
use crate::data::{
    extract_stat, extract_stat_then, DataOrSuffStat, GaussianSuffStat,
};
use crate::dist::{Gaussian, NormalInvGamma};
use crate::gaussian_prior_geweke_testable;
use crate::test::GewekeTestable;
use crate::traits::*;

#[inline]
fn ln_z(v: f64, a: f64, b: f64) -> f64 {
    // -(a * b.ln() - 0.5 * v.ln() - a.ln_gamma().0)
    let p1 = v.ln().mul_add(0.5, a.ln_gamma().0);
    -b.ln().mul_add(a, -p1)
}

// XXX: Check out section 6.3 from Kevin Murphy's paper
// https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
#[allow(clippy::many_single_char_names)]
fn posterior_from_stat(
    nig: &NormalInvGamma,
    stat: &GaussianSuffStat,
) -> NormalInvGamma {
    let n = stat.n() as f64;

    let (m, v, a, b) = nig.params();

    let v_inv = v.recip();

    let vn_inv = v_inv + n;
    let vn = vn_inv.recip();
    // let mn = (v_inv * m + stat.sum_x()) * vn;
    let mn = v_inv.mul_add(m, stat.sum_x()) / vn_inv;
    // let an = a + 0.5 * n;
    let an = n.mul_add(0.5, a);
    // let bn = b + 0.5 * (m * m * v_inv + stat.sum_x_sq() - mn * mn * vn_inv);
    let p1 = (m * m).mul_add(v_inv, stat.sum_x_sq());
    let bn = (-mn * mn).mul_add(vn_inv, p1).mul_add(0.5, b);

    NormalInvGamma::new(mn, vn, an, bn).expect("Invalid posterior params.")
}

impl ConjugatePrior<f64, Gaussian> for NormalInvGamma {
    type Posterior = Self;
    type LnMCache = f64;
    type LnPpCache = (GaussianSuffStat, f64);
    // type LnPpCache = NormalInvGamma;

    fn posterior(&self, x: &DataOrSuffStat<f64, Gaussian>) -> Self {
        extract_stat_then(x, GaussianSuffStat::new, |stat: GaussianSuffStat| {
            posterior_from_stat(self, &stat)
        })
    }

    #[inline]
    fn ln_m_cache(&self) -> Self::LnMCache {
        ln_z(self.v, self.a, self.b)
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::LnMCache,
        x: &DataOrSuffStat<f64, Gaussian>,
    ) -> f64 {
        extract_stat_then(x, GaussianSuffStat::new, |stat: GaussianSuffStat| {
            let post = posterior_from_stat(self, &stat);
            let n = stat.n() as f64;
            let lnz_n = ln_z(post.v, post.a, post.b);
            n.mul_add(-HALF_LN_2PI, lnz_n - cache)
            // lnz_n - cache - n * HALF_LN_PI - n*LN_2
        })
    }

    #[inline]
    fn ln_pp_cache(
        &self,
        x: &DataOrSuffStat<f64, Gaussian>,
    ) -> Self::LnPpCache {
        let stat = extract_stat(x, GaussianSuffStat::new);
        let post_n = posterior_from_stat(self, &stat);
        let lnz_n = ln_z(post_n.v, post_n.a, post_n.b);
        (stat, lnz_n)
    }

    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &f64) -> f64 {
        let mut stat = cache.0.clone();
        let lnz_n = cache.1;

        stat.observe(y);
        let post_m = posterior_from_stat(self, &stat);

        let lnz_m = ln_z(post_m.v, post_m.a, post_m.b);

        -HALF_LN_2PI + lnz_m - lnz_n
    }
}

gaussian_prior_geweke_testable!(NormalInvGamma, Gaussian);

#[cfg(test)]
mod test {
    use super::*;
    use crate::consts::LN_2PI;

    const TOL: f64 = 1E-12;

    #[test]
    fn geweke() {
        use crate::test::GewekeTester;

        let mut rng = rand::thread_rng();
        let pr = NormalInvGamma::new(0.1, 1.2, 0.5, 1.8).unwrap();
        let n_passes = (0..5)
            .map(|_| {
                let mut tester = GewekeTester::new(pr.clone(), 20);
                tester.run_chains(5_000, 20, &mut rng);
                u8::from(tester.eval(0.025).is_ok())
            })
            .sum::<u8>();
        assert!(n_passes > 1);
    }

    // Random reference I found using the same source
    // https://github.com/JuliaStats/ConjugatePriors.jl/blob/master/src/normalinversegamma.jl
    fn ln_f_ref(gauss: &Gaussian, nig: &NormalInvGamma) -> f64 {
        let (m, v, a, b) = nig.params();
        let mu = gauss.mu();
        let sigma = gauss.sigma();
        let sig2 = sigma * sigma;
        let lz_inv = a.mul_add(
            b.ln(),
            -(0.5_f64.mul_add(v.ln() + LN_2PI, a.ln_gamma().0)),
        );
        (0.5 / (sig2 * v) * (mu - m)).mul_add(
            -mu - m,
            (a + 1.).mul_add(-sig2.ln(), 0.5_f64.mul_add(-sig2.ln(), lz_inv))
                - b / sig2,
        )
    }

    fn post_params(
        xs: &[f64],
        m: f64,
        v: f64,
        a: f64,
        b: f64,
    ) -> (f64, f64, f64, f64) {
        let n = xs.len() as f64;
        let sum_x: f64 = xs.iter().sum();
        let sum_x_sq: f64 = xs.iter().map(|&x| x * x).sum();

        let v_inv = v.recip();
        let vn_inv = v_inv + n;
        let vn = vn_inv.recip();
        let mn = v_inv.mul_add(m, sum_x) * vn;
        let an = a + n / 2.0;
        let bn = 0.5_f64.mul_add(
            (mn * mn).mul_add(-vn_inv, (m * m).mul_add(v_inv, sum_x_sq)),
            b,
        );

        (mn, vn, an, bn)
    }

    // XXX: Implemented this directly against the Kevin Murphy whitepaper. Makes
    // things a little easier to understand compared to using the sufficient
    // statistics and traits and all that. Still possible for this to be wrong,
    // but if this matches what's in the code AND the DPGMM example (at
    // examples/dpgmm.rs) words with the NormalInvGamma prior, then we should be
    // good to go.
    fn alternate_ln_marginal(
        xs: &[f64],
        m: f64,
        v: f64,
        a: f64,
        b: f64,
    ) -> f64 {
        let n = xs.len() as f64;
        let (_, vn, an, bn) = post_params(xs, m, v, a, b);

        let numer = 0.5_f64.mul_add(vn.ln(), a * b.ln()) + an.ln_gamma().0;
        let denom = (n / 2.0).mul_add(
            LN_2PI,
            0.5_f64.mul_add(v.ln(), an * bn.ln()) + a.ln_gamma().0,
        );

        numer - denom
    }

    #[test]
    fn ln_f_vs_reference() {
        let (m, v, a, b) = (0.0, 1.2, 2.3, 3.4);
        let nig = NormalInvGamma::new(m, v, a, b).unwrap();
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let gauss = nig.draw(&mut rng);
            let ln_f = nig.ln_f(&gauss);
            let reference = ln_f_ref(&gauss, &nig);
            assert::close(reference, ln_f, TOL);
        }
    }

    #[test]
    fn ln_m_vs_reference() {
        let (m, v, a, b) = (0.0, 1.2, 2.3, 3.4);
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let reference = alternate_ln_marginal(&xs, m, v, a, b);
        let nig = NormalInvGamma::new(m, v, a, b).unwrap();
        let ln_m = nig.ln_m(&DataOrSuffStat::<f64, Gaussian>::from(&xs));

        assert::close(reference, ln_m, TOL);
    }

    #[test]
    fn ln_m_vs_monte_carlo() {
        use crate::misc::logsumexp;

        let n_samples = 1_000_000;
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (m, v, a, b) = (1.0, 2.2, 3.3, 4.4);
        let nig = NormalInvGamma::new(m, v, a, b).unwrap();
        let ln_m = nig.ln_m(&DataOrSuffStat::<f64, Gaussian>::from(&xs));
        // let ln_m = alternate_ln_marginal(&xs, m, v, a, b);

        let mc_est = {
            let ln_fs: Vec<f64> = nig
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
        use crate::dist::Gamma;
        use crate::misc::logsumexp;

        let n_samples = 1_000_000;
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (m, v, a, b) = (1.0, 2.2, 3.3, 4.4);
        let nig = NormalInvGamma::new(m, v, a, b).unwrap();
        let ln_m = nig.ln_m(&DataOrSuffStat::<f64, Gaussian>::from(&xs));

        let mc_est = {
            let mut rng = rand::thread_rng();
            let pr_m = Gaussian::new(1.0, 8.0).unwrap();
            let pr_s = Gamma::new(2.0, 0.4).unwrap();
            let ln_fs: Vec<f64> = (0..n_samples)
                .map(|_| {
                    let mu: f64 = pr_m.draw(&mut rng);
                    let var: f64 = pr_s.draw(&mut rng);
                    let gauss = Gaussian::new(mu, var.sqrt()).unwrap();
                    let ln_f = xs.iter().map(|x| gauss.ln_f(x)).sum::<f64>();
                    ln_f + nig.ln_f(&gauss) - pr_m.ln_f(&mu) - pr_s.ln_f(&var)
                })
                .collect();
            logsumexp(&ln_fs) - (n_samples as f64).ln()
        };
        // high error tolerance. MC estimation is not the most accurate...
        assert::close(ln_m, mc_est, 1e-2);
    }

    #[test]
    fn ln_pp_vs_monte_carlo() {
        use crate::misc::logsumexp;

        let n_samples = 1_000_000;
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let y: f64 = -0.3;
        let (m, v, a, b) = (1.0, 2.2, 3.3, 4.4);
        let nig = NormalInvGamma::new(m, v, a, b).unwrap();
        let post = nig.posterior(&DataOrSuffStat::<f64, Gaussian>::from(&xs));
        let ln_pp = nig.ln_pp(&y, &DataOrSuffStat::<f64, Gaussian>::from(&xs));
        // let ln_m = alternate_ln_marginal(&xs, m, v, a, b);

        let mc_est = {
            let ln_fs: Vec<f64> = post
                .sample_stream(&mut rand::thread_rng())
                .take(n_samples)
                .map(|gauss: Gaussian| gauss.ln_f(&y))
                .collect();
            logsumexp(&ln_fs) - (n_samples as f64).ln()
        };
        // high error tolerance. MC estimation is not the most accurate...
        assert::close(ln_pp, mc_est, 1e-2);
    }

    #[test]
    fn ln_pp_vs_ln_m_single() {
        let y: f64 = -0.3;
        let (m, v, a, b) = (0.0, 1.2, 2.3, 3.4);
        let nig = NormalInvGamma::new(m, v, a, b).unwrap();
        let ln_pp = nig.ln_pp(&y, &DataOrSuffStat::None);
        let ln_m = nig.ln_m(&DataOrSuffStat::from(&vec![y]));
        assert::close(ln_pp, ln_m, TOL);
    }

    #[test]
    fn ln_pp_vs_t() {
        use crate::dist::StudentsT;

        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: f64 = -0.3;
        let (m, v, a, b) = (0.0, 1.2, 2.3, 3.4);
        // let (m, v, a, b) = (0.0, 1.0, 1.0, 1.0);
        let (mn, vn, an, bn) = post_params(&xs, m, v, a, b);

        let ln_f_t = {
            // fit into non-shifted-and-scaled T using the parameterization in
            // 10.6 of the Kevin Murphy's whitepaper
            let t = StudentsT::new(2.0 * an).unwrap();
            let t_scale = bn * (1.0 + vn) / an;
            let t_shift = mn;
            let y_adj = (y - t_shift) / t_scale.sqrt();

            0.5_f64.mul_add(-t_scale.ln(), t.ln_f(&y_adj))
        };

        let ln_pp = {
            let nig = NormalInvGamma::new(m, v, a, b).unwrap();

            nig.ln_pp(&y, &DataOrSuffStat::<f64, Gaussian>::from(&xs))
        };
        assert::close(ln_f_t, ln_pp, TOL);
    }
}
