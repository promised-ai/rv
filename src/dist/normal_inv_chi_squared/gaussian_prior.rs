use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::consts::HALF_LN_PI;
use crate::data::{extract_stat, extract_stat_then, GaussianSuffStat};
use crate::dist::{Gaussian, NormalInvChiSquared};
use crate::gaussian_prior_geweke_testable;

use crate::misc::ln_gammafn;
use crate::test::GewekeTestable;
use crate::traits::*;

// XXX: Check out section 6.3 from Kevin Murphy's paper
// https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
fn posterior_from_stat(
    nix: &NormalInvChiSquared,
    stat: &GaussianSuffStat,
) -> NormalInvChiSquared {
    if stat.n() == 0 {
        return nix.clone();
    }

    let n = stat.n() as f64;

    let (m, k, v, s2) = nix.params();

    let xbar = stat.mean();
    let sum_x = xbar * n;
    // Sum (x - xbar)^2
    //   = Sum[ x*x - 2x*xbar + xbar*xbar ]
    //   = Sum[x^2] + n * xbar^2 - 2 * xbar + Sum[x]
    //   = Sum[x^2] + n * xbar^2 - 2 * n *xbar^2
    //   = Sum[x^2] - n * xbar^2
    let mid = (n * xbar).mul_add(-xbar, stat.sum_x_sq());

    let kn = k + n;
    let divby_k_plus_n = kn.recip();
    let vn = v + n;
    let mn = k.mul_add(m, sum_x) * divby_k_plus_n;
    let diff_m_xbar = m - xbar;
    let s2n = ((n * k * divby_k_plus_n) * diff_m_xbar)
        .mul_add(diff_m_xbar, v.mul_add(s2, mid))
        / vn;

    NormalInvChiSquared::new(mn, kn, vn, s2n)
        .expect("Invalid posterior params.")
}

impl ConjugatePrior<f64, Gaussian> for NormalInvChiSquared {
    type Posterior = Self;
    type MCache = f64;
    type PpCache = (NormalInvChiSquared, f64);

    fn posterior(&self, x: &DataOrSuffStat<f64, Gaussian>) -> Self {
        extract_stat_then(x, GaussianSuffStat::new, |stat: GaussianSuffStat| {
            posterior_from_stat(self, &stat)
        })
    }

    #[inline]
    fn ln_m_cache(&self) -> Self::MCache {
        self.ln_z()
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::MCache,
        x: &DataOrSuffStat<f64, Gaussian>,
    ) -> f64 {
        extract_stat_then(x, GaussianSuffStat::new, |stat: GaussianSuffStat| {
            let n = stat.n() as f64;
            let post = posterior_from_stat(self, &stat);
            let lnz_n = post.ln_z();
            n.mul_add(-HALF_LN_PI, lnz_n - cache)
        })
    }

    #[inline]
    fn ln_pp_cache(&self, x: &DataOrSuffStat<f64, Gaussian>) -> Self::PpCache {
        let stat = extract_stat(x, GaussianSuffStat::new);
        let post_n = posterior_from_stat(self, &stat);
        let kn = post_n.k;
        let vn = post_n.v;
        let half_vn = 0.5 * vn;
        let s2n = post_n.s2;

        let z1 = ln_gammafn(half_vn + 0.5) - ln_gammafn(half_vn);
        let z2 = 0.5 * (kn / ((kn + 1.0) * PI * vn * s2n)).ln();
        (post_n, z1 + z2)
    }

    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &f64) -> f64 {
        let post = &cache.0;
        let z = cache.1;
        let kn = post.k;

        let diff = y - post.m;

        // z - ((post.v + 1.0) / 2.0)
        //     * (1.0 + (kn * diff * diff) / ((kn + 1.0) * post.v * post.s2)).ln()
        // z - ((post.v + 1.0) / 2.0)
        //     * ((kn * diff * diff) / ((kn + 1.0) * post.v * post.s2)).ln_1p()
        ((post.v + 1.0) / 2.0).mul_add(
            -((kn * diff * diff) / ((kn + 1.0) * post.v * post.s2)).ln_1p(),
            z,
        )
    }
}

gaussian_prior_geweke_testable!(NormalInvChiSquared, Gaussian);

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_conjugate_prior;

    const TOL: f64 = 1E-12;

    test_conjugate_prior!(
        f64,
        Gaussian,
        NormalInvChiSquared,
        NormalInvChiSquared::new(0.1, 1.2, 0.5, 1.8).unwrap()
    );

    #[test]
    fn geweke() {
        use crate::test::GewekeTester;

        let mut rng = rand::thread_rng();
        let pr = NormalInvChiSquared::new(0.1, 1.2, 0.5, 1.8).unwrap();
        let n_passes = (0..5)
            .map(|_| {
                let mut tester = GewekeTester::new(pr.clone(), 20);
                tester.run_chains(5_000, 20, &mut rng);
                if tester.eval(0.025).is_ok() {
                    1_u8
                } else {
                    0_u8
                }
            })
            .sum::<u8>();
        assert!(n_passes > 1);
    }

    fn post_params(
        xs: &[f64],
        m: f64,
        k: f64,
        v: f64,
        s2: f64,
    ) -> (f64, f64, f64, f64) {
        let n = xs.len() as f64;
        let sum_x: f64 = xs.iter().sum();
        let mean = sum_x / n;
        let sse: f64 = xs.iter().map(|&x| (x - mean) * (x - mean)).sum();

        let kn = k + n;
        let vn = v + n;
        let mn = k.mul_add(m, sum_x) / kn;
        let s2n = ((n * k / kn) * (m - mean))
            .mul_add(m - mean, v.mul_add(s2, sse))
            / vn;

        (mn, kn, vn, s2n)
    }

    use crate::misc::ln_gammafn;
    // XXX: Implemented this directly against the Kevin Murphy whitepaper. Makes
    // things a little easier to understand compared to using the sufficient
    // statistics and traits and all that. Still possible for this to be wrong,
    // but if this matches what's in the code AND the DPGMM example (at
    // examples/dpgmm.rs) words with the NormalInvGamma prior, then we should be
    // good to go.
    fn alternate_ln_marginal(
        xs: &[f64],
        m: f64,
        k: f64,
        v: f64,
        s2: f64,
    ) -> f64 {
        let n = xs.len() as f64;
        let (_, kn, vn, s2n) = post_params(xs, m, k, v, s2);

        (n / 2.).mul_add(
            -1.144_729_885_849_399,
            (0.5 * vn).mul_add(
                -(vn * s2n).ln(),
                (0.5 * v).mul_add(
                    (v * s2).ln(),
                    0.5_f64.mul_add(
                        (k / kn).ln(),
                        ln_gammafn(vn / 2.) - ln_gammafn(v / 2.),
                    ),
                ),
            ),
        )
    }

    #[test]
    fn ln_m_vs_reference() {
        let (m, k, v, s2) = (0.0, 1.2, 2.3, 3.4);
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let reference = alternate_ln_marginal(&xs, m, k, v, s2);
        let nix = NormalInvChiSquared::new(m, k, v, s2).unwrap();
        let ln_m = nix.ln_m(&DataOrSuffStat::<f64, Gaussian>::from(&xs));

        assert::close(reference, ln_m, TOL);
    }

    #[test]
    fn post_params_vs_reference() {
        let (m, k, v, s2) = (0.0, 1.2, 2.3, 3.4);
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mn, kn, vn, s2n) = post_params(&xs, m, k, v, s2);
        let nix = NormalInvChiSquared::new(m, k, v, s2).unwrap();
        let post = nix.posterior(&DataOrSuffStat::<f64, Gaussian>::from(&xs));

        assert::close(mn, post.m(), TOL);
        assert::close(vn, post.v(), TOL);
        assert::close(kn, post.k(), TOL);
        assert::close(s2n, post.s2(), TOL);
    }

    #[test]
    fn posterior_of_nothing_is_prior() {
        let prior = NormalInvChiSquared::new_unchecked(1.2, 2.3, 3.4, 4.5);
        let post = prior.posterior(&DataOrSuffStat::from(&vec![]));
        assert_eq!(prior.m(), post.m());
        assert_eq!(prior.k(), post.k());
        assert_eq!(prior.v(), post.v());
        assert_eq!(prior.s2(), post.s2());
    }

    #[test]
    fn ln_m_single_datum_vs_monte_carlo() {
        use crate::misc::LogSumExp;

        let n_samples = 1_000_000;
        let x: f64 = -0.3;
        let xs = vec![x];

        let (m, k, v, s2) = (1.0, 2.2, 3.3, 4.4);
        let nix = NormalInvChiSquared::new(m, k, v, s2).unwrap();
        let ln_m = nix.ln_m(&DataOrSuffStat::<f64, Gaussian>::from(&xs));

        let mc_est = {
            nix.sample_stream(&mut rand::thread_rng())
                .take(n_samples)
                .map(|gauss: Gaussian| gauss.ln_f(&x))
                .logsumexp()
                - (n_samples as f64).ln()
        };
        // high error tolerance. MC estimation is not the most accurate...
        assert::close(ln_m, mc_est, 1e-2);
    }

    #[test]
    fn ln_m_vs_monte_carlo() {
        use crate::misc::LogSumExp;

        let n_samples = 1_000_000;
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (m, k, v, s2) = (1.0, 2.2, 3.3, 4.4);
        let nix = NormalInvChiSquared::new(m, k, v, s2).unwrap();
        let ln_m = nix.ln_m(&DataOrSuffStat::<f64, Gaussian>::from(&xs));

        let mc_est = {
            nix.sample_stream(&mut rand::thread_rng())
                .take(n_samples)
                .map(|gauss: Gaussian| {
                    xs.iter().map(|x| gauss.ln_f(x)).sum::<f64>()
                })
                .logsumexp()
                - (n_samples as f64).ln()
        };
        // high error tolerance. MC estimation is not the most accurate...
        assert::close(ln_m, mc_est, 1e-2);
    }

    #[test]
    fn ln_pp_vs_monte_carlo() {
        use crate::misc::LogSumExp;

        let n_samples = 1_000_000;
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let y: f64 = -0.3;
        let (m, k, v, s2) = (1.0, 2.2, 3.3, 4.4);
        let nix = NormalInvChiSquared::new(m, k, v, s2).unwrap();
        let post = nix.posterior(&DataOrSuffStat::<f64, Gaussian>::from(&xs));
        let ln_pp = nix.ln_pp(&y, &DataOrSuffStat::<f64, Gaussian>::from(&xs));

        let mc_est = {
            post.sample_stream(&mut rand::thread_rng())
                .take(n_samples)
                .map(|gauss: Gaussian| gauss.ln_f(&y))
                .logsumexp()
                - (n_samples as f64).ln()
        };
        // high error tolerance. MC estimation is not the most accurate...
        assert::close(ln_pp, mc_est, 1e-2);
    }

    #[test]
    fn ln_pp_single_vs_monte_carlo() {
        use crate::misc::LogSumExp;

        let n_samples = 1_000_000;
        let x: f64 = -0.3;

        let (m, k, v, s2) = (1.0, 2.2, 3.3, 4.4);
        let nix = NormalInvChiSquared::new(m, k, v, s2).unwrap();
        let ln_pp =
            nix.ln_pp(&x, &DataOrSuffStat::<f64, Gaussian>::from(&vec![]));

        let mc_est = {
            nix.sample_stream(&mut rand::thread_rng())
                .take(n_samples)
                .map(|gauss: Gaussian| gauss.ln_f(&x))
                .logsumexp()
                - (n_samples as f64).ln()
        };
        // high error tolerance. MC estimation is not the most accurate...
        assert::close(ln_pp, mc_est, 1e-2);
    }

    #[test]
    fn ln_pp_vs_ln_m_single() {
        // The log posterior predictive of p(x | nothing) should be the same as
        // p(x).
        let y: f64 = -0.3;

        let (m, k, v, s2) = (1.0, 2.2, 3.3, 4.4);

        let nix = NormalInvChiSquared::new(m, k, v, s2).unwrap();

        let (ln_pp, ln_m) = {
            let ys = vec![y];
            let new_vec = Vec::new();
            let data = DataOrSuffStat::<f64, Gaussian>::from(&new_vec);
            let y_data = DataOrSuffStat::<f64, Gaussian>::from(&ys);
            (nix.ln_pp(&y, &data), nix.ln_m(&y_data))
        };
        assert::close(ln_m, ln_pp, TOL);
    }

    #[test]
    fn ln_pp_vs_t() {
        use crate::dist::StudentsT;

        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: f64 = -0.3;

        let (m, k, v, s2) = (1.0, 2.2, 3.3, 4.4);

        let nix = NormalInvChiSquared::new(m, k, v, s2).unwrap();

        let data = DataOrSuffStat::<f64, Gaussian>::from(&xs);
        let (mn, kn, vn, s2n) = nix.posterior(&data).params();

        let ln_f_t = {
            // fit into non-shifted-and-scaled T using the parameterization in
            // 10.6 of the Kevin Murphy's whitepaper
            let t = StudentsT::new(vn).unwrap();
            let t_scale = (1.0 + kn) * s2n / kn;
            let t_shift = mn;
            let y_adj = (y - t_shift) / t_scale.sqrt();

            0.5_f64.mul_add(-t_scale.ln(), t.ln_f(&y_adj))
        };

        let ln_pp = nix.ln_pp(&y, &data);
        assert::close(ln_f_t, ln_pp, TOL);
    }
}
