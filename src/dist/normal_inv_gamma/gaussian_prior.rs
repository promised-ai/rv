use std::f64::consts::LN_2;

use special::Gamma as _;

use crate::consts::{HALF_LN_PI, LN_PI};
use crate::data::{DataOrSuffStat, GaussianSuffStat};
use crate::dist::{Gaussian, NormalInvGamma};
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

#[inline]
fn ln_z(v: f64, a: f64, b: f64) -> f64 {
    -(a * b.ln() - 0.5 * v.ln() - a.ln_gamma().0)
}

// XXX: Check out section 6.3 from Kevin Murphy's paper
// https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
#[allow(clippy::clippy::many_single_char_names)]
fn posterior_from_stat(
    nig: &NormalInvGamma,
    stat: &GaussianSuffStat,
) -> NormalInvGamma {
    let n = stat.n() as f64;

    let (m, v, a, b) = nig.params();

    let v_inv = v.recip();

    let vn_inv = v_inv + n;
    let vn = vn_inv.recip();
    let mn = (v_inv * m + stat.sum_x()) * vn;
    let an = a + 0.5 * n;
    let bn = b + 0.5 * (m * m * v_inv + stat.sum_x_sq() - mn * mn * vn_inv);

    NormalInvGamma::new(mn, vn, an, bn).expect("Invalid posterior params.")
}

impl ConjugatePrior<f64, Gaussian> for NormalInvGamma {
    type Posterior = Self;
    type LnMCache = f64;
    type LnPpCache = (GaussianSuffStat, f64);

    fn posterior(&self, x: &DataOrSuffStat<f64, Gaussian>) -> Self {
        extract_stat_then!(x, |stat: &GaussianSuffStat| {
            posterior_from_stat(&self, &stat)
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
        extract_stat_then!(x, |stat: &GaussianSuffStat| {
            let post = posterior_from_stat(&self, &stat);
            let n = stat.n() as f64;
            let lnz_n = ln_z(post.v, post.a, post.b);
            lnz_n - cache - 0.5 * n * LN_PI - n * LN_2
        })
    }

    #[inline]
    fn ln_pp_cache(
        &self,
        x: &DataOrSuffStat<f64, Gaussian>,
    ) -> Self::LnPpCache {
        let stat = extract_stat(&x);
        let post_n = posterior_from_stat(&self, &stat);
        let lnz_n = ln_z(post_n.v, post_n.a, post_n.b);
        (stat, lnz_n)
    }

    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &f64) -> f64 {
        let mut stat = cache.0.clone();
        let lnz_n = cache.1;

        stat.observe(y);
        let post_m = posterior_from_stat(&self, &stat);

        let lnz_m = ln_z(post_m.v, post_m.a, post_m.b);

        HALF_LN_PI + LN_2 + lnz_m - lnz_n
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
mod test {
    use super::*;

    const TOL: f64 = 1E-12;

    // XXX: Implemented this directly against the Kevin Murphy whitepaper. Makes
    // things a little easier to understand compared to using the sufficient
    // statistics and traits and all that. Still possible for this to be wrong,
    // but if this matches what's in the code AND the DPGMM example (at
    // examples/dpgmm.rs) words with the NormalInvGamma prior, then we should be
    // good to go.
    fn alternate_ln_marginal(
        xs: &Vec<f64>,
        m: f64,
        v: f64,
        a: f64,
        b: f64,
    ) -> f64 {
        let n = xs.len() as f64;
        let sum_x: f64 = xs.iter().sum();
        let sum_x_sq: f64 = xs.iter().map(|&x| x * x).sum();

        let v_inv = v.recip();
        let vn_inv = v_inv + n;
        let vn = vn_inv.recip();
        let mn = (v_inv * m + sum_x) * vn;
        let an = a + n / 2.0;
        let bn = b + 0.5 * (m * m * v_inv + sum_x_sq - mn * mn * vn_inv);

        let numer = 0.5 * vn.ln() + a * b.ln() + an.ln_gamma().0;
        let denom = 0.5 * v.ln()
            + an * bn.ln()
            + a.ln_gamma().0
            + (n / 2.0) * LN_PI
            + n * LN_2;

        numer - denom
    }

    #[test]
    fn ln_m_vs_reference() {
        let (m, v, a, b) = (0.0, 1.0, 1.0, 1.0);
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let reference = alternate_ln_marginal(&xs, m, v, a, b);
        let nig = NormalInvGamma::new(m, v, a, b).unwrap();
        let ln_m = nig.ln_m(&DataOrSuffStat::<f64, Gaussian>::from(&xs));

        assert::close(reference, ln_m, TOL);
    }
}
