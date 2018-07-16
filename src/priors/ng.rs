extern crate rand;
extern crate special;

use std::f64::consts::LN_2;

use self::rand::Rng;
use self::special::Gamma as SGamma;

use consts::*;
use dist::{Gamma, Gaussian};
use suffstats::GaussianSuffStat;
use traits::*;

/// Prior for Gaussian
///
/// Given `x ~ N(μ, σ)`, the Normal Gamma prior implies that `μ ~ N(m, 1/(rρ))`
/// and `ρ ~ Gamma(ν/2, s/2)`.
pub struct NormalGamma {
    pub m: f64,
    pub r: f64,
    pub s: f64,
    pub v: f64,
}

impl NormalGamma {
    pub fn new(m: f64, r: f64, s: f64, v: f64) -> Self {
        NormalGamma {
            m: m,
            r: r,
            s: s,
            v: v,
        }
    }

    fn posterior_from_stat(&self, stat: &GaussianSuffStat) -> Self {
        let r = self.r + (stat.n as f64);
        let v = self.v + (stat.n as f64);
        let m = (self.m * self.r + stat.sum_x) / r;
        let s = self.s + stat.sum_x_sq + self.r * self.m * self.m - r * m * m;
        NormalGamma::new(m, r, s, v)
    }
}

impl Rv<Gaussian> for NormalGamma {
    fn ln_f(&self, x: &Gaussian) -> f64 {
        let rho = x.sigma.powi(2).recip();
        let lnf_rho = Gamma::new(self.v / 2.0, self.s / 2.0).ln_f(&rho);
        let prior_sigma = (self.r * rho).recip().sqrt();
        let lnf_mu = Gaussian::new(self.m, prior_sigma).ln_f(&x.mu);
        lnf_rho + lnf_mu
    }

    fn ln_normalizer(&self) -> f64 {
        HALF_LOG_2PI
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Gaussian {
        let rho: f64 = Gamma::new(self.v / 2.0, self.s / 2.0).draw(&mut rng);
        let post_sigma: f64 = (self.r * rho).recip().sqrt();
        let mu: f64 = Gaussian::new(self.m, post_sigma).draw(&mut rng);

        Gaussian::new(mu, rho.sqrt().recip())
    }
}

impl Support<Gaussian> for NormalGamma {
    fn contains(&self, x: &Gaussian) -> bool {
        x.mu.is_finite() && x.sigma > 0.0 && x.sigma.is_finite()
    }
}

impl HasSuffStat<f64> for NormalGamma {
    type Stat = GaussianSuffStat;
    fn empty_suffstat(&self) -> Self::Stat {
        GaussianSuffStat::new()
    }
}

impl ContinuousDistr<Gaussian> for NormalGamma {}

fn extract_stat(x: &DataOrSuffStat<f64, Gaussian>) -> GaussianSuffStat {
    match x {
        DataOrSuffStat::SuffStat(ref s) => (*s).clone(),
        DataOrSuffStat::Data(xs) => {
            let mut stat = GaussianSuffStat::new();
            xs.iter().for_each(|y| stat.observe(y));
            stat
        }
    }
}

pub fn ln_z(r: f64, s: f64, v: f64) -> f64 {
    (v + 1.0) / 2.0 * LN_2 + HALF_LOG_PI - 0.5 * r.ln() - (v / 2.0) * s.ln()
        + (v / 2.0).ln_gamma().0
}

impl ConjugatePrior<f64, Gaussian> for NormalGamma {
    fn posterior(&self, x: &DataOrSuffStat<f64, Gaussian>) -> NormalGamma {
        let stat = extract_stat(&x);
        self.posterior_from_stat(&stat)
    }

    fn ln_m(&self, x: &DataOrSuffStat<f64, Gaussian>) -> f64 {
        let stat = extract_stat(&x);
        let post = self.posterior_from_stat(&stat);
        let lnz_0 = ln_z(self.r, self.s, self.v);
        let lnz_n = ln_z(post.r, post.s, post.v);
        -(stat.n as f64) * HALF_LOG_2PI + lnz_n - lnz_0
    }

    fn ln_pp(&self, y: &f64, x: &DataOrSuffStat<f64, Gaussian>) -> f64 {
        let mut stat = extract_stat(&x);
        let post_n = self.posterior_from_stat(&stat);
        stat.observe(y);
        let post_m = self.posterior_from_stat(&stat);

        let lnz_n = ln_z(post_n.r, post_n.s, post_n.v);
        let lnz_m = ln_z(post_m.r, post_m.s, post_m.v);

        -HALF_LOG_2PI + lnz_m - lnz_n
    }
}
