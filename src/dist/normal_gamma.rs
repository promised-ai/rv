extern crate rand;

use self::rand::Rng;
use consts::HALF_LOG_2PI;
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
        NormalGamma { m, r, s, v }
    }
}

impl Rv<Gaussian> for NormalGamma {
    fn ln_f(&self, x: &Gaussian) -> f64 {
        let rho = x.sigma.powi(2).recip();
        let lnf_rho =
            Gamma::new(self.v / 2.0, self.s / 2.0).unwrap().ln_f(&rho);
        let prior_sigma = (self.r * rho).recip().sqrt();
        let lnf_mu = Gaussian::new(self.m, prior_sigma).ln_f(&x.mu);
        lnf_rho + lnf_mu
    }

    fn ln_normalizer(&self) -> f64 {
        HALF_LOG_2PI
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Gaussian {
        let rho: f64 = Gamma::new(self.v / 2.0, self.s / 2.0)
            .unwrap()
            .draw(&mut rng);
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
