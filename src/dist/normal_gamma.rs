//! A common conjugate prior for Gaussians
extern crate rand;

use self::rand::Rng;
use consts::HALF_LN_2PI;
use data::GaussianSuffStat;
use dist::{Gamma, Gaussian};
use std::io;
use traits::*;

/// Prior for Gaussian
///
/// Given `x ~ N(μ, σ)`, the Normal Gamma prior implies that `μ ~ N(m, 1/(rρ))`
/// and `ρ ~ Gamma(ν/2, s/2)`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NormalGamma {
    // TODO: document parameters
    pub m: f64,
    pub r: f64,
    pub s: f64,
    pub v: f64,
}

impl NormalGamma {
    pub fn new(m: f64, r: f64, s: f64, v: f64) -> io::Result<Self> {
        let m_ok = m.is_finite();
        let r_ok = r > 0.0 && r.is_finite();
        let s_ok = s > 0.0 && s.is_finite();
        let v_ok = v > 0.0 && v.is_finite();
        if !m_ok {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "m must be finite");
            Err(err)
        } else if r_ok && s_ok && v_ok {
            Ok(NormalGamma { m, r, s, v })
        } else {
            let err_kind = io::ErrorKind::InvalidInput;
            let msg = "r, s, and v must be finite and greater than zero";
            let err = io::Error::new(err_kind, msg);
            Err(err)
        }
    }
}

impl Rv<Gaussian> for NormalGamma {
    fn ln_f(&self, x: &Gaussian) -> f64 {
        let rho = x.sigma.powi(2).recip();
        let lnf_rho =
            Gamma::new(self.v / 2.0, self.s / 2.0).unwrap().ln_f(&rho);
        let prior_sigma = (self.r * rho).recip().sqrt();
        let lnf_mu = Gaussian::new(self.m, prior_sigma).unwrap().ln_f(&x.mu);
        lnf_rho + lnf_mu
    }

    #[inline]
    fn ln_normalizer() -> f64 {
        HALF_LN_2PI
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Gaussian {
        // NOTE: The parameter errors in this fn shouldn't happen if the prior
        // parameters are valid.
        let rho: f64 = Gamma::new(self.v / 2.0, self.s / 2.0)
            .expect("Invalid σ posterior params")
            .draw(&mut rng);
        let post_sigma: f64 = (self.r * rho).recip().sqrt();
        let mu: f64 = Gaussian::new(self.m, post_sigma)
            .expect("Invalid μ posterior params")
            .draw(&mut rng);

        Gaussian::new(mu, rho.sqrt().recip()).expect("Invalid params")
    }
}

impl Support<Gaussian> for NormalGamma {
    fn contains(&self, x: &Gaussian) -> bool {
        // NOTE: Could replace this with Gaussian::new(mu, sigma).is_ok(),
        // but this is more explicit.
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
