//! A common conjugate prior for Gaussians
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::consts::HALF_LN_2PI;
use crate::data::GaussianSuffStat;
use crate::dist::{Gamma, Gaussian};
use crate::impl_display;
use crate::result;
use crate::traits::*;
use getset::Setters;
use rand::Rng;
use std::f64::{MAX, MIN, MIN_POSITIVE};

/// Prior for Gaussian
///
/// Given `x ~ N(μ, σ)`, the Normal Gamma prior implies that `μ ~ N(m, 1/(rρ))`
/// and `ρ ~ Gamma(ν/2, s/2)`.
#[derive(Debug, Clone, PartialEq, PartialOrd, Setters)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NormalGamma {
    #[set = "pub"]
    m: f64,
    #[set = "pub"]
    r: f64,
    #[set = "pub"]
    s: f64,
    #[set = "pub"]
    v: f64,
}

impl NormalGamma {
    /// Create a new Normal Gamma distribution
    ///
    /// # Arguments
    /// - m: The prior mean
    /// - r: Relative precision of μ versus data
    /// - s: The mean of rho (the precision) is v/s.
    /// - v: Degrees of freedom of precision of rho
    pub fn new(m: f64, r: f64, s: f64, v: f64) -> result::Result<Self> {
        let m_ok = m.is_finite();
        let r_ok = r > 0.0 && r.is_finite();
        let s_ok = s > 0.0 && s.is_finite();
        let v_ok = v > 0.0 && v.is_finite();
        if !m_ok {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let err = result::Error::new(err_kind, "m must be finite");
            Err(err)
        } else if r_ok && s_ok && v_ok {
            Ok(NormalGamma { m, r, s, v })
        } else {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = format!("r ({}), s ({}), and v ({}) must be finite and greater than zero", r, s, v);
            let err = result::Error::new(err_kind, msg.as_str());
            Err(err)
        }
    }

    /// Creates a new NormalGamma without checking whether the parameters are
    /// valid.
    pub fn new_unchecked(m: f64, r: f64, s: f64, v: f64) -> Self {
        NormalGamma { m, r, s, v }
    }

    /// Get the m parameter
    pub fn m(&self) -> f64 {
        self.m
    }

    /// Get the r parameter
    pub fn r(&self) -> f64 {
        self.r
    }

    /// Get the s parameter
    pub fn s(&self) -> f64 {
        self.s
    }

    /// Get the v parameter
    pub fn v(&self) -> f64 {
        self.v
    }
}

impl From<&NormalGamma> for String {
    fn from(ng: &NormalGamma) -> String {
        format!(
            "Normal-Gamma(m: {}, r: {}, s: {}, ν: {})",
            ng.m, ng.r, ng.s, ng.v
        )
    }
}

impl_display!(NormalGamma);

// NOTE: We could use f64::clamp(MIN, MAX) to confine values if we were on
// nightly. Maybe we could do some conditional compilation?
impl Rv<Gaussian> for NormalGamma {
    fn ln_f(&self, x: &Gaussian) -> f64 {
        let rho = x.sigma().recip().powi(2);
        let lnf_rho =
            Gamma::new(self.v / 2.0, self.s / 2.0).unwrap().ln_f(&rho);
        let prior_sigma =
            (self.r * rho).recip().sqrt().max(MIN_POSITIVE).min(MAX);
        let lnf_mu = Gaussian::new(self.m, prior_sigma).unwrap().ln_f(&x.mu());
        lnf_rho + lnf_mu - HALF_LN_2PI
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Gaussian {
        let rho: f64 = Gamma::new(self.v / 2.0, self.s / 2.0)
            .expect("Invalid σ posterior params")
            .draw(&mut rng);
        let post_sigma: f64 =
            (self.r * rho).recip().sqrt().max(MIN_POSITIVE).min(MAX);
        let mu: f64 = Gaussian::new(self.m, post_sigma)
            .expect("Invalid μ posterior params")
            .draw(&mut rng);
        let mu_clamped = mu.max(MIN).min(MAX);

        // XXX: Underflow is a problem here
        let sigma = rho.recip().sqrt().max(MIN_POSITIVE).min(MAX);
        Gaussian::new(mu_clamped, sigma).expect("Invalid params")
    }
}

impl Support<Gaussian> for NormalGamma {
    fn supports(&self, x: &Gaussian) -> bool {
        // NOTE: Could replace this with Gaussian::new(mu, sigma).is_ok(),
        // but this is more explicit.
        x.mu().is_finite() && x.sigma() > 0.0 && x.sigma().is_finite()
    }
}

impl HasSuffStat<f64> for NormalGamma {
    type Stat = GaussianSuffStat;
    fn empty_suffstat(&self) -> Self::Stat {
        GaussianSuffStat::new()
    }
}

impl ContinuousDistr<Gaussian> for NormalGamma {}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn never_a_bad_draw() {
        let mut rng = rand::thread_rng();
        let ng_inf = NormalGamma::new(0.0, 0.000_1, 1_000.0, 0.000_1).unwrap();

        for _ in 0..100_000 {
            let gauss: Gaussian = ng_inf.draw(&mut rng);
            assert!(gauss.sigma() > 0.0);
        }

        let ng_zero = NormalGamma::new(0.0, 1e84, 1e-75, 1e84).unwrap();

        for _ in 0..100_000 {
            let gauss: Gaussian = ng_zero.draw(&mut rng);
            assert!(gauss.sigma() > 0.0);
        }
    }
}
