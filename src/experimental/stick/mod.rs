pub mod sb;
pub mod sbd;
pub mod seq;
pub mod stat;

pub use sb::{BreakSequence, StickBreaking, StickWeights};
pub use sbd::StickBreakingDiscrete;
pub use seq::StickSequence;
pub use stat::StickBreakingDiscreteSuffStat;

use crate::{
    dist::UnitPowerLawError,
    traits::{HasDensity, InverseCdf, Sampleable, Support},
};

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct HalfBeta {
    pub alpha: f64,
    #[cfg_attr(feature = "serde1", serde(skip))]
    alpha_ln: f64,
}

impl HalfBeta {
    pub fn new(alpha: f64) -> Result<Self, UnitPowerLawError> {
        if alpha <= 0.0 {
            Err(UnitPowerLawError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(UnitPowerLawError::AlphaNotFinite { alpha })
        } else {
            Ok(Self {
                alpha,
                alpha_ln: alpha.ln(),
            })
        }
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn set_alpha(&mut self, alpha: f64) -> Result<(), UnitPowerLawError> {
        if alpha <= 0.0 {
            Err(UnitPowerLawError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(UnitPowerLawError::AlphaNotFinite { alpha })
        } else {
            self.alpha = alpha;
            self.alpha_ln = alpha.ln();
            Ok(())
        }
    }

    pub fn alpha_ln(&self) -> f64 {
        self.alpha_ln
    }
}

impl HasDensity<f64> for HalfBeta {
    fn ln_f(&self, x: &f64) -> f64 {
        (1.0 - *x).ln().mul_add(self.alpha - 1.0, self.alpha_ln())
    }
}

impl Support<f64> for HalfBeta {
    fn supports(&self, x: &f64) -> bool {
        0.0 <= *x && *x <= 1.0
    }
}

impl InverseCdf<f64> for HalfBeta {
    fn invcdf(&self, p: f64) -> f64 {
        1.0 - p.powf(self.alpha.recip())
    }
}

impl Sampleable<f64> for HalfBeta {
    fn draw<R: rand::Rng>(&self, rng: &mut R) -> f64 {
        let p: f64 = rng.random();
        self.invcdf(p)
    }
}
