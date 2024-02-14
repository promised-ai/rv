use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

// use super::sb_stat::StickBreakingSuffStat;
use crate::dist::UnitPowerLaw;
use crate::experimental::stick_breaking_stat::StickBreakingSuffStat;
use crate::experimental::stick_sequence::StickSequence;
use crate::process_traits::Process;
use crate::suffstat_traits::*;
use crate::traits::Rv;
use rand::SeedableRng;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum StickBreakingError {
    InvalidAlpha(f64),
}

impl std::error::Error for StickBreakingError {}

impl std::fmt::Display for StickBreakingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAlpha(alpha) => {
                write!(
                    f,
                    "alpha ({}) must be finite and greater than zero",
                    alpha
                )
            }
        }
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct StickBreaking {
    pub alpha: f64,
}

impl StickBreaking {
    pub fn new(alpha: f64) -> Result<Self, StickBreakingError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(StickBreakingError::InvalidAlpha(alpha));
        }
        Ok(Self { alpha })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl Process<StickSequence, &[f64]> for StickBreaking {
    fn ln_f(&self, x: &&[f64]) -> f64 {
        let stat = StickBreakingSuffStat::from(x);
        self.ln_f_stat(&stat)
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> StickSequence {
        let seed: u64 = rng.gen();

        StickSequence::new(self.alpha(), Some(seed)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid_alpha() {
        let alpha = 1.5;
        let stick_breaking = StickBreaking::new(alpha).unwrap();
        assert_eq!(stick_breaking.alpha(), alpha);
    }

    #[test]
    fn test_new_invalid_alpha() {
        let alpha = -1.0;
        let result = StickBreaking::new(alpha);
        assert!(result.is_err());
    }

    #[test]
    fn test_ln_f() {
        let alpha = 1.5;
        let stick_breaking = StickBreaking::new(alpha).unwrap();
        let x = &[0.2, 0.3, 0.5] as &[f64];
        let ln_f = stick_breaking.ln_f(&x);
        assert_eq!(ln_f, 2.01);
    }

    #[test]
    fn test_draw() {
        let alpha = 1.5;
        let stick_breaking = StickBreaking::new(alpha).unwrap();
        let mut rng = rand::thread_rng();
        let seq = stick_breaking.draw(&mut rng);
        todo!()
    }
}
