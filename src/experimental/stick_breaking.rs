use crate::experimental::{StickBreakingSuffStat, StickSequence};
use crate::traits::*;
use crate::suffstat_traits::*;
use rand::Rng;

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

impl HasDensity<&[f64]> for StickBreaking {
    fn ln_f(&self, x: &&[f64]) -> f64 {
        let stat = StickBreakingSuffStat::from(x);
        self.ln_f_stat(&stat)
    }
}

impl Sampleable<StickSequence> for StickBreaking {
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
}
