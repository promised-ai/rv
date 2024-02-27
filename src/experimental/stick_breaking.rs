use crate::experimental::{StickBreakingSuffStat, StickSequence};
use crate::prelude::{UnitPowerLaw, UnitPowerLawError};
use crate::suffstat_traits::*;
use crate::traits::*;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct StickBreaking<B: Rv<f64> + Clone> {
    pub breaker: B,
}

impl StickBreaking<UnitPowerLaw> {
    pub fn new(
        alpha: f64,
    ) -> Result<StickBreaking<UnitPowerLaw>, UnitPowerLawError> {
        let breaker = UnitPowerLaw::new(alpha)?;
        Ok(Self { breaker })
    }
}

// impl<UnitPowerLaw> HasDensity<&[f64]> for StickBreaking<UnitPowerLaw> {
//     fn ln_f(&self, x: &&[f64]) -> f64 {
//         let stat = StickBreakingSuffStat::from(x);
//         self.ln_f_stat(&stat)
//     }
// }

impl<B: Rv<f64> + Clone> Sampleable<StickSequence<B>> for StickBreaking<B> {
    fn draw<R: Rng>(&self, rng: &mut R) -> StickSequence<B> {
        let seed: u64 = rng.gen();

        StickSequence::new(self.breaker, Some(seed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid_alpha() {
        let alpha = 1.5;
        let stick_breaking = StickBreaking::new(alpha).unwrap();
        assert_eq!(stick_breaking.breaker.alpha(), alpha);
    }

    #[test]
    fn test_new_invalid_alpha() {
        let alpha = -1.0;
        let result = StickBreaking::new(alpha);
        assert!(result.is_err());
    }
}
