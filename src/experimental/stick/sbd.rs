use super::HalfBeta;
use super::StickSequence;
use crate::dist::Mixture;
use crate::dist::UnitPowerLawError;
use crate::misc::ConvergentSequence;
use crate::traits::{
    DiscreteDistr, Entropy, HasDensity, InverseCdf, Sampleable, Support,
};
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// A "Stick-breaking discrete" distribution parameterized by a `StickSequence`.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct StickBreakingDiscrete {
    sticks: StickSequence,
}

impl StickBreakingDiscrete {
    /// Creates a new instance of `StickBreakingDiscrete` with the specified
    /// `StickSequence`.
    ///
    /// # Arguments
    /// - `sticks` - The `StickSequence` used for generating random numbers.
    ///
    /// # Returns
    /// A new instance of `StickBreakingDiscrete`.
    pub fn new(sticks: StickSequence) -> Self {
        Self { sticks }
    }

    pub fn from_alpha(
        alpha: f64,
        seed: Option<u64>,
    ) -> Result<Self, UnitPowerLawError> {
        let breaker = HalfBeta::new(alpha)?;

        Ok(Self {
            sticks: StickSequence::new(breaker, seed),
        })
    }

    /// Provides a reference to the underlying `StickSequence`
    pub fn stick_sequence(&self) -> &StickSequence {
        &self.sticks
    }
}

impl Support<usize> for StickBreakingDiscrete {
    fn supports(&self, _: &usize) -> bool {
        true
    }
}

impl DiscreteDistr<usize> for StickBreakingDiscrete {}

impl HasDensity<usize> for StickBreakingDiscrete {
    fn f(&self, n: &usize) -> f64 {
        self.stick_sequence().ensure_breaks(*n + 1);
        self.sticks.weight(*n)
    }

    fn ln_f(&self, n: &usize) -> f64 {
        self.f(n).ln()
    }
}

impl InverseCdf<usize> for StickBreakingDiscrete {
    fn invcdf(&self, p: f64) -> usize {
        self.stick_sequence().ensure_rm_mass(1.0 - p);
        self.sticks.with_inner(|inner| {
            let mut cdf = 0.0;
            for (i, w) in inner.weights.iter().enumerate() {
                cdf += w;
                if p < cdf {
                    return i;
                }
            }
            return inner.weights.len();
        })
    }
}

impl Sampleable<usize> for StickBreakingDiscrete {
    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        let u: f64 = rng.random();
        self.stick_sequence().ensure_rm_mass(1.0 - u);
        self.invcdf(u)
    }
}

impl Entropy for StickBreakingDiscrete {
    fn entropy(&self) -> f64 {
        let probs = (0..).map(|n| self.f(&n));
        probs
            .map(|p| p * p.ln())
            .scan(0.0, |state, x| {
                *state -= x;
                Some(*state)
            })
            .limit(1e-10)
    }
}

impl Entropy for &Mixture<StickBreakingDiscrete> {
    fn entropy(&self) -> f64 {
        let probs = (0..).map(|n| self.f(&n));
        probs
            .map(|p| p * p.ln())
            .scan(0.0, |state, x| {
                *state -= x;
                Some(*state)
            })
            .limit(1e-10)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[cfg(feature = "serde1")]
    #[test]
    fn seed_control_after_write_read() {
        let sbd0 = StickBreakingDiscrete::from_alpha(2.0, Some(1337)).unwrap();
        let json = serde_json::to_string(&sbd0).unwrap();
        let sbd1: StickBreakingDiscrete = serde_json::from_str(&json).unwrap();
        let sbd2 =
            StickBreakingDiscrete::from_alpha(2.0, Some(8675309)).unwrap();

        // At this point each stick sequence should have zero instantiated
        // weights, so when a z comes in that is greater than the number of
        // instantiated weights, the sequence will break the stick using the
        // internal RNG. If the RNG states are the same, the `ln_f` values
        // should be the same. Otherwise, they will not
        for z in 0..20 {
            let ln_f0 = sbd0.ln_f(&z);
            let ln_f1 = sbd1.ln_f(&z);
            let ln_f2 = sbd2.ln_f(&z);

            // Same seed
            assert::close(ln_f0, ln_f1, 1e-12);

            // Different seed should fail (testing that the test holds)
            assert!((1.0 - ln_f2 / ln_f0).abs() > 0.01);
        }
    }
}
