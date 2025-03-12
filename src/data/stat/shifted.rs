use crate::traits::SuffStat;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// A wrapper for sufficient statistics that accounts for a shift parameter
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct ShiftedSuffStat<S> {
    parent: S,
    shift: f64,
}

impl<S> ShiftedSuffStat<S> {
    /// Create a new ShiftedSuffStat with the given parent statistic and shift
    pub fn new(parent: S, shift: f64) -> Self {
        ShiftedSuffStat { parent, shift }
    }

    /// Get a reference to the parent sufficient statistic
    pub fn parent(&self) -> &S {
        &self.parent
    }

    /// Get the shift parameter
    pub fn shift(&self) -> f64 {
        self.shift
    }
}

impl<S> SuffStat<f64> for ShiftedSuffStat<S>
where
    S: SuffStat<f64>,
{
    fn n(&self) -> usize {
        self.parent.n()
    }

    fn observe(&mut self, x: &f64) {
        // Shift the observation back to the parent's space
        self.parent.observe(&(x - self.shift));
    }

    fn forget(&mut self, x: &f64) {
        // Shift the observation back to the parent's space
        self.parent.forget(&(x - self.shift));
    }

    fn merge(&mut self, other: Self) {
        assert_eq!(
            self.shift, other.shift,
            "Cannot merge ShiftedSuffStat with different shifts"
        );
        self.parent.merge(other.parent);
    }
}
