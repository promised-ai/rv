use crate::traits::SuffStat;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// A wrapper for sufficient statistics that accounts for a scale parameter
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct ScaledSuffStat<S> {
    parent: S,
    scale: f64,
    rate: f64,
}

impl<S> ScaledSuffStat<S> {
    /// Create a new ScaledSuffStat with the given parent statistic and scale
    pub fn new(parent: S, scale: f64) -> Self {
        ScaledSuffStat {
            parent,
            scale,
            rate: scale.recip(),
        }
    }

    /// Get a reference to the parent sufficient statistic
    pub fn parent(&self) -> &S {
        &self.parent
    }

    /// Get the scale parameter
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Get the rate parameter
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl<S> SuffStat<f64> for ScaledSuffStat<S>
where
    S: SuffStat<f64>,
{
    fn n(&self) -> usize {
        self.parent.n()
    }

    fn observe(&mut self, x: &f64) {
        // Scale the observation back to the parent's space
        self.parent.observe(&(x * self.rate));
    }

    fn forget(&mut self, x: &f64) {
        // Scale the observation back to the parent's space
        self.parent.forget(&(x * self.rate));
    }

    fn merge(&mut self, other: Self) {
        assert_eq!(
            self.scale, other.scale,
            "Cannot merge ScaledSuffStat with different scales"
        );
        self.parent.merge(other.parent);
    }
}
