use crate::experimental::stick_breaking_process::sbd::StickBreakingDiscrete;
use crate::traits::{HasSuffStat, SuffStat};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Represents the sufficient statistics for a Stick-Breaking Discrete distribution.
///
/// This struct encapsulates the sufficient statistics for a Stick-Breaking Discrete distribution,
/// primarily involving a vector of counts representing the observed data.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct StickBreakingDiscreteSuffStat {
    /// A vector of counts for observed data.
    ///
    /// Each element represents the count of observations for a given category.
    counts: Vec<usize>,
}

impl StickBreakingDiscreteSuffStat {
    /// Constructs a new instance.
    ///
    /// Initializes a new `StickBreakingDiscreteSuffStat` with an empty vector of counts.
    ///
    /// # Returns
    ///
    /// A new `StickBreakingDiscreteSuffStat` instance.
    #[must_use]
    pub fn new() -> Self {
        Self { counts: Vec::new() }
    }

    #[must_use]
    pub fn from_counts(counts: Vec<usize>) -> Self {
        Self { counts }
    }

    /// Calculates break pairs for probabilities.
    ///
    /// Returns a vector of pairs where each pair consists of the sum of all counts after the current index and the count at the current index.
    ///
    /// # Returns
    ///
    /// A vector of `(usize, usize)` pairs for calculating probabilities.
    #[must_use]
    pub fn break_pairs(&self) -> Vec<(usize, usize)> {
        let mut s = self.counts.iter().sum();
        self.counts
            .iter()
            .map(|&x| {
                s -= x;
                (s, x)
            })
            .collect()
    }

    /// Provides read-only access to counts.
    ///
    /// # Returns
    ///
    /// A reference to the vector of counts.
    #[must_use]
    pub fn counts(&self) -> &Vec<usize> {
        &self.counts
    }
}

impl From<&[usize]> for StickBreakingDiscreteSuffStat {
    /// Constructs from a slice of counts.
    ///
    /// Allows creation from a slice of counts, converting raw observation data into a sufficient statistic.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of counts.
    ///
    /// # Returns
    ///
    /// A new `StickBreakingDiscreteSuffStat` instance.
    fn from(data: &[usize]) -> Self {
        let mut stat = StickBreakingDiscreteSuffStat::new();
        stat.observe_many(data);
        stat
    }
}

impl Default for StickBreakingDiscreteSuffStat {
    /// Returns a default instance.
    ///
    /// Equivalent to `new()`, for APIs requiring a default constructor.
    ///
    /// # Returns
    ///
    /// A default `StickBreakingDiscreteSuffStat` instance.
    fn default() -> Self {
        Self::new()
    }
}

impl HasSuffStat<usize> for StickBreakingDiscrete {
    type Stat = StickBreakingDiscreteSuffStat;

    /// Initializes an empty sufficient statistic.
    ///
    /// # Returns
    ///
    /// An empty `StickBreakingDiscreteSuffStat`.
    fn empty_suffstat(&self) -> Self::Stat {
        Self::Stat::new()
    }

    /// Calculates the log probability density of observed data.
    ///
    /// # Arguments
    ///
    /// * `stat` - A reference to the sufficient statistic.
    ///
    /// # Returns
    ///
    /// The natural logarithm of the probability of the observed data.
    fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
        self.stick_sequence()
            .weights(stat.counts.len())
            .0
            .iter()
            .zip(stat.counts.iter())
            .map(|(w, c)| (*c as f64) * w.ln())
            .sum()
    }
}

impl SuffStat<usize> for StickBreakingDiscreteSuffStat {
    /// Returns the total count of observations.
    ///
    /// # Returns
    ///
    /// The total count of all observed data.
    fn n(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Updates the statistic with a new observation.
    ///
    /// # Arguments
    ///
    /// * `i` - The index at which to increment the count.
    fn observe(&mut self, i: &usize) {
        if self.counts.len() < *i + 1 {
            self.counts.resize(*i + 1, 0);
        }
        self.counts[*i] += 1;
    }

    /// Removes a previously observed data point.
    ///
    /// # Arguments
    ///
    /// * `i` - The index at which to decrement the count.
    ///
    /// # Panics
    ///
    /// Panics if there are no observations of the specified category to forget.
    fn forget(&mut self, i: &usize) {
        assert!(self.counts[*i] > 0, "No observations of {i} to forget.");
        self.counts[*i] -= 1;
    }

    fn merge(&mut self, other: Self) {
        if other.counts.len() > self.counts.len() {
            self.counts.resize(other.counts.len(), 0);
        }
        self.counts
            .iter_mut()
            .zip(other.counts.iter())
            .for_each(|(ct_a, &ct_b)| *ct_a += ct_b);
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_break_pairs() {
        let suff_stat = StickBreakingDiscreteSuffStat {
            counts: vec![1, 2, 3],
        };

        let pairs = suff_stat.break_pairs();
        assert_eq!(pairs, vec![(5, 1), (3, 2), (0, 3)]);
    }

    // #[test]
    // fn test_ln_f_stat() {
    //     let sbd = StickBreakingDiscrete::new();
    //     let suff_stat = StickBreakingDiscreteSuffStat {
    //         counts: vec![1, 2, 3],
    //     };

    //     let ln_f_stat = sbd.ln_f_stat(&suff_stat);
    //     assert_eq!(ln_f_stat, 2.1972245773362196); // Replace with the expected value
    // }

    #[test]
    fn test_observe_and_forget() {
        let mut suff_stat = StickBreakingDiscreteSuffStat::new();

        suff_stat.observe(&1);
        suff_stat.observe(&2);
        suff_stat.observe(&2);
        suff_stat.forget(&2);

        assert_eq!(suff_stat.counts, vec![0, 1, 1]);
        assert_eq!(suff_stat.n(), 2);
    }

    #[test]
    fn test_new_is_default() {
        assert!(
            StickBreakingDiscreteSuffStat::new()
                == StickBreakingDiscreteSuffStat::default()
        );
    }
}
