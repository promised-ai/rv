use crate::experimental::stick_breaking::sbd::StickBreakingDiscrete;
use crate::suffstat_traits::{HasSuffStat, SuffStat};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// A struct representing the sufficient statistics for a Stick-Breaking Discrete distribution.
/// 
/// This struct is used to encapsulate the sufficient statistics for a Stick-Breaking Discrete distribution,
/// which primarily involves a vector of counts representing the observed data.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct StickBreakingDiscreteSuffStat {
    /// A vector of counts representing the observed data.
    /// Each element in the vector represents the count of observations for a given category.
    counts: Vec<usize>,
}

impl StickBreakingDiscreteSuffStat {
    /// Constructs a new `StickBreakingDiscreteSuffStat`.
    /// 
    /// This constructor initializes a new instance of `StickBreakingDiscreteSuffStat` with an empty vector of counts.
    pub fn new() -> Self {
        Self { counts: Vec::new() }
    }

    /// Returns a vector of pairs where each pair consists of the sum of all counts after the current index and the count at the current index.
    /// 
    /// This method is useful for calculating probabilities in a Stick-Breaking Discrete distribution, where the probability
    /// of a category depends on the counts of subsequent categories.
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

    /// Returns a reference to the vector of counts.
    /// 
    /// This method provides read-only access to the internal vector of counts, allowing for inspection without modification.
    pub fn counts(&self) -> &Vec<usize> {
        &self.counts
    }
}

impl From<&[usize]> for StickBreakingDiscreteSuffStat {
    /// Constructs a `StickBreakingDiscreteSuffStat` from a slice of counts.
    /// 
    /// This constructor allows for the creation of a `StickBreakingDiscreteSuffStat` instance from a slice of counts,
    /// effectively converting raw observation data into a sufficient statistic for further analysis.
    fn from(data: &[usize]) -> Self {
        let mut stat = StickBreakingDiscreteSuffStat::new();
        stat.observe_many(data);
        stat
    }
}

impl Default for StickBreakingDiscreteSuffStat {
    /// Returns a default instance of `StickBreakingDiscreteSuffStat`, equivalent to `new()`.
    /// 
    /// This implementation allows for the use of `StickBreakingDiscreteSuffStat` with APIs that require a default constructor.
    fn default() -> Self {
        Self::new()
    }
}

impl HasSuffStat<usize> for StickBreakingDiscrete {
    type Stat = StickBreakingDiscreteSuffStat;

    /// Returns an empty sufficient statistic for the Stick-Breaking Discrete distribution.
    /// 
    /// This method is part of the `HasSuffStat` trait implementation, providing a way to initialize
    /// an empty sufficient statistic for the Stick-Breaking Discrete distribution.
    fn empty_suffstat(&self) -> Self::Stat {
        Self::Stat::new()
    }

    /// Computes the natural logarithm of the probability of the observed data under the Stick-Breaking Discrete distribution.
    /// 
    /// This method calculates the log probability of the observed data given the Stick-Breaking Discrete distribution,
    /// using the sufficient statistics encapsulated by `StickBreakingDiscreteSuffStat`.
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
    /// Returns the total count of all observed data.
    /// 
    /// This method sums up all the counts in the vector, providing the total number of observations represented by this sufficient statistic.
    fn n(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Observes a new data point, incrementing the count at the specified index.
    /// 
    /// This method is used to update the sufficient statistic with a new observation, incrementing the count for the specified category.
    fn observe(&mut self, i: &usize) {
        if self.counts.len() < *i + 1 {
            self.counts.resize(*i + 1, 0)
        }
        self.counts[*i] += 1;
    }

    /// Forgets a previously observed data point, decrementing the count at the specified index.
    /// 
    /// This method allows for the removal of a previously observed data point from the sufficient statistic, decrementing the count for the specified category.
    /// It asserts that there was at least one observation of the specified category to forget.
    fn forget(&mut self, i: &usize) {
        assert!(self.counts[*i] > 0, "No observations of {i} to forget.");
        self.counts[*i] -= 1;
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
