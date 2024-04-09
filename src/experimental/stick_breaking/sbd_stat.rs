use crate::experimental::stick_breaking::sbd::StickBreakingDiscrete;
use crate::suffstat_traits::{HasSuffStat, SuffStat};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct StickBreakingDiscreteSuffStat {
    counts: Vec<usize>,
}

impl StickBreakingDiscreteSuffStat {
    pub fn new() -> Self {
        Self { counts: Vec::new() }
    }

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

    pub fn counts(&self) -> &Vec<usize> {
        &self.counts
    }
}

impl From<&[usize]> for StickBreakingDiscreteSuffStat {
    fn from(data: &[usize]) -> Self {
        let mut stat = StickBreakingDiscreteSuffStat::new();
        stat.observe_many(data);
        stat
    }
}

impl Default for StickBreakingDiscreteSuffStat {
    fn default() -> Self {
        Self::new()
    }
}

impl HasSuffStat<usize> for StickBreakingDiscrete {
    type Stat = StickBreakingDiscreteSuffStat;

    fn empty_suffstat(&self) -> Self::Stat {
        Self::Stat::new()
    }

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
    fn n(&self) -> usize {
        self.counts.iter().sum()
    }

    fn observe(&mut self, i: &usize) {
        if self.counts.len() < *i + 1 {
            self.counts.resize(*i + 1, 0)
        }
        self.counts[*i] += 1;
    }

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
