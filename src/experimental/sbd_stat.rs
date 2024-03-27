use crate::experimental::sbd::Sbd;
use crate::suffstat_traits::{HasSuffStat, SuffStat};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct SbdSuffStat {
    pub counts: Vec<usize>,
}

impl SbdSuffStat {
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
}

impl From<&[usize]> for SbdSuffStat {
    fn from(data: &[usize]) -> Self {
        let mut stat = SbdSuffStat::new();
        stat.observe_many(data);
        stat
    }
}

impl Default for SbdSuffStat {
    fn default() -> Self {
        Self::new()
    }
}

impl HasSuffStat<usize> for Sbd {
    type Stat = SbdSuffStat;

    fn empty_suffstat(&self) -> Self::Stat {
        Self::Stat::new()
    }

    fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
        self.sticks
            .weights(stat.counts.len())
            .0
            .iter()
            .zip(stat.counts.iter())
            .fold(0.0, |acc, (w, c)| (*c as f64).mul_add(w.ln(), acc))
    }
}

impl SuffStat<usize> for SbdSuffStat {
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
        let suff_stat = SbdSuffStat {
            counts: vec![1, 2, 3],
        };

        let pairs = suff_stat.break_pairs();
        assert_eq!(pairs, vec![(5, 1), (3, 2), (0, 3)]);
    }

    // #[test]
    // fn test_ln_f_stat() {
    //     let sbd = Sbd::new();
    //     let suff_stat = SbdSuffStat {
    //         counts: vec![1, 2, 3],
    //     };

    //     let ln_f_stat = sbd.ln_f_stat(&suff_stat);
    //     assert_eq!(ln_f_stat, 2.1972245773362196); // Replace with the expected value
    // }

    // #[test]
    // fn test_observe_and_forget() {
    //     let mut suff_stat = SbdSuffStat::new();

    //     suff_stat.observe(&1);
    //     suff_stat.observe(&2);
    //     suff_stat.observe(&2);
    //     suff_stat.forget(&2);

    //     assert_eq!(suff_stat.counts, vec![0, 1, 1]);
    // }
}
