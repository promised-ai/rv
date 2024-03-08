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

impl Default for SbdSuffStat {
    fn default() -> Self {
        Self::new()
    }
}

impl SbdSuffStat {
    pub fn new() -> Self {
        Self { counts: Vec::new() }
    }
}

impl HasSuffStat<usize> for Sbd {
    type Stat = SbdSuffStat;

    fn empty_suffstat() -> Self::Stat {
        Self::Stat::new()
    }

    fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
        let weights = &self.sticks.weights(stat.counts.len());
        let counts = &stat.counts;
        let pairs = weights.iter().zip(counts.iter());

        // This can probably be sped up later if necessary
        pairs.fold(0.0, |acc, (w, c)| (*c as f64).mul_add(w.ln(), acc))
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
