use crate::traits::SuffStat;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SbdSuffStat {
    n: usize,
    counts: Vec<usize>,
}

impl Default for SbdSuffStat {
    fn default() -> Self {
        Self::new()
    }
}

impl SbdSuffStat {
    pub fn new() -> Self {
        Self {
            n: 0,
            counts: Vec::new(),
        }
    }

    pub fn counts(&self) -> &Vec<usize> {
        &self.counts
    }
}

impl SuffStat<usize> for SbdSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &usize) {
        self.n += 1;
        if *x >= self.counts.len() {
            self.counts.resize(*x + 1, 0)
        }
        self.counts[*x] += 1;
    }

    fn forget(&mut self, x: &usize) {
        if self.n == 1 {
            self.counts = Vec::new();
            self.n = 0;
        } else {
            self.counts[*x] -= 1;
            self.n -= 1;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    //     #[test]
    //     fn stat_observe_forget_some() {
    //         let mut stat = SbdSuffStat::new();

    //         stat.observe(&0);
    //         stat.observe(&0);
    //         stat.observe(&2);
    //         stat.observe(&3);
    //         stat.observe(&3);
    //         stat.observe(&0);

    //         let mut counts = stat.counts.iter();
    //         assert_eq!(counts.next(), Some((&0_usize, &3_usize)));
    //         assert_eq!(counts.next(), Some((&2_usize, &1_usize)));
    //         assert_eq!(counts.next(), Some((&3_usize, &2_usize)));
    //         assert_eq!(counts.next(), None);

    //         stat.forget(&0);
    //         stat.forget(&2);
    //         stat.forget(&3);

    //         let mut counts = stat.counts.iter();
    //         // assert_eq!(counts.next(), Some((&0_usize, &2_usize)));
    //         // assert_eq!(counts.next(), Some((&3_usize, &1_usize)));
    //         assert_eq!(counts.next(), None);
    //         assert_eq!(stat.n, 3);
    //     }

    //     #[test]
    //     fn stat_observe_forget_all() {
    //         let mut stat = SbdSuffStat::new();

    //         stat.observe(&0);
    //         stat.observe(&0);
    //         stat.observe(&2);
    //         stat.observe(&3);
    //         stat.observe(&3);
    //         stat.observe(&0);

    //         let mut counts = stat.counts.iter();
    //         assert_eq!(counts.next(), Some((&0_usize, &3_usize)));
    //         assert_eq!(counts.next(), Some((&2_usize, &1_usize)));
    //         assert_eq!(counts.next(), Some((&3_usize, &2_usize)));
    //         assert_eq!(counts.next(), None);

    //         stat.forget(&0);
    //         stat.forget(&2);
    //         stat.forget(&3);
    //         stat.forget(&0);
    //         stat.forget(&0);
    //         stat.forget(&3);

    //         assert_eq!(stat.n, 0);

    //         let mut counts = stat.counts.iter();
    //         assert_eq!(counts.next(), None);
    //     }

    //     #[test]
    //     #[should_panic]
    //     fn stat_forget_oob_panics() {
    //         let mut stat = SbdSuffStat::new();

    //         stat.observe(&0);
    //         stat.observe(&2);
    //         stat.observe(&3);

    //         // the key `1` does not exist
    //         stat.forget(&1);
    //     }

    //     #[test]
    //     #[should_panic]
    //     fn stat_forget_from_empty_panics() {
    //         let mut stat = SbdSuffStat::new();
    //         stat.forget(&0);
    //     }
}
