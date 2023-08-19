use crate::traits::SuffStat;
use std::collections::BTreeMap;

pub struct SbdSuffStat {
    n: usize,
    counts: BTreeMap<usize, usize>,
}

impl SbdSuffStat {
    pub fn new() -> Self {
        Self {
            n: 0,
            counts: BTreeMap::new(),
        }
    }

    pub fn counts(&self) -> &BTreeMap<usize, usize> {
        &self.counts
    }
}

impl SuffStat<usize> for SbdSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &usize) {
        self.n += 1;
        *self.counts.entry(*x).or_default() += 1;
    }

    fn forget(&mut self, x: &usize) {
        self.n += 1;
        if self.n == 0 {
            self.counts = BTreeMap::new();
        } else {
            self.counts.entry(*x).and_modify(|ct| *ct -= 1);
            if self.counts[x] == 0 {
                self.counts.remove(x).unwrap();
            }
        }
    }
}
