use crate::suffstat_traits::SuffStat;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct StickBreakingSuffStat {
    n: usize,
    num_breaks: usize,
    sum_log_q: f64,
}

impl Default for StickBreakingSuffStat {
    fn default() -> Self {
        Self::new()
    }
}

impl StickBreakingSuffStat {
    pub fn new() -> Self {
        Self {
            n: 0,
            num_breaks: 0,
            sum_log_q: 0.0,
        }
    }
}

// A standard stick-breaking process requires a Beta(1, α) distribution. We
// instead parameterize by a UnitPowerLaw(α), which is equivalent to a
// Beta(α,1). Given a sequence of stick lengths coming form such a process, we
// can recover the sufficient statistic for this UnitPowerLaw distribution
fn stick_stat(sticks: &[f64]) -> (usize, f64) {
    // First we need to find the sequence of remaining stick lengths. Because we
    // broke the sticks left-to-right, we need to reverse the sequence.
    let remaining = sticks.iter().rev().scan(0.0, |acc, &x| {
        *acc += x;
        Some(*acc)
    });

    let qs = sticks
        .iter()
        // Reversing `remaining` would force us to collect the intermediate
        // result e.g. into a `Vec`. Instead, we can reverse the sequence of
        // stick lengths to match.
        .rev()
        // Now zip the sequences together and do the main computation we're interested in.
        .zip(remaining)
        // In theory the broken stick lengths should all be less than what was
        // remaining before the break. In practice, numerical instabilities can
        // cause problems. So we filter to be sure we only consider valid
        // values.
        .filter(|(&len, remaining)| len < *remaining)
        .map(|(&len, remaining)| 1.0 - len / remaining);

    // The sufficient statistic is (n, ∑ᵢ log(1 - pᵢ)) == (n, log ∏ᵢ(1 - pᵢ)).
    // First we compute `n` and `∏ᵢ(1 - pᵢ)`
    let (num_breaks, prod_q) =
        qs.fold((0, 1.0), |(n, prod_q), q| (n + 1, prod_q * q));

    (num_breaks, prod_q.ln())
}

impl SuffStat<&[f64]> for StickBreakingSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, sticks: &&[f64]) {
        let (num_breaks, sum_log_q) = stick_stat(sticks);
        self.n += 1;
        self.num_breaks += num_breaks;
        self.sum_log_q += sum_log_q;
    }

    fn forget(&mut self, sticks: &&[f64]) {
        let (num_breaks, sum_log_q) = stick_stat(sticks);
        self.n -= 1;
        self.num_breaks -= num_breaks;
        self.sum_log_q -= sum_log_q;
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;

//     #[test]
//     fn stat_observe_forget_some() {
//         let mut stat = StickBreakingSuffStat::new();

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
//         assert_eq!(counts.next(), Some((&0_usize, &2_usize)));
//         assert_eq!(counts.next(), Some((&3_usize, &1_usize)));
//         assert_eq!(counts.next(), None);
//         assert_eq!(stat.n, 3);
//     }

//     #[test]
//     fn stat_observe_forget_all() {
//         let mut stat = StickBreakingSuffStat::new();

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
//         let mut stat = StickBreakingSuffStat::new();

//         stat.observe(&0);
//         stat.observe(&2);
//         stat.observe(&3);

//         // the key `1` does not exist
//         stat.forget(&1);
//     }

//     #[test]
//     #[should_panic]
//     fn stat_forget_from_empty_panics() {
//         let mut stat = StickBreakingSuffStat::new();
//         stat.forget(&0);
//     }
// }
