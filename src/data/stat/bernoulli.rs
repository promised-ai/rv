#[cfg(feature = "serde1")]
use serde_derive::{Deserialize, Serialize};

use crate::data::Booleable;
use crate::data::DataOrSuffStat;
use crate::dist::Bernoulli;
use crate::traits::SuffStat;

/// Sufficient statistic for the Bernoulli distribution.
///
/// Contains the number of trials and the number of successes.
#[derive(Debug, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct BernoulliSuffStat {
    n: usize,
    k: usize,
}

impl BernoulliSuffStat {
    /// Create a new Bernoulli sufficient statistic
    pub fn new() -> Self {
        BernoulliSuffStat { n: 0, k: 0 }
    }

    /// Get the total number of trials, n.
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::BernoulliSuffStat;
    /// # use rv::traits::SuffStat;
    /// let mut stat = BernoulliSuffStat::new();
    ///
    /// stat.observe(&true);
    /// stat.observe(&false);
    ///
    /// assert_eq!(stat.n(), 2);
    /// ```
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the number of successful trials, k.
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::BernoulliSuffStat;
    /// # use rv::traits::SuffStat;
    /// let mut stat = BernoulliSuffStat::new();
    ///
    /// stat.observe(&true);
    /// stat.observe(&false);
    ///
    /// assert_eq!(stat.k(), 1);
    /// ```
    pub fn k(&self) -> usize {
        self.k
    }
}

impl Default for BernoulliSuffStat {
    fn default() -> Self {
        BernoulliSuffStat::new()
    }
}

impl SuffStat<bool> for BernoulliSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &bool) {
        self.n += 1;
        if *x {
            self.k += 1
        }
    }

    fn forget(&mut self, x: &bool) {
        self.n -= 1;
        if *x {
            self.k -= 1
        }
    }
}

impl<'a> Into<DataOrSuffStat<'a, bool, Bernoulli>> for &'a BernoulliSuffStat {
    fn into(self) -> DataOrSuffStat<'a, bool, Bernoulli> {
        DataOrSuffStat::SuffStat(self)
    }
}

impl<'a> Into<DataOrSuffStat<'a, bool, Bernoulli>> for &'a Vec<bool> {
    fn into(self) -> DataOrSuffStat<'a, bool, Bernoulli> {
        DataOrSuffStat::Data(self)
    }
}

impl<'a, X: Booleable> Into<DataOrSuffStat<'a, X, Bernoulli>>
    for &'a BernoulliSuffStat
{
    fn into(self) -> DataOrSuffStat<'a, X, Bernoulli> {
        DataOrSuffStat::SuffStat(self)
    }
}

impl<'a, X: Booleable> Into<DataOrSuffStat<'a, X, Bernoulli>> for &'a Vec<X> {
    fn into(self) -> DataOrSuffStat<'a, X, Bernoulli> {
        DataOrSuffStat::Data(self)
    }
}

impl<X: Booleable> SuffStat<X> for BernoulliSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &X) {
        self.n += 1;
        if x.into_bool() {
            self.k += 1
        }
    }

    fn forget(&mut self, x: &X) {
        self.n -= 1;
        if x.into_bool() {
            self.k -= 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_should_be_empty() {
        let stat = BernoulliSuffStat::new();
        assert_eq!(stat.n, 0);
        assert_eq!(stat.k, 0);
    }

    #[test]
    fn observe_1() {
        let mut stat = BernoulliSuffStat::new();
        stat.observe(&1_u8);
        assert_eq!(stat.n, 1);
        assert_eq!(stat.k, 1);
    }

    #[test]
    fn observe_true() {
        let mut stat = BernoulliSuffStat::new();
        stat.observe(&true);
        assert_eq!(stat.n, 1);
        assert_eq!(stat.k, 1);
    }

    #[test]
    fn observe_0() {
        let mut stat = BernoulliSuffStat::new();
        stat.observe(&0_i8);
        assert_eq!(stat.n, 1);
        assert_eq!(stat.k, 0);
    }

    #[test]
    fn observe_false() {
        let mut stat = BernoulliSuffStat::new();
        stat.observe(&false);
        assert_eq!(stat.n, 1);
        assert_eq!(stat.k, 0);
    }
}
