#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

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
    #[inline]
    pub fn new() -> Self {
        BernoulliSuffStat { n: 0, k: 0 }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    pub fn from_parts_unchecked(n: usize, k: usize) -> Self {
        BernoulliSuffStat { n, k }
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
    #[inline]
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
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }
}

impl Default for BernoulliSuffStat {
    fn default() -> Self {
        BernoulliSuffStat::new()
    }
}

impl<'a, X> From<&'a BernoulliSuffStat> for DataOrSuffStat<'a, X, Bernoulli>
where
    X: Booleable,
{
    fn from(stat: &'a BernoulliSuffStat) -> Self {
        DataOrSuffStat::SuffStat(stat)
    }
}

impl<'a, X> From<&'a Vec<X>> for DataOrSuffStat<'a, X, Bernoulli>
where
    X: Booleable,
{
    fn from(xs: &'a Vec<X>) -> Self {
        DataOrSuffStat::Data(xs.as_slice())
    }
}

impl<'a, X> From<&'a [X]> for DataOrSuffStat<'a, X, Bernoulli>
where
    X: Booleable,
{
    fn from(xs: &'a [X]) -> Self {
        DataOrSuffStat::Data(xs)
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
    fn from_parts_unchecked() {
        let stat = BernoulliSuffStat::from_parts_unchecked(10, 3);
        assert_eq!(stat.n(), 10);
        assert_eq!(stat.k(), 3);
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
