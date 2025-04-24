#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::CategoricalDatum;
use crate::data::DataOrSuffStat;
use crate::dist::Categorical;
use crate::traits::SuffStat;

/// Categorical distribution sufficient statistic.
///
/// Store the number of observations and the count of observations of each
/// instance.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct CategoricalSuffStat {
    n: usize,
    counts: Vec<f64>,
}

impl CategoricalSuffStat {
    #[inline]
    #[must_use] pub fn new(k: usize) -> Self {
        CategoricalSuffStat {
            n: 0,
            counts: vec![0.0; k],
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    #[must_use] pub fn from_parts_unchecked(n: usize, counts: Vec<f64>) -> Self {
        CategoricalSuffStat { n, counts }
    }

    /// Get the total number of trials
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::CategoricalSuffStat;
    /// # use rv::traits::SuffStat;
    /// let mut stat = CategoricalSuffStat::new(2);
    ///
    /// stat.observe(&0_u8);
    /// stat.observe(&1_u8);
    /// stat.observe(&1_u8);
    ///
    /// assert_eq!(stat.n(), 3);
    /// ```
    #[inline]
    #[must_use] pub fn n(&self) -> usize {
        self.n
    }

    /// Get the number of occurrences of each class, counts
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::CategoricalSuffStat;
    /// # use rv::traits::SuffStat;
    /// let mut stat = CategoricalSuffStat::new(3);
    ///
    /// stat.observe(&0_u8);
    /// stat.observe(&2_u8);
    /// stat.observe(&2_u8);
    ///
    /// assert_eq!(*stat.counts(), vec![1.0, 0.0, 2.0]);
    /// ```
    #[inline]
    #[must_use] pub fn counts(&self) -> &Vec<f64> {
        &self.counts
    }
}

impl<'a, X> From<&'a CategoricalSuffStat> for DataOrSuffStat<'a, X, Categorical>
where
    X: CategoricalDatum,
{
    fn from(stat: &'a CategoricalSuffStat) -> Self {
        DataOrSuffStat::SuffStat(stat)
    }
}

impl<'a, X> From<&'a Vec<X>> for DataOrSuffStat<'a, X, Categorical>
where
    X: CategoricalDatum,
{
    fn from(xs: &'a Vec<X>) -> Self {
        DataOrSuffStat::Data(xs.as_slice())
    }
}

impl<'a, X> From<&'a [X]> for DataOrSuffStat<'a, X, Categorical>
where
    X: CategoricalDatum,
{
    fn from(xs: &'a [X]) -> Self {
        DataOrSuffStat::Data(xs)
    }
}

impl<X: CategoricalDatum> SuffStat<X> for CategoricalSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &X) {
        let ix = x.into_usize();
        self.n += 1;
        self.counts[ix] += 1.0;
    }

    fn forget(&mut self, x: &X) {
        let ix = x.into_usize();
        self.n -= 1;
        self.counts[ix] -= 1.0;
    }

    fn merge(&mut self, other: Self) {
        self.n += other.n;
        self.counts
            .iter_mut()
            .zip(other.counts.iter().copied())
            .for_each(|(ct, ct_o)| {
                *ct += ct_o;
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let sf = CategoricalSuffStat::new(4);
        assert_eq!(sf.counts.len(), 4);
        assert_eq!(sf.n, 0);
        assert!(sf.counts.iter().all(|&ct| ct.abs() < 1E-12));
    }

    #[test]
    fn from_parts_unchecked() {
        let stat = CategoricalSuffStat::from_parts_unchecked(
            10,
            vec![1.0, 2.0, 3.0, 4.0],
        );

        assert_eq!(stat.n(), 10);
        assert_eq!(stat.counts()[0], 1.0);
        assert_eq!(stat.counts()[1], 2.0);
        assert_eq!(stat.counts()[2], 3.0);
        assert_eq!(stat.counts()[3], 4.0);
    }
}
