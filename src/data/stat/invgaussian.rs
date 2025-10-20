#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::InvGaussian;
use crate::traits::SuffStat;

/// Gaussian sufficient statistic.
///
/// Holds the number of observations, their sum, and the sum of their squared
/// values.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct InvGaussianSuffStat {
    /// Number of observations
    n: usize,
    /// sum of `x`
    sum_x: f64,
    /// sum of `1/x`
    sum_inv_x: f64,
    /// sum of ln(x)
    sum_ln_x: f64,
}

impl InvGaussianSuffStat {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        InvGaussianSuffStat {
            n: 0,
            sum_x: 0.0,
            sum_inv_x: 0.0,
            sum_ln_x: 0.0,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    ///
    /// # Example
    /// ```rust
    /// use rv::data::InvGaussianSuffStat;
    /// use rv::prelude::SuffStat;
    ///
    /// let data: Vec<f64> = vec![0.1, 0.2, 0.3];
    ///
    /// let mut stat_b = InvGaussianSuffStat::new();
    /// stat_b.observe_many(&data);
    ///
    /// let n = data.len();
    /// let sum_x = data.iter().sum();
    /// let sum_inv_x = data.iter().map(|x: &f64| x.recip()).sum();
    /// let sum_ln_x = data.iter().map(|x: &f64| x.ln()).sum();
    ///
    /// let stat_a = InvGaussianSuffStat::from_parts_unchecked(n, sum_x, sum_inv_x, sum_ln_x);
    ///
    /// assert_eq!(stat_a.n(), stat_b.n());
    /// assert::close(stat_a.sum_x(), stat_b.sum_x(), 1e-10);
    /// assert::close(stat_a.sum_inv_x(), stat_b.sum_inv_x(), 1e-10);
    /// assert::close(stat_a.sum_ln_x(), stat_b.sum_ln_x(), 1e-10);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_parts_unchecked(
        n: usize,
        sum_x: f64,
        sum_inv_x: f64,
        sum_ln_x: f64,
    ) -> Self {
        InvGaussianSuffStat {
            n,
            sum_x,
            sum_inv_x,
            sum_ln_x,
        }
    }

    /// Get the number of observations
    #[inline]
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sample mean
    #[inline]
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.sum_x / self.n as f64
    }

    /// Sum of `x`
    #[inline]
    #[must_use]
    pub fn sum_x(&self) -> f64 {
        self.sum_x
    }

    /// Sum of `1/x`
    #[inline]
    #[must_use]
    pub fn sum_inv_x(&self) -> f64 {
        self.sum_inv_x
    }

    #[inline]
    #[must_use]
    pub fn sum_ln_x(&self) -> f64 {
        self.sum_ln_x
    }
}

impl Default for InvGaussianSuffStat {
    fn default() -> Self {
        InvGaussianSuffStat::new()
    }
}

macro_rules! impl_invgaussian_suffstat {
    ($kind:ty) => {
        impl<'a> From<&'a InvGaussianSuffStat>
            for DataOrSuffStat<'a, $kind, InvGaussian>
        {
            fn from(stat: &'a InvGaussianSuffStat) -> Self {
                DataOrSuffStat::SuffStat(stat)
            }
        }

        impl<'a> From<&'a Vec<$kind>>
            for DataOrSuffStat<'a, $kind, InvGaussian>
        {
            fn from(xs: &'a Vec<$kind>) -> Self {
                DataOrSuffStat::Data(xs.as_slice())
            }
        }

        impl<'a> From<&'a [$kind]> for DataOrSuffStat<'a, $kind, InvGaussian> {
            fn from(xs: &'a [$kind]) -> Self {
                DataOrSuffStat::Data(xs)
            }
        }

        impl From<&Vec<$kind>> for InvGaussianSuffStat {
            fn from(xs: &Vec<$kind>) -> Self {
                let mut stat = InvGaussianSuffStat::new();
                stat.observe_many(xs);
                stat
            }
        }

        impl From<&[$kind]> for InvGaussianSuffStat {
            fn from(xs: &[$kind]) -> Self {
                let mut stat = InvGaussianSuffStat::new();
                stat.observe_many(xs);
                stat
            }
        }

        impl SuffStat<$kind> for InvGaussianSuffStat {
            fn n(&self) -> usize {
                self.n
            }

            fn observe(&mut self, x: &$kind) {
                let xf = f64::from(*x);

                self.n += 1;

                self.sum_x += xf;
                self.sum_inv_x += xf.recip();
                self.sum_ln_x += xf.ln();
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = f64::from(*x);

                    self.sum_x -= xf;
                    self.sum_inv_x -= xf.recip();
                    self.sum_ln_x -= xf.ln();
                    self.n -= 1;
                } else {
                    self.n = 0;
                    self.sum_x = 0.0;
                    self.sum_inv_x = 0.0;
                    self.sum_ln_x = 0.0;
                }
            }
            fn merge(&mut self, other: Self) {
                self.n += other.n;
                self.sum_x += other.sum_x;
                self.sum_inv_x += other.sum_inv_x;
                self.sum_ln_x += other.sum_ln_x;
            }
        }
    };
}

impl_invgaussian_suffstat!(f32);
impl_invgaussian_suffstat!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_forget() {
        let mut stat = InvGaussianSuffStat::new();

        stat.observe(&0.1);
        stat.observe(&0.2);

        assert_eq!(stat.n(), 2);
        assert::close(stat.sum_x, 0.1_f64 + 0.2_f64, 1e-10);
        assert::close(
            stat.sum_inv_x,
            (0.1_f64).recip() + (0.2_f64).recip(),
            1e-10,
        );
        assert::close(stat.sum_ln_x, (0.1_f64).ln() + (0.2_f64).ln(), 1e-10);

        stat.forget(&0.1);

        assert_eq!(stat.n(), 1);
        assert::close(stat.sum_x, 0.2_f64, 1e-10);
        assert::close(stat.sum_inv_x, (0.2_f64).recip(), 1e-10);
        assert::close(stat.sum_ln_x, (0.2_f64).ln(), 1e-10);

        stat.forget(&0.2);

        assert_eq!(stat.n(), 0);
        assert_eq!(stat.sum_ln_x, 0.0);
    }

    #[test]
    fn merge() {
        let mut a = InvGaussianSuffStat::new();
        let mut b = InvGaussianSuffStat::new();
        let mut c = InvGaussianSuffStat::new();

        a.observe_many(&[0.1_f64, 0.2, 0.3]);
        b.observe_many(&[0.9_f64, 0.8, 0.7]);

        c.observe_many(&[0.1_f64, 0.2, 0.3, 0.9, 0.8, 0.7]);

        <InvGaussianSuffStat as SuffStat<f64>>::merge(&mut a, b);

        assert_eq!(a.n(), c.n());
        assert::close(a.sum_x(), c.sum_x(), 1e-10);
        assert::close(a.sum_inv_x(), c.sum_inv_x(), 1e-10);
        assert::close(a.sum_ln_x(), c.sum_ln_x(), 1e-10);
    }
}
