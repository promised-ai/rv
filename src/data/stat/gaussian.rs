#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::Gaussian;
use crate::traits::SuffStat;

/// Gaussian sufficient statistic.
///
/// Holds the number of observations, their sum, and the sum of their squared
/// values.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct GaussianSuffStat {
    /// Number of observations
    n: usize,
    /// Mean of `x`
    mean: f64,
    /// Intermediate quantity for computing sample and population variance
    sx: f64,
}

impl GaussianSuffStat {
    #[inline]
    pub fn new() -> Self {
        GaussianSuffStat {
            n: 0,
            mean: 0.0,
            sx: 0.0,
        }
    }

    /// Create a sufficient statistic from components without checking whether
    /// they are valid.
    #[inline]
    pub fn from_parts_unchecked(n: usize, mean: f64, sx: f64) -> Self {
        GaussianSuffStat { n, mean, sx }
    }

    /// Get the number of observations
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the sample mean
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Sum of `x`
    #[inline]
    pub fn sum_x(&self) -> f64 {
        self.mean * self.n as f64
    }

    /// Sum of `x^2`
    #[inline]
    pub fn sum_x_sq(&self) -> f64 {
        let nf = self.n as f64;
        self.mean().powi(2).mul_add(nf, self.sx)
    }
}

impl Default for GaussianSuffStat {
    fn default() -> Self {
        GaussianSuffStat::new()
    }
}

macro_rules! impl_gaussian_suffstat {
    ($kind:ty) => {
        impl<'a> From<&'a GaussianSuffStat>
            for DataOrSuffStat<'a, $kind, Gaussian>
        {
            fn from(stat: &'a GaussianSuffStat) -> Self {
                DataOrSuffStat::SuffStat(stat)
            }
        }

        impl<'a> From<&'a Vec<$kind>> for DataOrSuffStat<'a, $kind, Gaussian> {
            fn from(xs: &'a Vec<$kind>) -> Self {
                DataOrSuffStat::Data(xs)
            }
        }

        impl From<&Vec<$kind>> for GaussianSuffStat {
            fn from(xs: &Vec<$kind>) -> Self {
                let mut stat = GaussianSuffStat::new();
                stat.observe_many(xs);
                stat
            }
        }

        impl SuffStat<$kind> for GaussianSuffStat {
            fn n(&self) -> usize {
                self.n
            }

            fn observe(&mut self, x: &$kind) {
                let xf = f64::from(*x);

                self.n += 1;

                let mean_xn = (xf - self.mean)
                    .mul_add((self.n as f64).recip(), self.mean);
                self.sx = (xf - self.mean).mul_add(xf - mean_xn, self.sx);
                self.mean = mean_xn;
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = f64::from(*x);

                    let n = self.n as f64;
                    let nm1 = (self.n - 1) as f64;

                    let old_mean = (n / nm1).mul_add(self.mean, -xf / nm1);

                    self.sx -= (xf - old_mean) * (xf - self.mean);
                    self.mean = old_mean;
                    self.n -= 1;
                } else {
                    self.n = 0;
                    self.mean = 0.0;
                    self.sx = 0.0;
                }
            }
        }
    };
}

impl_gaussian_suffstat!(f32);
impl_gaussian_suffstat!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_parts_unchecked() {
        let stat = GaussianSuffStat::from_parts_unchecked(10, 0.5, 1.2);
        assert_eq!(stat.n(), 10);
        assert_eq!(stat.mean(), 0.5);
        assert_eq!(stat.sx, 1.2);
    }

    #[test]
    fn suffstat_increments_correctly() {
        let xs: Vec<f64> = vec![0.0, 1.2, 2.3, 4.6];
        let mut suffstat = GaussianSuffStat::new();

        for x in xs {
            suffstat.observe(&x);
        }

        assert_eq!(suffstat.n(), 4);
        assert::close(suffstat.mean(), 2.025, 1e-14);
        assert::close(suffstat.sum_x(), 8.1, 1e-14);
        assert::close(suffstat.sum_x_sq(), 27.889999999999993, 1e-14);
    }

    #[test]
    fn suffstat_decrements_correctly() {
        let xs: Vec<f64> = vec![0.0, 1.2, 2.3, 4.6];
        let mut suffstat = GaussianSuffStat::new();

        for x in xs {
            suffstat.observe(&x);
        }

        suffstat.observe(&5.0);
        suffstat.forget(&5.0);

        assert_eq!(suffstat.n(), 4);
        assert::close(suffstat.mean(), 2.025, 1e-14);
        assert::close(suffstat.sum_x(), 8.1, 1e-14);
        assert::close(suffstat.sum_x_sq(), 27.889999999999993, 1e-13);
    }
}
