#[cfg(feature = "serde1")]
use serde_derive::{Deserialize, Serialize};

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
        self.sx + self.mean() * self.mean() * nf
    }
}

impl Default for GaussianSuffStat {
    fn default() -> Self {
        GaussianSuffStat::new()
    }
}

macro_rules! impl_gaussian_suffstat {
    ($kind:ty) => {
        impl<'a> Into<DataOrSuffStat<'a, $kind, Gaussian>>
            for &'a GaussianSuffStat
        {
            fn into(self) -> DataOrSuffStat<'a, $kind, Gaussian> {
                DataOrSuffStat::SuffStat(self)
            }
        }

        impl<'a> Into<DataOrSuffStat<'a, $kind, Gaussian>> for &'a Vec<$kind> {
            fn into(self) -> DataOrSuffStat<'a, $kind, Gaussian> {
                DataOrSuffStat::Data(self)
            }
        }

        // TODO: store in a more numerically stable form
        impl SuffStat<$kind> for GaussianSuffStat {
            fn n(&self) -> usize {
                self.n
            }

            fn observe(&mut self, x: &$kind) {
                let xf = f64::from(*x);

                self.n += 1;

                let mean_xn = self.mean + (xf - self.mean) / (self.n as f64);

                self.sx += (xf - self.mean) * (xf - mean_xn);
                self.mean = mean_xn;
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = f64::from(*x);

                    let n = self.n as f64;
                    let nm1 = (self.n - 1) as f64;

                    let old_mean = n / nm1 * self.mean - xf / nm1;

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
    fn suffstat_increments_correctly() {
        let xs: Vec<f64> = vec![0.0, 1.2, 2.3, 4.6];
        let mut suffstat = GaussianSuffStat::new();

        for x in xs {
            suffstat.observe(&x);
        }

        assert_eq!(suffstat.n(), 4);
        assert::close(suffstat.mean(), 2.0249999999999999, 1e-14);
        assert::close(suffstat.sum_x(), 8.0999999999999996, 1e-14);
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
        assert::close(suffstat.mean(), 2.0249999999999999, 1e-14);
        assert::close(suffstat.sum_x(), 8.0999999999999996, 1e-14);
        assert::close(suffstat.sum_x_sq(), 27.889999999999993, 1e-13);
    }
}
