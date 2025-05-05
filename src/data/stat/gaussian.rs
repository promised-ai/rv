#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::Gaussian;
use crate::traits::SuffStat;

/// Gaussian sufficient statistic.
///
/// Holds the number of observations, their sum, and the sum of their squared
/// values.
#[derive(Debug, Clone, PartialEq, Copy)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
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
    #[must_use]
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
    #[must_use]
    pub fn from_parts_unchecked(n: usize, mean: f64, sx: f64) -> Self {
        GaussianSuffStat { n, mean, sx }
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
        self.mean
    }

    /// Sum of `x`
    #[inline]
    #[must_use]
    pub fn sum_x(&self) -> f64 {
        self.mean * self.n as f64
    }

    /// Sum of `x^2`
    #[inline]
    #[must_use]
    pub fn sum_x_sq(&self) -> f64 {
        let nf = self.n as f64;
        (self.mean() * self.mean()).mul_add(nf, self.sx)
    }

    #[inline]
    #[must_use]
    pub fn sum_sq_diff(&self) -> f64 {
        self.sx
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

        impl<'a> From<&'a [$kind]> for DataOrSuffStat<'a, $kind, Gaussian> {
            fn from(xs: &'a [$kind]) -> Self {
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

        impl From<&[$kind]> for GaussianSuffStat {
            fn from(xs: &[$kind]) -> Self {
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

                if self.n == 0 {
                    *self = GaussianSuffStat {
                        n: 1,
                        mean: xf,
                        sx: 0.0,
                    };
                } else {
                    let n = self.n;
                    let mean = self.mean;
                    let sx = self.sx;

                    let n1 = n + 1;
                    let mean_xn =
                        (xf - mean).mul_add((n1 as f64).recip(), mean);

                    self.n = n + 1;
                    self.mean = mean_xn;
                    self.sx = (xf - mean).mul_add(xf - mean_xn, sx);
                }
            }

            fn forget(&mut self, x: &$kind) {
                let n = self.n;

                if n > 1 {
                    let xf = f64::from(*x);
                    let mean = self.mean;
                    let n_float = n as f64;
                    let nm1 = n_float - 1.0;
                    let nm1_recip = nm1.recip();

                    let old_mean =
                        (n_float * nm1_recip).mul_add(mean, -xf * nm1_recip);

                    let sx = (xf - old_mean).mul_add(-(xf - mean), self.sx);

                    *self = GaussianSuffStat {
                        n: n - 1,
                        mean: old_mean,
                        sx,
                    };
                } else {
                    *self = GaussianSuffStat {
                        n: 0,
                        mean: 0.0,
                        sx: 0.0,
                    };
                }
            }

            fn merge(&mut self, other: Self) {
                if other.n == 0 {
                    return;
                }
                let n1 = self.n as f64;
                let n2 = other.n as f64;
                let m1 = self.mean;
                let m2 = other.mean;
                let sum = n1 + n2;

                let mean = n1.mul_add(m1, n2 * m2) / sum;

                let d1 = m1 - mean;
                let d2 = m2 - mean;
                // let sx = self.sx + other.sx + n1 * d1 * d1 + n2 * d2 * d2;
                let sx = (n2 * d2)
                    .mul_add(d2, (n1 * d1).mul_add(d1, self.sx + other.sx));

                self.mean = mean;
                self.sx = sx;
                self.n += other.n;
            }
        }
    };
}

#[cfg(feature = "experimental")]
impl_gaussian_suffstat!(f16);
impl_gaussian_suffstat!(f32);
impl_gaussian_suffstat!(f64);

#[cfg(test)]
mod tests {
    use crate::traits::Sampleable;

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
        assert::close(suffstat.sum_x_sq(), 27.889_999_999_999_993, 1e-14);
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
        assert::close(suffstat.sum_x_sq(), 27.889_999_999_999_993, 1e-13);
    }

    #[test]
    fn incremental_merge() {
        let mut rng = rand::thread_rng();
        let g = crate::dist::Gaussian::standard();

        let xs: Vec<f64> = g.sample(5, &mut rng);
        let stat_a = {
            let mut stat = GaussianSuffStat::new();
            stat.observe_many(&xs);
            stat
        };

        let mut stat_b = {
            let mut stat = GaussianSuffStat::new();
            stat.observe(&xs[0]);
            stat
        };

        for x in xs.iter().skip(1) {
            let mut stat_temp = GaussianSuffStat::new();
            stat_temp.observe(x);
            <GaussianSuffStat as SuffStat<f64>>::merge(&mut stat_b, stat_temp);
        }

        assert_eq!(stat_a.n, stat_b.n);
        assert::close(stat_a.mean, stat_b.mean, 1e-10);
        assert::close(stat_a.sx, stat_b.sx, 1e-10);
    }
}
