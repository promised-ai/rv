//! Skellam distribution on signed integers
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::dist::Poisson;
use crate::impl_display;
use crate::misc::bessel::bessel_iv;
use crate::traits::*;
use lru::LruCache;
use rand::Rng;
use std::{cell::RefCell, num::NonZeroUsize};

/// [Skellam distribution](https://en.wikipedia.org/wiki/Skellam_distribution)
/// over x in {.., -2, -1, 0, 1, ... }.
///
/// # Example
///
/// ```
/// use rv::prelude::*;
///
/// // Create Skellam(μ_1=5.3, μ_2=2.5)
/// let skel = Skellam::new(5.3, 2.5).unwrap();
///
/// // Draw 100 samples
/// let mut rng = rand::thread_rng();
/// let xs: Vec<i32> = skel.sample(100, &mut rng);
/// assert_eq!(xs.len(), 100)
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Skellam {
    /// Mean of first poisson
    mu_1: f64,
    /// Mean of second poisson
    mu_2: f64,
    /// Cached values of bessel_iv. Note that the cache is not invalidated when
    /// the values of mu_1 or mu_2 change.
    #[cfg_attr(feature = "serde1", serde(skip, default = "cache_default"))]
    bessel_iv_cache: RefCell<LruCache<i32, f64>>,
}

fn cache_default() -> RefCell<LruCache<i32, f64>> {
    // SAFETY: 100 is a valid usize.
    RefCell::new(LruCache::new(unsafe { NonZeroUsize::new_unchecked(100) }))
}

pub struct SkellamParameters {
    pub mu_1: f64,
    pub mu_2: f64,
}

impl Parameterized for Skellam {
    type Parameters = SkellamParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            mu_1: self.mu_1(),
            mu_2: self.mu_2(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.mu_1, params.mu_2)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum SkellamError {
    /// The first rate parameter is less than or equal to zero
    Mu1TooLow { mu_1: f64 },
    /// The first rate parameter is infinite or NaN
    Mu1NotFinite { mu_1: f64 },
    /// The second rate parameter is less than or equal to zero
    Mu2TooLow { mu_2: f64 },
    /// The second rate parameter is infinite or NaN
    Mu2NotFinite { mu_2: f64 },
}

impl Skellam {
    /// Create a new Skellam distribution with given rates
    pub fn new(mu_1: f64, mu_2: f64) -> Result<Self, SkellamError> {
        if mu_1 <= 0.0 {
            Err(SkellamError::Mu1TooLow { mu_1 })
        } else if mu_2 <= 0.0 {
            Err(SkellamError::Mu2TooLow { mu_2 })
        } else if !mu_1.is_finite() {
            Err(SkellamError::Mu1NotFinite { mu_1 })
        } else if !mu_2.is_finite() {
            Err(SkellamError::Mu2NotFinite { mu_2 })
        } else {
            Ok(Self::new_unchecked(mu_1, mu_2))
        }
    }

    /// Creates a new Skellam without checking whether the parameters are valid.
    #[inline]
    pub fn new_unchecked(mu_1: f64, mu_2: f64) -> Self {
        Skellam {
            mu_1,
            mu_2,
            bessel_iv_cache: cache_default(),
        }
    }

    /// Get the mu_1 parameter
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::Skellam;
    /// let skel = Skellam::new(2.0, 3.0).unwrap();
    /// assert_eq!(skel.mu_1(), 2.0);
    /// ```
    #[inline]
    pub fn mu_1(&self) -> f64 {
        self.mu_1
    }

    /// Set the mu_1 (first rate) parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::Skellam;
    /// let mut skel = Skellam::new(2.0, 1.0).unwrap();
    /// assert_eq!(skel.mu_1(), 2.0);
    ///
    /// skel.set_mu_1(1.1).unwrap();
    /// assert_eq!(skel.mu_1(), 1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Skellam;
    /// # let mut skel = Skellam::new(2.0, 1.0).unwrap();
    /// assert!(skel.set_mu_1(1.1).is_ok());
    /// assert!(skel.set_mu_1(0.0).is_err());
    /// assert!(skel.set_mu_1(-1.0).is_err());
    /// assert!(skel.set_mu_1(f64::INFINITY).is_err());
    /// assert!(skel.set_mu_1(f64::NEG_INFINITY).is_err());
    /// assert!(skel.set_mu_1(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_mu_1(&mut self, mu_1: f64) -> Result<(), SkellamError> {
        if mu_1 <= 0.0 {
            Err(SkellamError::Mu1TooLow { mu_1 })
        } else if !mu_1.is_finite() {
            Err(SkellamError::Mu1NotFinite { mu_1 })
        } else {
            self.set_mu_1_unchecked(mu_1);
            Ok(())
        }
    }

    /// Set the mu_1 (first rate) parameter without input validation
    #[inline]
    pub fn set_mu_1_unchecked(&mut self, mu_1: f64) {
        self.mu_1 = mu_1;
    }

    /// Get the mu_2 parameter
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::Skellam;
    /// let skel = Skellam::new(2.0, 3.0).unwrap();
    /// assert_eq!(skel.mu_2(), 3.0);
    /// ```
    #[inline]
    pub fn mu_2(&self) -> f64 {
        self.mu_2
    }

    /// Set the mu_2 (second rate) parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::Skellam;
    /// let mut skel = Skellam::new(2.0, 1.0).unwrap();
    /// assert_eq!(skel.mu_2(), 1.0);
    ///
    /// skel.set_mu_2(1.1).unwrap();
    /// assert_eq!(skel.mu_2(), 1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Skellam;
    /// # let mut skel = Skellam::new(2.0, 1.0).unwrap();
    /// assert!(skel.set_mu_2(1.1).is_ok());
    /// assert!(skel.set_mu_2(0.0).is_err());
    /// assert!(skel.set_mu_2(-1.0).is_err());
    /// assert!(skel.set_mu_2(f64::INFINITY).is_err());
    /// assert!(skel.set_mu_2(f64::NEG_INFINITY).is_err());
    /// assert!(skel.set_mu_2(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_mu_2(&mut self, mu_2: f64) -> Result<(), SkellamError> {
        if mu_2 <= 0.0 {
            Err(SkellamError::Mu2TooLow { mu_2 })
        } else if !mu_2.is_finite() {
            Err(SkellamError::Mu2NotFinite { mu_2 })
        } else {
            self.set_mu_2_unchecked(mu_2);
            Ok(())
        }
    }

    /// Set the mu_2 (first rate) parameter without input validation
    #[inline]
    pub fn set_mu_2_unchecked(&mut self, mu_2: f64) {
        self.mu_2 = mu_2;
    }

    /// Set the cache size on the internal LRU for Bessel Iv calls.
    ///
    /// # Panics
    ///
    /// Panics if `cap` is 0.
    #[inline]
    pub fn set_cache_cap(&self, cap: usize) {
        self.bessel_iv_cache
            .borrow_mut()
            .resize(NonZeroUsize::new(cap).unwrap());
    }
}

impl From<&Skellam> for String {
    fn from(skellam: &Skellam) -> String {
        format!("Skellam(μ_1: {}, μ_2: {})", skellam.mu_1, skellam.mu_2)
    }
}

impl_display!(Skellam);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for Skellam {
            fn ln_f(&self, x: &$kind) -> f64 {
                let kf = f64::from(*x);
                let mut cache = self.bessel_iv_cache.borrow_mut();
                let bf: f64 =
                    cache.get(&(*x as i32)).map(|b| *b).unwrap_or_else(|| {
                        let b =
                            bessel_iv(kf, 2.0 * (self.mu_1 * self.mu_2).sqrt())
                                .unwrap()
                                .ln();
                        cache.put((*x as i32), b);
                        b
                    });

                -(self.mu_1 + self.mu_2)
                    + (kf / 2.0).mul_add((self.mu_1 / self.mu_2).ln(), bf)
            }
        }

        impl Sampleable<$kind> for Skellam {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let pois_1 = Poisson::new_unchecked(self.mu_1);
                let pois_2 = Poisson::new_unchecked(self.mu_2);
                let x_1: u32 = pois_1.draw(rng);
                let x_2: u32 = pois_2.draw(rng);
                (x_1 as i32 - x_2 as i32) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let pois_1 = Poisson::new_unchecked(self.mu_1);
                let pois_2 = Poisson::new_unchecked(self.mu_2);
                pois_1
                    .sample(n, rng)
                    .into_iter()
                    .zip(pois_2.sample(n, rng).into_iter())
                    .map(|(x_1, x_2): (u32, u32)| {
                        (x_1 as $kind) - (x_2 as $kind)
                    })
                    .collect()
            }
        }

        impl Support<$kind> for Skellam {
            #[allow(unused_comparisons)]
            fn supports(&self, _x: &$kind) -> bool {
                true
            }
        }

        impl DiscreteDistr<$kind> for Skellam {}
    };
}

impl Mean<f64> for Skellam {
    fn mean(&self) -> Option<f64> {
        Some(self.mu_1 - self.mu_2)
    }
}

impl Variance<f64> for Skellam {
    fn variance(&self) -> Option<f64> {
        Some(self.mu_1 + self.mu_2)
    }
}

impl Skewness for Skellam {
    fn skewness(&self) -> Option<f64> {
        Some((self.mu_1 - self.mu_2) / (self.mu_1 + self.mu_2).powi(3).sqrt())
    }
}

impl Kurtosis for Skellam {
    fn kurtosis(&self) -> Option<f64> {
        Some(3.0 + (self.mu_1 + self.mu_2).recip())
    }
}

impl_traits!(i8);
impl_traits!(i16);
impl_traits!(i32);

impl PartialEq for Skellam {
    fn eq(&self, other: &Skellam) -> bool {
        self.mu_1 == other.mu_1 && self.mu_2 == other.mu_2
    }
}

impl Clone for Skellam {
    fn clone(&self) -> Self {
        let old_cache = self.bessel_iv_cache.borrow();
        let mut cache = LruCache::new(old_cache.cap());
        for (key, value) in old_cache.iter() {
            cache.put(*key, *value);
        }
        Skellam {
            mu_1: self.mu_1,
            mu_2: self.mu_2,
            bessel_iv_cache: RefCell::new(cache),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::x2_test;
    use crate::test_basic_impls;
    use std::f64;

    const TOL: f64 = 1E-12;
    const N_TRIES: usize = 5;
    const X2_PVAL: f64 = 0.2;

    test_basic_impls!(i32, Skellam, Skellam::new(1.0, 2.0).unwrap());

    #[test]
    fn new() {
        let skel = Skellam::new(0.001, 3.456).unwrap();
        assert::close(skel.mu_1, 0.001, TOL);
        assert::close(skel.mu_2, 3.456, TOL);

        let skel = Skellam::new(1.234, 5.432).unwrap();
        assert::close(skel.mu_1, 1.234, TOL);
        assert::close(skel.mu_2, 5.432, TOL);
    }

    #[test]
    fn new_should_reject_non_finite_rate() {
        assert!(Skellam::new(1.0, f64::INFINITY).is_err());
        assert!(Skellam::new(1.0, f64::NAN).is_err());
        assert!(Skellam::new(f64::INFINITY, 1.0).is_err());
        assert!(Skellam::new(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn new_should_reject_rate_lteq_zero() {
        assert!(Skellam::new(0.0, 0.0).is_err());
        assert!(Skellam::new(1.0, -1E-12).is_err());
        assert!(Skellam::new(-1E-12, 1.0).is_err());
        assert!(Skellam::new(1.0, f64::NEG_INFINITY).is_err());
        assert!(Skellam::new(f64::NEG_INFINITY, 1.0).is_err());
    }

    #[test]
    fn ln_pdf() {
        let skel = Skellam::new(5.3, 6.5).unwrap();
        assert::close(skel.ln_pmf(&1_i32), -2.347_033_152_058_002, TOL);
        assert::close(skel.ln_pmf(&5_i32), -3.805_689_157_233_512_5, TOL);
        assert::close(skel.ln_pmf(&11_i32), -8.339_462_666_191_974, TOL);

        assert::close(skel.ln_pmf(&0_i32), -2.200_416_098_697_956, TOL);

        assert::close(skel.ln_pmf(&-1_i32), -2.142_937_795_714_486_6, TOL);
        assert::close(skel.ln_pmf(&-5_i32), -2.785_212_375_515_935_7, TOL);
        assert::close(skel.ln_pmf(&-11_i32), -6.094_413_746_413_306, TOL);
    }

    #[test]
    fn mean() {
        let m1 = Skellam::new(1.5, 2.3).unwrap().mean().unwrap();
        assert::close(m1, -0.8, TOL);

        let m2 = Skellam::new(33.2, 10.5).unwrap().mean().unwrap();
        assert::close(m2, 22.7, TOL);
    }

    #[test]
    fn variance() {
        let v1 = Skellam::new(1.5, 2.3).unwrap().variance().unwrap();
        assert::close(v1, 3.8, TOL);

        let v2 = Skellam::new(33.2, 10.5).unwrap().variance().unwrap();
        assert::close(v2, 43.7, TOL);
    }

    #[test]
    fn skewness() {
        let s = Skellam::new(5.3, 4.5).unwrap().skewness().unwrap();
        assert::close(s, 0.026_076_594_489_793_457, TOL);
    }

    #[test]
    fn kurtosis() {
        let k = Skellam::new(5.3, 4.5).unwrap().kurtosis().unwrap();
        assert::close(k, 3.102_040_816_326_530_5, TOL);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::rng();
        let pois = Skellam::new(3.0, 3.0).unwrap();

        // How many bins do we need?
        let right_len: usize = (0..100)
            .position(|x| pois.pmf(&x) < f64::EPSILON)
            .unwrap_or(99)
            + 1;

        let left_len: usize = (0..100)
            .position(|x| pois.pmf(&(-x)) < f64::EPSILON)
            .unwrap_or(99)
            + 1;

        let total_bins = left_len + right_len;
        let ps: Vec<f64> = (-(left_len as i32)..(right_len as i32))
            .map(|x| pois.pmf(&x))
            .collect();

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; total_bins];
            let xs: Vec<i32> = pois
                .sample(1000, &mut rng)
                .into_iter()
                .map(|x: i32| x.min(right_len as i32).max(-(left_len as i32)))
                .collect();

            xs.iter().for_each(|&x| {
                f_obs[(x + left_len as i32) as usize] += 1;
            });
            let (_, p) = x2_test(&f_obs, &ps);
            if p > X2_PVAL {
                acc + 1
            } else {
                acc
            }
        });
        assert!(passes > 0);
    }
}
