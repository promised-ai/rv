use crate::dist::Poisson;
use crate::misc::ln_binom;
use crate::traits::{
    Cdf, DiscreteDistr, HasDensity, Kurtosis, Mean, Parameterized, Sampleable,
    Skewness, Support, Variance,
};
use rand::Rng;
use std::fmt;
use std::sync::OnceLock;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Negative Binomial distribution errors
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum NegBinomialError {
    /// The probability parameter, p, is not in [0, 1]
    POutOfRange { p: f64 },
    /// the p parameter is infinite or NaN
    PNotFinite { p: f64 },
    /// R is less that 1.0
    RLessThanOne { r: f64 },
    /// the r parameter is infinite or NaN
    RNotFinite { r: f64 },
}

/// Negative Binomial distribution NBin(r, p).
///
/// # Notes
/// This crate uses [the parameterization found on Wolfram
/// Mathworld](http://mathworld.wolfram.com/NegativeBinomialDistribution.html),
/// which is also the parameterization used in scipy.
///
/// # Parameters
/// - r: The number of successes before the trials are stopped
/// - p: The success probability
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct NegBinomial {
    r: f64,
    p: f64,
    // ln(1-p)
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_1mp: OnceLock<f64>,
    // r*ln(p)
    #[cfg_attr(feature = "serde1", serde(skip))]
    r_ln_p: OnceLock<f64>,
}

pub struct NegBinomialParameters {
    pub r: f64,
    pub p: f64,
}

impl Parameterized for NegBinomial {
    type Parameters = NegBinomialParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            r: self.r(),
            p: self.p(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.r, params.p)
    }
}

impl PartialEq for NegBinomial {
    fn eq(&self, other: &NegBinomial) -> bool {
        self.r == other.r && self.p == other.p
    }
}

impl NegBinomial {
    /// Create a new Negative Binomial distribution
    #[inline]
    pub fn new(r: f64, p: f64) -> Result<Self, NegBinomialError> {
        if r < 1.0 {
            Err(NegBinomialError::RLessThanOne { r })
        } else if !r.is_finite() {
            Err(NegBinomialError::RNotFinite { r })
        } else if !(0.0..=1.0).contains(&p) {
            Err(NegBinomialError::POutOfRange { p })
        } else if !p.is_finite() {
            Err(NegBinomialError::PNotFinite { p })
        } else {
            Ok(Self::new_unchecked(r, p))
        }
    }

    /// Create a new Negative Binomial distribution without input validation.
    #[inline]
    #[must_use]
    pub fn new_unchecked(r: f64, p: f64) -> Self {
        NegBinomial {
            r,
            p,
            ln_1mp: OnceLock::new(),
            r_ln_p: OnceLock::new(),
        }
    }

    /// Get the value of the `r` parameter
    #[inline]
    pub fn r(&self) -> f64 {
        self.r
    }

    /// Change the value of the r parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::NegBinomial;
    ///
    /// let mut nbin = NegBinomial::new(4.0, 0.8).unwrap();
    ///
    /// assert!((nbin.r() - 4.0).abs() < 1E-10);
    ///
    /// nbin.set_r(2.5).unwrap();
    ///
    /// assert!((nbin.r() - 2.5).abs() < 1E-10);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NegBinomial;
    /// # let mut nbin = NegBinomial::new(4.0, 0.8).unwrap();
    /// assert!(nbin.set_r(2.0).is_ok());
    /// assert!(nbin.set_r(1.0).is_ok());
    ///
    /// // r must be >= 1.0
    /// assert!(nbin.set_r(0.99).is_err());
    ///
    /// assert!(nbin.set_r(f64::INFINITY).is_err());
    /// assert!(nbin.set_r(f64::NEG_INFINITY).is_err());
    /// assert!(nbin.set_r(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_r(&mut self, r: f64) -> Result<(), NegBinomialError> {
        if r < 1.0 {
            Err(NegBinomialError::RLessThanOne { r })
        } else if !r.is_finite() {
            Err(NegBinomialError::RNotFinite { r })
        } else {
            self.set_r_unchecked(r);
            Ok(())
        }
    }

    /// Set the value of r without input validation
    #[inline]
    pub fn set_r_unchecked(&mut self, r: f64) {
        self.r = r;
        self.r_ln_p = OnceLock::new();
    }

    /// Get the value of the `p` parameter
    #[inline]
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Change the value of the p parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::NegBinomial;
    ///
    /// let mut nbin = NegBinomial::new(4.0, 0.8).unwrap();
    ///
    /// assert!((nbin.p() - 0.8).abs() < 1E-10);
    ///
    /// nbin.set_p(0.51).unwrap();
    ///
    /// assert!((nbin.p() - 0.51).abs() < 1E-10);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::NegBinomial;
    /// # let mut nbin = NegBinomial::new(4.0, 0.8).unwrap();
    /// // OK values in [0, 1]
    /// assert!(nbin.set_p(0.51).is_ok());
    /// assert!(nbin.set_p(0.0).is_ok());
    /// assert!(nbin.set_p(1.0).is_ok());
    ///
    /// // Too low, not in [0, 1]
    /// assert!(nbin.set_p(-0.1).is_err());
    ///
    /// // Too high, not in [0, 1]
    /// assert!(nbin.set_p(-1.1).is_err());
    ///
    /// assert!(nbin.set_p(f64::INFINITY).is_err());
    /// assert!(nbin.set_p(f64::NEG_INFINITY).is_err());
    /// assert!(nbin.set_p(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_p(&mut self, p: f64) -> Result<(), NegBinomialError> {
        if !(0.0..=1.0).contains(&p) {
            Err(NegBinomialError::POutOfRange { p })
        } else if !p.is_finite() {
            Err(NegBinomialError::PNotFinite { p })
        } else {
            self.set_p_unchecked(p);
            Ok(())
        }
    }

    /// Set the value of p without input validation
    #[inline]
    pub fn set_p_unchecked(&mut self, p: f64) {
        self.p = p;
        self.ln_1mp = OnceLock::new();
        self.r_ln_p = OnceLock::new();
    }

    #[inline]
    fn ln_1mp(&self) -> f64 {
        *self.ln_1mp.get_or_init(|| (1.0 - self.p).ln())
    }

    #[inline]
    fn r_ln_p(&self) -> f64 {
        *self.r_ln_p.get_or_init(|| self.r * self.p.ln())
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for NegBinomial {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xf = (*x) as f64;
                ln_binom(xf + self.r - 1.0, self.r - 1.0)
                    + xf.mul_add(self.ln_1mp(), self.r_ln_p())
            }
        }

        impl Sampleable<$kind> for NegBinomial {
            fn draw<R: Rng>(&self, mut rng: &mut R) -> $kind {
                let q = 1.0 - self.p;
                let scale = q / (1.0 - q);
                let gamma = rand_distr::Gamma::new(self.r, scale).unwrap();
                let pois_rate = rng.sample(gamma);
                Poisson::new_unchecked(pois_rate).draw(&mut rng)
            }

            fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<$kind> {
                let q = 1.0 - self.p;
                let scale = q / (1.0 - q);
                let gamma = rand_distr::Gamma::new(self.r, scale).unwrap();
                (0..n)
                    .map(|_| {
                        let pois_rate = rng.sample(gamma);
                        Poisson::new_unchecked(pois_rate).draw(&mut rng)
                    })
                    .collect()
            }
        }

        impl DiscreteDistr<$kind> for NegBinomial {}

        impl Support<$kind> for NegBinomial {
            fn supports(&self, _x: &$kind) -> bool {
                // support is [0, Inf), so any unsigned int is OK
                true
            }
        }

        impl Cdf<$kind> for NegBinomial {
            fn cdf(&self, x: &$kind) -> f64 {
                use special::Beta as _;
                let xp1 = (x + 1) as f64;
                let ln_beta = self.r.ln_beta(xp1);
                self.p.inc_beta(self.r, xp1, ln_beta)
            }
        }
    };
}

impl Mean<f64> for NegBinomial {
    fn mean(&self) -> Option<f64> {
        let q = 1.0 - self.p;
        Some((q * self.r) / self.p)
    }
}

impl Variance<f64> for NegBinomial {
    fn variance(&self) -> Option<f64> {
        let q = 1.0 - self.p;
        Some((q * self.r) / (self.p * self.p))
    }
}

impl Skewness for NegBinomial {
    fn skewness(&self) -> Option<f64> {
        let q = 1.0 - self.p;
        Some((2.0 - self.p) / (self.r * q).sqrt())
    }
}

impl Kurtosis for NegBinomial {
    fn kurtosis(&self) -> Option<f64> {
        let q = 1.0 - self.p;
        Some(self.p.mul_add(self.p - 6.0, 6.0) / (self.r * q))
    }
}

impl_traits!(u8);
impl_traits!(u16);
impl_traits!(u32);

impl std::error::Error for NegBinomialError {}

impl fmt::Display for NegBinomialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::POutOfRange { p } => {
                write!(f, "p ({p}) not in range [0, 1]")
            }
            Self::PNotFinite { p } => write!(f, "non-finite p: {p}"),
            Self::RLessThanOne { r } => {
                write!(f, "r ({r}) must be one or greater")
            }
            Self::RNotFinite { r } => write!(f, "non-finite r: {r}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-10;

    test_basic_impls!(u32, NegBinomial, NegBinomial::new(2.1, 0.6).unwrap());

    #[test]
    fn new_with_good_params() {
        let nbin_res = NegBinomial::new(2.1, 0.82);

        if let Ok(nbin) = nbin_res {
            assert::close(nbin.r(), 2.1, TOL);
            assert::close(nbin.p(), 0.82, TOL);
        } else {
            panic!("Negative Binomial constructor failed");
        }
    }

    #[test]
    fn new_with_too_low_r_errors() {
        let nbin_res = NegBinomial::new(0.99999, 0.82);

        match nbin_res {
            Err(NegBinomialError::RLessThanOne { .. }) => (),
            Err(err) => panic!("wrong error {err:?}"),
            Ok(_) => panic!("should have failed"),
        }
    }

    #[test]
    fn new_with_too_low_or_high_p_errors() {
        match NegBinomial::new(2.0, -0.1) {
            Err(NegBinomialError::POutOfRange { .. }) => (),
            Err(err) => panic!("wrong error {err:?}"),
            Ok(_) => panic!("should have failed"),
        }

        match NegBinomial::new(2.0, 1.001) {
            Err(NegBinomialError::POutOfRange { .. }) => (),
            Err(err) => panic!("wrong error {err:?}"),
            Ok(_) => panic!("should have failed"),
        }
    }

    #[test]
    fn set_r_sets_r() {
        let mut nbin = NegBinomial::new(3.0, 0.5).unwrap();
        assert::close(nbin.r(), 3.0, TOL);

        nbin.set_r(4.1).unwrap();
        assert::close(nbin.r(), 4.1, TOL);
    }

    #[test]
    fn set_r_too_low_errors() {
        let mut nbin = NegBinomial::new(3.0, 0.5).unwrap();
        assert::close(nbin.r(), 3.0, TOL);

        match nbin.set_r(0.1) {
            Err(NegBinomialError::RLessThanOne { .. }) => (),
            Err(err) => panic!("wrong error {err:?}"),
            Ok(()) => panic!("should have failed"),
        }
    }

    #[test]
    fn set_p_sets_p() {
        let mut nbin = NegBinomial::new(3.0, 0.5).unwrap();
        assert::close(nbin.p(), 0.5, TOL);

        nbin.set_p(0.9).unwrap();
        assert::close(nbin.p(), 0.9, TOL);
    }

    #[test]
    fn set_p_too_low_errors() {
        let mut nbin = NegBinomial::new(3.0, 0.5).unwrap();
        assert::close(nbin.p(), 0.5, TOL);

        match nbin.set_p(-0.1) {
            Err(NegBinomialError::POutOfRange { .. }) => (),
            Err(err) => panic!("wrong error {err:?}"),
            Ok(()) => panic!("should have failed"),
        }
    }

    #[test]
    fn set_p_too_high_errors() {
        let mut nbin = NegBinomial::new(3.0, 0.5).unwrap();
        assert::close(nbin.p(), 0.5, TOL);

        match nbin.set_p(1.1) {
            Err(NegBinomialError::POutOfRange { .. }) => (),
            Err(err) => panic!("wrong error {err:?}"),
            Ok(()) => panic!("should have failed"),
        }
    }

    #[test]
    fn set_r_preserves_pmf() {
        let mut nbin = NegBinomial::new(2.0, 0.6).unwrap();
        let x = 5_u32;
        let f_start = nbin.f(&x);

        nbin.set_r(3.1).unwrap();
        let f_mid = nbin.f(&x);

        assert!((f_start - f_mid).abs() > TOL);

        nbin.set_r(2.0).unwrap();
        let f_end = nbin.f(&x);

        assert!((f_start - f_end).abs() <= TOL);
    }

    #[test]
    fn set_p_preserves_pmf() {
        let mut nbin = NegBinomial::new(2.0, 0.6).unwrap();
        let x = 5_u32;
        let f_start = nbin.f(&x);

        nbin.set_p(0.9).unwrap();
        let f_mid = nbin.f(&x);

        assert!((f_start - f_mid).abs() > TOL);

        nbin.set_p(0.6).unwrap();
        let f_end = nbin.f(&x);

        assert!((f_start - f_end).abs() <= TOL);
    }

    #[test]
    fn ln_f() {
        // NOTE: Truth values taken from scipy
        let ln_fs = [
            -1.021_651_247_531_981_4,
            -1.244_794_798_846_191_2,
            -1.755_620_422_612_181_9,
            -2.384_229_082_034_555_5,
            -3.077_376_262_594_501_4,
            -3.811_345_437_674_700_7,
            -4.573_485_489_721_598,
            -5.356_244_828_971_23,
            -6.154_752_525_189_004,
            -6.965_682_741_405_329,
            -7.786_663_293_475_16,
            -8.615_942_648_359_688,
            -9.452_190_672_560_304,
            -10.294_373_432_280_734,
            -11.141_671_292_667_94,
        ];
        let nbin = NegBinomial::new(2.0, 0.6).unwrap();
        (0..15).zip(ln_fs.iter()).for_each(|(x, ln_f)| {
            assert::close(nbin.ln_f(&(x as u32)), *ln_f, 1E-8);
        });
    }

    #[test]
    fn f() {
        // NOTE: Truth values taken from scipy
        let fs = [
            0.36,
            0.288,
            0.172_799_999_999_999_98,
            0.092_160_000_000_000_03,
            0.046_079_999_999_999_996,
            0.022_118_400_000_000_02,
            0.010_321_920_000_000_004,
            0.004_718_592_000_000_006,
            0.002_123_366_399_999_997_8,
            0.000_943_718_400_000_001_8,
            0.000_415_236_096_000_000_8,
            0.000_181_193_932_799_999_74,
            7.851_737_088_000_01e-05,
            3.382_286_745_600_015_4e-5,
            1.449_551_462_400_003e-5,
        ];
        let nbin = NegBinomial::new(2.0, 0.6).unwrap();
        (0..15).zip(fs.iter()).for_each(|(x, f)| {
            assert::close(nbin.f(&(x as u32)), *f, 1E-8);
        });
    }

    #[test]
    fn cdf() {
        // NOTE: Truth values taken from scipy
        let cdfs = [
            0.064_000_000_000_000_02,
            0.179_200_000_000_000_03,
            0.317_440_000_000_000_3,
            0.455_680_000_000_000_1,
            0.580_096_000_000_000_2,
            0.684_605_439_999_999_9,
            0.768_212_992,
            0.832_710_246_4,
            0.881_083_187_2,
            0.916_556_677_12,
            0.942_097_589_862_4,
            0.960_208_418_897_920_1,
            0.972_885_999_222_784,
            0.981_662_785_601_536,
            0.987_681_153_404_108_8,
        ];
        let nbin = NegBinomial::new(3.0, 0.4).unwrap();
        (0..15).zip(cdfs.iter()).for_each(|(x, cdf)| {
            assert::close(nbin.cdf(&(x as u32)), *cdf, 1E-8);
        });
    }

    #[test]
    fn mean() {
        let mean = NegBinomial::new(3.0, 0.4).unwrap().mean().unwrap();
        assert::close(mean, 4.5, TOL);
    }

    #[test]
    fn variance() {
        let var = NegBinomial::new(3.0, 0.4).unwrap().variance().unwrap();
        assert::close(var, 11.25, TOL);
    }

    #[test]
    fn skewness() {
        let skew = NegBinomial::new(3.0, 0.4).unwrap().skewness().unwrap();
        assert::close(skew, 1.192_569_587_999_887_9, TOL);
    }

    #[test]
    fn kurtosis() {
        let kts = NegBinomial::new(3.0, 0.4).unwrap().kurtosis().unwrap();
        assert::close(kts, 2.088_888_888_888_889, TOL);
    }

    #[test]
    fn clone_preserves_pmf() {
        let nbin_1 = NegBinomial::new(3.2, 0.8).unwrap();
        let nbin_2 = nbin_1.clone();

        let x = 2_u32;
        assert::close(nbin_1.f(&x), nbin_2.f(&x), TOL);
    }

    #[test]
    fn partial_eq() {
        let nbin_1 = NegBinomial::new(3.2, 0.8).unwrap();
        let nbin_2 = nbin_1.clone();
        let nbin_3 = NegBinomial::new(3.2, 0.7).unwrap();
        let nbin_4 = NegBinomial::new(3.1, 0.8).unwrap();
        let nbin_5 = NegBinomial::new(3.1, 0.7).unwrap();

        assert_eq!(nbin_1, nbin_2);
        assert_ne!(nbin_1, nbin_3);
        assert_ne!(nbin_1, nbin_4);
        assert_ne!(nbin_1, nbin_5);
    }

    #[test]
    fn sample_test() {
        use crate::misc::x2_test;

        let n_tries = 5;
        let x2_pval = 0.2;
        let mut rng = rand::rng();
        let nbin = NegBinomial::new(3.0, 0.6).unwrap();

        // How many bins do we need?
        let k: usize = (0..100)
            .position(|x| nbin.pmf(&(x as u32)) < f64::EPSILON)
            .unwrap_or(99)
            + 1;

        let ps: Vec<f64> = (0..k).map(|x| nbin.pmf(&(x as u32))).collect();

        let passes = (0..n_tries).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; k];
            let xs: Vec<u32> = nbin.sample(1000, &mut rng);
            xs.iter().for_each(|&x| f_obs[x as usize] += 1);
            let (_, p) = x2_test(&f_obs, &ps);
            if p > x2_pval { acc + 1 } else { acc }
        });

        assert!(passes > 0);
    }

    #[test]
    fn draw_test() {
        use crate::misc::x2_test;

        let n_tries = 5;
        let x2_pval = 0.2;
        let mut rng = rand::rng();
        let nbin = NegBinomial::new(3.0, 0.6).unwrap();

        // How many bins do we need?
        let k: usize = (0..100)
            .position(|x| nbin.pmf(&(x as u32)) < f64::EPSILON)
            .unwrap_or(99)
            + 1;

        let ps: Vec<f64> = (0..k).map(|x| nbin.pmf(&(x as u32))).collect();

        let passes = (0..n_tries).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; k];
            let xs: Vec<u32> = (0..1000).map(|_| nbin.draw(&mut rng)).collect();
            xs.iter().for_each(|&x| f_obs[x as usize] += 1);
            let (_, p) = x2_test(&f_obs, &ps);
            if p > x2_pval { acc + 1 } else { acc }
        });

        assert!(passes > 0);
    }
}
