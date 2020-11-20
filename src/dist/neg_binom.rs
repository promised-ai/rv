use crate::clone_cache_f64;
use crate::dist::Poisson;
use crate::misc::ln_binom;
use crate::traits::*;
use once_cell::sync::OnceCell;
use rand::Rng;
use std::fmt;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Negative Binomial distribution errors
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
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
#[derive(Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NegBinomial {
    r: f64,
    p: f64,
    // ln(1-p)
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_1mp: OnceCell<f64>,
    // r*ln(p)
    #[cfg_attr(feature = "serde1", serde(skip))]
    r_ln_p: OnceCell<f64>,
}

impl Clone for NegBinomial {
    fn clone(&self) -> Self {
        Self {
            r: self.r,
            p: self.p,
            ln_1mp: clone_cache_f64!(self, ln_1mp),
            r_ln_p: clone_cache_f64!(self, r_ln_p),
        }
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
        } else if 1.0 < p || p < 0.0 {
            Err(NegBinomialError::POutOfRange { p })
        } else if !p.is_finite() {
            Err(NegBinomialError::PNotFinite { p })
        } else {
            Ok(Self::new_unchecked(r, p))
        }
    }

    /// Create a new Negative Binomial distribution without input validation.
    #[inline]
    pub fn new_unchecked(r: f64, p: f64) -> Self {
        NegBinomial {
            r,
            p,
            ln_1mp: OnceCell::new(),
            r_ln_p: OnceCell::new(),
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
    /// assert!(nbin.set_r(std::f64::INFINITY).is_err());
    /// assert!(nbin.set_r(std::f64::NEG_INFINITY).is_err());
    /// assert!(nbin.set_r(std::f64::NAN).is_err());
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
        self.r_ln_p = OnceCell::new();
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
    /// assert!(nbin.set_p(std::f64::INFINITY).is_err());
    /// assert!(nbin.set_p(std::f64::NEG_INFINITY).is_err());
    /// assert!(nbin.set_p(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_p(&mut self, p: f64) -> Result<(), NegBinomialError> {
        if 1.0 < p || p < 0.0 {
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
        self.ln_1mp = OnceCell::new();
        self.r_ln_p = OnceCell::new();
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
        impl Rv<$kind> for NegBinomial {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xf = (*x) as f64;
                ln_binom(xf + self.r - 1.0, self.r - 1.0)
                    + self.r_ln_p()
                    + xf * self.ln_1mp()
            }

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
        Some((q * self.r) / self.p.powi(2))
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
        Some((self.p * (self.p - 6.0) + 6.0) / (self.r * q))
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
                write!(f, "p ({}) not in range [0, 1]", p)
            }
            Self::PNotFinite { p } => write!(f, "non-finite p: {}", p),
            Self::RLessThanOne { r } => {
                write!(f, "r ({}) must be one or greater", r)
            }
            Self::RNotFinite { r } => write!(f, "non-finite r: {}", r),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-10;

    test_basic_impls!([count] NegBinomial::new(2.1, 0.6).unwrap());

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
            Err(err) => panic!("wrong error {:?}", err),
            Ok(_) => panic!("should have failed"),
        }
    }

    #[test]
    fn new_with_too_low_or_high_p_errors() {
        match NegBinomial::new(2.0, -0.1) {
            Err(NegBinomialError::POutOfRange { .. }) => (),
            Err(err) => panic!("wrong error {:?}", err),
            Ok(_) => panic!("should have failed"),
        }

        match NegBinomial::new(2.0, 1.001) {
            Err(NegBinomialError::POutOfRange { .. }) => (),
            Err(err) => panic!("wrong error {:?}", err),
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
            Err(err) => panic!("wrong error {:?}", err),
            Ok(_) => panic!("should have failed"),
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
            Err(err) => panic!("wrong error {:?}", err),
            Ok(_) => panic!("should have failed"),
        }
    }

    #[test]
    fn set_p_too_high_errors() {
        let mut nbin = NegBinomial::new(3.0, 0.5).unwrap();
        assert::close(nbin.p(), 0.5, TOL);

        match nbin.set_p(1.1) {
            Err(NegBinomialError::POutOfRange { .. }) => (),
            Err(err) => panic!("wrong error {:?}", err),
            Ok(_) => panic!("should have failed"),
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
        let ln_fs = vec![
            -1.0216512475319814,
            -1.2447947988461912,
            -1.7556204226121819,
            -2.3842290820345555,
            -3.0773762625945014,
            -3.8113454376747007,
            -4.573485489721598,
            -5.35624482897123,
            -6.154752525189004,
            -6.965682741405329,
            -7.78666329347516,
            -8.615942648359688,
            -9.452190672560304,
            -10.294373432280734,
            -11.14167129266794,
        ];
        let nbin = NegBinomial::new(2.0, 0.6).unwrap();
        (0..15).zip(ln_fs.iter()).for_each(|(x, ln_f)| {
            assert::close(nbin.ln_f(&(x as u32)), *ln_f, 1E-8);
        });
    }

    #[test]
    fn f() {
        // NOTE: Truth values taken from scipy
        let fs = vec![
            0.36,
            0.288,
            0.17279999999999998,
            0.09216000000000003,
            0.046079999999999996,
            0.02211840000000002,
            0.010321920000000004,
            0.004718592000000006,
            0.0021233663999999978,
            0.0009437184000000018,
            0.0004152360960000008,
            0.00018119393279999974,
            7.85173708800001e-05,
            3.3822867456000154e-05,
            1.449551462400003e-05,
        ];
        let nbin = NegBinomial::new(2.0, 0.6).unwrap();
        (0..15).zip(fs.iter()).for_each(|(x, f)| {
            assert::close(nbin.f(&(x as u32)), *f, 1E-8);
        });
    }

    #[test]
    fn cdf() {
        // NOTE: Truth values taken from scipy
        let cdfs = vec![
            0.06400000000000002,
            0.17920000000000003,
            0.3174400000000003,
            0.4556800000000001,
            0.5800960000000002,
            0.6846054399999999,
            0.768212992,
            0.8327102464,
            0.8810831872,
            0.91655667712,
            0.9420975898624,
            0.9602084188979201,
            0.972885999222784,
            0.981662785601536,
            0.9876811534041088,
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
        assert::close(skew, 1.1925695879998879, TOL);
    }

    #[test]
    fn kurtosis() {
        let kts = NegBinomial::new(3.0, 0.4).unwrap().kurtosis().unwrap();
        assert::close(kts, 2.088888888888889, TOL);
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
        let mut rng = rand::thread_rng();
        let nbin = NegBinomial::new(3.0, 0.6).unwrap();

        // How many bins do we need?
        let k: usize = (0..100)
            .position(|x| nbin.pmf(&(x as u32)) < std::f64::EPSILON)
            .unwrap_or(99)
            + 1;

        let ps: Vec<f64> = (0..k).map(|x| nbin.pmf(&(x as u32))).collect();

        let passes = (0..n_tries).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; k];
            let xs: Vec<u32> = nbin.sample(1000, &mut rng);
            xs.iter().for_each(|&x| f_obs[x as usize] += 1);
            let (_, p) = x2_test(&f_obs, &ps);
            if p > x2_pval {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }

    #[test]
    fn draw_test() {
        use crate::misc::x2_test;

        let n_tries = 5;
        let x2_pval = 0.2;
        let mut rng = rand::thread_rng();
        let nbin = NegBinomial::new(3.0, 0.6).unwrap();

        // How many bins do we need?
        let k: usize = (0..100)
            .position(|x| nbin.pmf(&(x as u32)) < std::f64::EPSILON)
            .unwrap_or(99)
            + 1;

        let ps: Vec<f64> = (0..k).map(|x| nbin.pmf(&(x as u32))).collect();

        let passes = (0..n_tries).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; k];
            let xs: Vec<u32> = (0..1000).map(|_| nbin.draw(&mut rng)).collect();
            xs.iter().for_each(|&x| f_obs[x as usize] += 1);
            let (_, p) = x2_test(&f_obs, &ps);
            if p > x2_pval {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }
}
