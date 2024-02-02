//! UnitPowerLaw distribution over x in (0, 1)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::prelude::Beta;  
use crate::data::UnitPowerLawSuffStat;
use crate::impl_display;
use crate::traits::*;
use rand::Rng;
// use special::UnitPowerLaw as _;
use num_traits::Inv;
use special::Gamma as _;
use std::f64;
use std::fmt;
use std::sync::OnceLock;

// pub mod bernoulli_prior;

/// UnitPowerLaw(α) over x in (0, 1).
///
/// # Examples
///
/// UnitPowerLaw as a conjugate prior for Bernoulli
///
/// ```
/// use rv::prelude::*;
///
/// // A prior that encodes our strong belief that coins are fair:
/// let powlaw = UnitPowerLaw::new(5.0, 5.0).unwrap();
///
/// // The posterior predictive probability that a coin will come up heads given
/// // no new observations.
/// let p_prior_heads = powlaw.pp(&true, &DataOrSuffStat::None); // 0.5
/// assert!((p_prior_heads - 0.5).abs() < 1E-12);
///
/// // Five Bernoulli trials. We flipped a coin five times and it came up head
/// // four times.
/// let flips = vec![true, true, false, true, true];
///
/// // The posterior predictive probability that a coin will come up heads given
/// // the five flips we just saw.
/// let p_pred_heads = powlaw.pp(&true, &DataOrSuffStat::Data(&flips)); // 9/15
/// assert!((p_pred_heads - 3.0/5.0).abs() < 1E-12);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct UnitPowerLaw {
    alpha: f64,
    
    // Cached alpha.inv()
    #[cfg_attr(feature = "serde1", serde(skip))]
    alpha_inv: OnceLock<f64>,

    // Cached alpha.ln()
    #[cfg_attr(feature = "serde1", serde(skip))]
    alpha_ln: OnceLock<f64>,
}

impl PartialEq for UnitPowerLaw {
    fn eq(&self, other: &UnitPowerLaw) -> bool {
        self.alpha == other.alpha
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum UnitPowerLawError {
    /// The alpha parameter is less than or equal to zero
    AlphaTooLow { alpha: f64 },
    /// The alpha parameter is infinite or NaN
    AlphaNotFinite { alpha: f64 },
}

impl UnitPowerLaw {
    /// Create a `UnitPowerLaw` distribution with even density over (0, 1).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::powlaw::UnitPowerLaw;
    /// // Uniform
    /// let powlaw_unif = UnitPowerLaw::new(1.0, 1.0);
    /// assert!(powlaw_unif.is_ok());
    ///
    /// // Jefferey's prior
    /// let powlaw_jeff  = UnitPowerLaw::new(0.5, 0.5);
    /// assert!(powlaw_jeff.is_ok());
    ///
    /// // Invalid negative parameter
    /// let powlaw_nope  = UnitPowerLaw::new(-5.0, 1.0);
    /// assert!(powlaw_nope.is_err());
    /// ```
    pub fn new(alpha: f64) -> Result<Self, UnitPowerLawError> {
        if alpha <= 0.0 {
            Err(UnitPowerLawError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(UnitPowerLawError::AlphaNotFinite { alpha })
        } else {
            Ok(UnitPowerLaw {
                alpha,
                alpha_inv: OnceLock::new(),
                alpha_ln: OnceLock::new(),
            })
        }
    }

    /// Creates a new UnitPowerLaw without checking whether the parameters are valid.
    #[inline]
    pub fn new_unchecked(alpha: f64) -> Self {
        UnitPowerLaw {
            alpha,
            alpha_inv: OnceLock::new(),
            alpha_ln: OnceLock::new(),
        }
    }

    /// Create a `UnitPowerLaw` distribution with even density over (0, 1).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::powlaw::UnitPowerLaw;
    /// let powlaw = UnitPowerLaw::uniform();
    /// assert_eq!(powlaw, UnitPowerLaw::new(1.0, 1.0).unwrap());
    /// ```
    #[inline]
    pub fn uniform() -> Self {
        UnitPowerLaw {
            alpha: 1.0,
            alpha_inv: OnceLock::new(),
            alpha_ln: OnceLock::new(),
        }
    }

    // /// Create a `UnitPowerLaw` distribution with the Jeffrey's parameterization,
    // /// *UnitPowerLaw(0.5, 0.5)*.
    // ///
    // /// # Example
    // ///
    // /// ```rust
    // /// # use rv::powlaw::UnitPowerLaw;
    // /// let powlaw = UnitPowerLaw::jeffreys();
    // /// assert_eq!(powlaw, UnitPowerLaw::new(0.5, 0.5).unwrap());
    // /// ```
    // #[inline]
    // pub fn jeffreys() -> Self {
    //     UnitPowerLaw {
    //         alpha: 0.5,
    //         powlaw: 0.5,
    //         ln_powlaw_ab: OnceLock::new(),
    //     }
    // }

    /// Get the alpha parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::powlaw::UnitPowerLaw;
    /// let powlaw = UnitPowerLaw::new(1.0, 5.0).unwrap();
    /// assert_eq!(powlaw.alpha(), 1.0);
    /// ```
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set the alpha parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::powlaw::UnitPowerLaw;
    /// let mut powlaw = UnitPowerLaw::new(1.0, 5.0).unwrap();
    ///
    /// powlaw.set_alpha(2.0).unwrap();
    /// assert_eq!(powlaw.alpha(), 2.0);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::powlaw::UnitPowerLaw;
    /// # let mut powlaw = UnitPowerLaw::new(1.0, 5.0).unwrap();
    /// assert!(powlaw.set_alpha(0.1).is_ok());
    /// assert!(powlaw.set_alpha(0.0).is_err());
    /// assert!(powlaw.set_alpha(-1.0).is_err());
    /// assert!(powlaw.set_alpha(std::f64::INFINITY).is_err());
    /// assert!(powlaw.set_alpha(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_alpha(&mut self, alpha: f64) -> Result<(), UnitPowerLawError> {
        if alpha <= 0.0 {
            Err(UnitPowerLawError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(UnitPowerLawError::AlphaNotFinite { alpha })
        } else {
            self.set_alpha_unchecked(alpha);
            Ok(())
        }
    }

    /// Set alpha without input validation
    #[inline]
    pub fn set_alpha_unchecked(&mut self, alpha: f64) {
        self.alpha = alpha;
        self.alpha_inv = OnceLock::new();
        self.alpha_ln = OnceLock::new();
    }

    /// Evaluate or fetch cached ln(a*b)
    #[inline]
    fn alpha_inv(&self) -> f64 {
        *self.alpha_inv.get_or_init(|| self.alpha.inv())
    }

    /// Evaluate or fetch cached ln(a*b)
    #[inline]
    fn alpha_ln(&self) -> f64 {
        *self.alpha_ln.get_or_init(|| self.alpha.ln())
    }
}

impl From<&UnitPowerLaw> for Beta {
    fn from(powlaw: &UnitPowerLaw) -> Beta {
        Beta::new(powlaw.alpha, 1.0).unwrap()
    }
}

impl Default for UnitPowerLaw {
    fn default() -> Self {
        todo!()
    }
}

impl From<&UnitPowerLaw> for String {
    fn from(powlaw: &UnitPowerLaw) -> String {
        format!("UnitPowerLaw(α: {})", powlaw.alpha)
    }
}

impl_display!(UnitPowerLaw);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for UnitPowerLaw {
            fn ln_f(&self, x: &$kind) -> f64 {
                // TODO: If we evaluate a lot of xs for a fixed alpha (which
                // seems likely), we should cache self.alpha.ln()
                (*x as f64).ln().mul_add(self.alpha - 1.0, self.alpha_ln())
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                rng.gen::<f64>().powf(self.alpha_inv()) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let alpha_inv = self.alpha_inv() as $kind;
                (0..n).map(|_| rng.gen::<$kind>().powf(alpha_inv)).collect()
            }
        }

        impl Support<$kind> for UnitPowerLaw {
            fn supports(&self, x: &$kind) -> bool {
                let xf = f64::from(*x);
                0.0 < xf && xf < 1.0
            }
        }

        impl ContinuousDistr<$kind> for UnitPowerLaw {}

        impl Cdf<$kind> for UnitPowerLaw {
            fn cdf(&self, x: &$kind) -> f64 {
                (*x as f64).powf(self.alpha)
            }
        }

        impl Mean<$kind> for UnitPowerLaw {
            fn mean(&self) -> Option<$kind> {
                Some((self.alpha / (self.alpha + 1.0)) as $kind)
            }
        }

        impl Mode<$kind> for UnitPowerLaw {
            fn mode(&self) -> Option<$kind> {
                if self.alpha > 1.0 {
                    Some(1.0)
                } else {
                    None
                }
            }
        }

        impl HasSuffStat<$kind> for UnitPowerLaw {
            type Stat = UnitPowerLawSuffStat;

            fn empty_suffstat(&self) -> Self::Stat {
                Self::Stat::new()
            }

            fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
                let n = stat.n() as f64;
                let t1 = n * self.alpha_inv();
                let t2 = (self.alpha - 1.0) * stat.sum_ln_x();
                t2 - t1
            }
        }
    };
}

impl Variance<f64> for UnitPowerLaw {
    fn variance(&self) -> Option<f64> {
        let apb = self.alpha + 1.0;
        Some(self.alpha / (apb * apb * (apb + 1.0)))
    }
}

impl Entropy for UnitPowerLaw {
    fn entropy(&self) -> f64 {
        let apb = self.alpha + 1.0;
        (apb - 2.0).mul_add(
            apb.digamma(),

                (self.alpha - 1.0)
                    .mul_add(-self.alpha.digamma(), -self.alpha_ln()),
            )
        
    }
}

impl Skewness for UnitPowerLaw {
    fn skewness(&self) -> Option<f64> {
        let apb = self.alpha + 1.0;
        let numer = 2.0 * (1.0 - self.alpha) * (apb + 1.0).sqrt();
        let denom = (apb + 2.0) * (self.alpha * 1.0).sqrt();
        Some(numer / denom)
    }
}

impl Kurtosis for UnitPowerLaw {
    fn kurtosis(&self) -> Option<f64> {
        let apb = self.alpha + 1.0;
        let amb = self.alpha - 1.0;
        let atb = self.alpha * 1.0;
        let numer = 6.0 * (amb * amb).mul_add(apb + 1.0, -atb * (apb + 2.0));
        let denom = atb * (apb + 2.0) * (apb + 3.0);
        Some(numer / denom)
    }
}

impl_traits!(f32);
impl_traits!(f64);

impl std::error::Error for UnitPowerLawError {}

impl fmt::Display for UnitPowerLawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlphaTooLow { alpha } => {
                write!(f, "alpha ({}) must be greater than zero", alpha)
            }
            Self::AlphaNotFinite { alpha } => {
                write!(f, "alpha ({}) was non finite", alpha)
            }
        }
    }
}

#[cfg(test)]
mod tests {

    // use argmin::solver::conjugategradient::beta;

    use super::*;
    use crate::misc::ks_test;
    use crate::test_basic_impls;
    use std::f64;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!([continuous] UnitPowerLaw::new(1.5).unwrap());
    // test_basic_impls!([continuous] UnitPowerLaw::jeffreys());

    #[test]
    fn new() {
        let powlaw = UnitPowerLaw::new(2.0).unwrap();
        assert::close(powlaw.alpha, 2.0, TOL);
    }

    #[test]
    fn uniform() {
        let powlaw = UnitPowerLaw::uniform();
        assert::close(powlaw.alpha, 1.0, TOL);
    }

    // #[test]
    // fn jeffreys() {
    //     let powlaw = UnitPowerLaw::jeffreys();
    //     assert::close(powlaw.alpha, 0.5, TOL);
    //     assert::close(powlaw.powlaw, 0.5, TOL);
    // }

    #[test]
    fn ln_pdf_center_value() {
        let powlaw = UnitPowerLaw::new(1.5).unwrap();

        assert::close(powlaw.ln_pdf(&0.5), 0.282_035_069_142_401_84, TOL);
    }

    #[test]
    fn ln_pdf_low_value() {
        let powlaw = UnitPowerLaw::new(1.5).unwrap();
        assert::close(powlaw.ln_pdf(&0.01), -0.990_879_588_865_227_3, TOL);
    }

    #[test]
    fn ln_pdf_high_value() {
        let powlaw = UnitPowerLaw::new(1.5).unwrap();
        assert::close(powlaw.ln_pdf(&0.99), -3.288_439_513_932_521_8, TOL);
    }

    #[test]
    fn pdf_preserved_after_set_reset_alpha() {
        let x: f64 = 0.6;
        let alpha = 1.5;

        let mut powlaw = UnitPowerLaw::new(alpha).unwrap();

        let f_1 = powlaw.f(&x);
        let ln_f_1 = powlaw.ln_f(&x);

        powlaw.set_alpha(3.4).unwrap();

        assert_ne!(f_1, powlaw.f(&x));
        assert_ne!(ln_f_1, powlaw.ln_f(&x));

        powlaw.set_alpha(alpha).unwrap();

        assert_eq!(f_1, powlaw.f(&x));
        assert_eq!(ln_f_1, powlaw.ln_f(&x));
    }


    #[test]
    fn cdf_hump_shaped() {
        let powlaw = UnitPowerLaw::new(1.5).unwrap();
        let beta: Beta = (&powlaw).into();
        let xs: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        for x in xs.iter() {
            assert::close(powlaw.cdf(x), beta.cdf(x), TOL);
        }
    }

    #[test]
    fn cdf_bowl_shaped() {
        let powlaw = UnitPowerLaw::new(0.5).unwrap();
        let beta: Beta = (&powlaw).into();
        let xs: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        for x in xs.iter() {
            assert::close(powlaw.cdf(x), beta.cdf(x), TOL);
        }
    }

    #[test]
    fn draw_should_resturn_values_within_0_to_1() {
        let mut rng = rand::thread_rng();
        let powlaw = UnitPowerLaw::new(2.0).unwrap();
        for _ in 0..100 {
            let x = powlaw.draw(&mut rng);
            assert!(0.0 < x && x < 1.0);
        }
    }

    #[test]
    fn sample_returns_the_correct_number_draws() {
        let mut rng = rand::thread_rng();
        let powlaw = UnitPowerLaw::new(2.0).unwrap();
        let xs: Vec<f32> = powlaw.sample(103, &mut rng);
        assert_eq!(xs.len(), 103);
    }

    #[test]
    fn uniform_mean() {
        let mean: f64 = UnitPowerLaw::uniform().mean().unwrap();
        assert::close(mean, 0.5, TOL);
    }

    // #[test]
    // fn jeffreys_mean() {
    //     let mean: f64 = UnitPowerLaw::jeffreys().mean().unwrap();
    //     assert::close(mean, 0.5, TOL);
    // }

    #[test]
    fn mean() {
        let mean: f64 = UnitPowerLaw::new(5.0).unwrap().mean().unwrap();
        assert::close(mean, 5.0 / 6.0, TOL);
    }

    #[test]
    fn variance() {
        let powlaw = UnitPowerLaw::new(1.5).unwrap();
        let beta: Beta = (&powlaw).into();
        assert::close(powlaw.variance().unwrap(), beta.variance().unwrap(), TOL);
    }

    #[test]
    fn mode_for_alpha_and_powlaw_greater_than_one() {
        let mode: f64 = UnitPowerLaw::new(1.5).unwrap().mode().unwrap();
        assert::close(mode, 0.5 / 1.5, TOL);
    }

    #[test]
    fn mode_for_alpha_one_and_large_powlaw() {
        let mode: f64 = UnitPowerLaw::new(2.0).unwrap().mode().unwrap();
        assert::close(mode, 0.0, TOL);
    }

    #[test]
    fn mode_for_large_alpha_and_powlaw_one() {
        let mode: f64 = UnitPowerLaw::new(2.0).unwrap().mode().unwrap();
        assert::close(mode, 1.0, TOL);
    }

    #[test]
    fn mode_for_alpha_less_than_one_is_none() {
        let mode_opt: Option<f64> =
            UnitPowerLaw::new(0.99).unwrap().mode();
        assert!(mode_opt.is_none());
    }

    #[test]
    fn entropy() {
        let powlaw = UnitPowerLaw::new(1.5).unwrap();
        let beta: Beta = (&powlaw).into();
        assert::close(powlaw.entropy(), beta.entropy(), TOL);
    }

    #[test]
    fn uniform_skewness_should_be_zero() {
        assert::close(UnitPowerLaw::uniform().skewness().unwrap(), 0.0, TOL);
    }

    // #[test]
    // fn jeffreysf_skewness_should_be_zero() {
    //     assert::close(UnitPowerLaw::jeffreys().skewness().unwrap(), 0.0, TOL);
    // }

    #[test]
    fn skewness() {
        let powlaw = UnitPowerLaw::new(1.5).unwrap();
        let beta: Beta = (&powlaw).into();
        assert::close(powlaw.skewness().unwrap(), beta.skewness().unwrap(), TOL);
    }

    #[test]
    fn kurtosis() {
        let powlaw = UnitPowerLaw::new(1.5).unwrap();
        let beta: Beta = (&powlaw).into();
        assert::close(powlaw.kurtosis().unwrap(), beta.kurtosis().unwrap(), TOL);
    }

    #[test]
    fn draw_test_alpha_powlaw_gt_one() {
        let mut rng = rand::thread_rng();
        let powlaw = UnitPowerLaw::new(1.2).unwrap();
        let cdf = |x: f64| powlaw.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = powlaw.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }

    #[test]
    fn draw_test_alpha_powlaw_lt_one() {
        let mut rng = rand::thread_rng();
        let powlaw = UnitPowerLaw::new(0.2).unwrap();
        let cdf = |x: f64| powlaw.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = powlaw.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }

    #[test]
    fn ln_f_stat() {
        let data: Vec<f64> = vec![0.1, 0.23, 0.4, 0.65, 0.22, 0.31];
        let mut stat = UnitPowerLawSuffStat::new();
        stat.observe_many(&data);

        let powlaw = UnitPowerLaw::new(0.3).unwrap();

        let ln_f_base: f64 = data.iter().map(|x| powlaw.ln_f(x)).sum();
        let ln_f_stat: f64 =
            <UnitPowerLaw as HasSuffStat<f64>>::ln_f_stat(&powlaw, &stat);

        assert::close(ln_f_base, ln_f_stat, 1e-12);
    }

    #[test]
    fn set_alpha() {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let a1 = rng.gen::<f64>();
            let mut powlaw1 = UnitPowerLaw::new(a1).unwrap();

            // Any value in the unit interval
            let x: f64 = rng.gen();

            // Evaluate the pdf to force computation of `ln_powlaw_ab`
            let _ = powlaw1.pdf(&x);

            // Next we'll `set_alpha` to a2, and compare with a fresh UnitPowerLaw
            let a2 = rng.gen::<f64>();

            // Setting the new values
            powlaw1.set_alpha(a2).unwrap();

            // ... and here's the fresh version
            let powlaw2 = UnitPowerLaw::new(a2).unwrap();

            let pdf_1 = powlaw1.ln_f(&x);
            let pdf_2 = powlaw2.ln_f(&x);

            assert::close(pdf_1, pdf_2, 1e-14);
        }
    }

}
