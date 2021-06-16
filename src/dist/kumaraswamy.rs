//! Kumaraswamy distribution over x in (0, 1)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::consts::EULER_MASCERONI;
use crate::traits::*;
use crate::{clone_cache_f64, impl_display};
use once_cell::sync::OnceCell;
use rand::Rng;
use special::Gamma as _;
use std::f64;
use std::fmt;

/// [Kumaraswamy distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution),
/// Kumaraswamy(α, β) over x in (0, 1).
///
/// # Examples
///
/// The relationship between the CDF and the inverse CDF.
///
/// ```
/// # use rv::prelude::*;
/// let kuma = Kumaraswamy::new(2.1, 3.4).unwrap();
///
/// let x = 0.6_f64;
/// let p: f64 = kuma.cdf(&x);
/// let y: f64 = kuma.invcdf(p);
///
/// assert::close(x, y, 1E-10);
/// ```
///
/// Kumaraswamy(a, 1) is equivalent to Beta(a, 1)  and Kumaraswamy(1, b) is equivalent to Beta(1, b)
///
/// ```
/// # use rv::prelude::*;
/// let kuma = Kumaraswamy::new(1.0, 3.5).unwrap();
/// let beta = Beta::new(1.0, 3.5).unwrap();
///
/// let xs = rv::misc::linspace(0.1, 0.9, 10);
///
/// for x in xs.iter() {
///     assert::close(kuma.f(x), beta.f(x), 1E-10);
/// }
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Kumaraswamy {
    a: f64,
    b: f64,
    #[cfg_attr(feature = "serde1", serde(skip))]
    /// Cached log(a*b)
    ab_ln: OnceCell<f64>,
}

impl Clone for Kumaraswamy {
    fn clone(&self) -> Self {
        Self {
            a: self.a,
            b: self.b,
            ab_ln: clone_cache_f64!(self, ab_ln),
        }
    }
}

impl PartialEq for Kumaraswamy {
    fn eq(&self, other: &Kumaraswamy) -> bool {
        self.a == other.a && self.b == other.b
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum KumaraswamyError {
    /// The a parameter is less than or equal to zero
    ATooLow { a: f64 },
    /// The a parameter is infinite or NaN
    ANotFinite { a: f64 },
    /// The b parameter is less than or equal to zero
    BTooLow { b: f64 },
    /// The b parameter is infinite or NaN
    BNotFinite { b: f64 },
}

impl Default for Kumaraswamy {
    fn default() -> Self {
        Kumaraswamy::uniform()
    }
}

impl From<&Kumaraswamy> for String {
    fn from(kuma: &Kumaraswamy) -> String {
        format!("Kumaraswamy(a: {}, b: {})", kuma.a, kuma.b)
    }
}

impl_display!(Kumaraswamy);

impl Kumaraswamy {
    /// Create a `Beta` distribution with even density over (0, 1).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    ///
    /// let kuma_good = Kumaraswamy::new(1.0, 1.0);
    /// assert!(kuma_good.is_ok());
    ///
    /// // Invalid negative parameter
    /// let kuma_bad  = Kumaraswamy::new(-5.0, 1.0);
    /// assert!(kuma_bad.is_err());
    /// ```
    #[inline]
    pub fn new(a: f64, b: f64) -> Result<Self, KumaraswamyError> {
        if a <= 0.0 {
            Err(KumaraswamyError::ATooLow { a })
        } else if !a.is_finite() {
            Err(KumaraswamyError::ANotFinite { a })
        } else if b <= 0.0 {
            Err(KumaraswamyError::BTooLow { b })
        } else if !b.is_finite() {
            Err(KumaraswamyError::BNotFinite { b })
        } else {
            Ok(Kumaraswamy {
                a,
                b,
                ab_ln: OnceCell::new(),
            })
        }
    }

    /// Creates a new Kumaraswamy without checking whether the parameters are
    /// valid.
    #[inline]
    pub fn new_unchecked(a: f64, b: f64) -> Self {
        Kumaraswamy {
            a,
            b,
            ab_ln: OnceCell::new(),
        }
    }

    /// Create a `Kumaraswamy` distribution with even density over (0, 1).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// let kuma = Kumaraswamy::uniform();
    /// assert_eq!(kuma, Kumaraswamy::new(1.0, 1.0).unwrap());
    /// ```
    #[inline]
    pub fn uniform() -> Self {
        Kumaraswamy {
            a: 1.0,
            b: 1.0,
            ab_ln: OnceCell::from(0.0),
        }
    }

    /// Create a `Kumaraswamy` distribution with median 0.5
    ///
    /// # Notes
    ///
    /// The distribution will not necessarily be symmetrical about x = 0.5,
    /// i.e., for c < 0.5, f(0.5 - c) may not equal f(0.5 + c).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// # use rv::traits::{Rv, Cdf, Median};
    /// // Bowl-shaped
    /// let kuma_1 = Kumaraswamy::centered(0.5).unwrap();
    /// let median_1: f64 = kuma_1.median().unwrap();
    /// assert::close(median_1, 0.5, 1E-10);
    /// assert::close(kuma_1.cdf(&0.5), 0.5, 1E-10);
    /// assert::close(kuma_1.b(), 0.5644763825137, 1E-10);
    ///
    /// // Cone-shaped
    /// let kuma_2 = Kumaraswamy::centered(2.0).unwrap();
    /// let median_2: f64 = kuma_2.median().unwrap();
    /// assert::close(median_2, 0.5, 1E-10);
    /// assert::close(kuma_2.cdf(&0.5), 0.5, 1E-10);
    /// assert::close(kuma_2.b(), 2.409420839653209, 1E-10);
    /// ```
    ///
    /// The PDF will most likely not be symmetrical about 0.5
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// # use rv::traits::{Rv, Cdf};
    /// fn absolute_error(a: f64, b: f64) -> f64 {
    ///     (a - b).abs()
    /// }
    ///
    /// let kuma = Kumaraswamy::centered(2.0).unwrap();
    /// assert!(absolute_error(kuma.f(&0.1), kuma.f(&0.9)) > 1E-8);
    /// ```
    #[inline]
    pub fn centered(a: f64) -> Result<Self, KumaraswamyError> {
        if a <= 0.0 {
            Err(KumaraswamyError::ATooLow { a })
        } else if !a.is_finite() {
            Err(KumaraswamyError::ANotFinite { a })
        } else {
            let b = 0.5_f64.log(1.0 - 0.5_f64.powf(a));
            Ok(Kumaraswamy {
                a,
                b,
                ab_ln: OnceCell::new(),
            })
        }
    }

    /// Get the `a` parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// let kuma = Kumaraswamy::new(1.0, 5.0).unwrap();
    /// assert_eq!(kuma.a(), 1.0);
    /// ```
    #[inline]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Get the `b` parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// let kuma = Kumaraswamy::new(1.0, 5.0).unwrap();
    /// assert_eq!(kuma.b(), 5.0);
    /// ```
    #[inline]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Set the value of the a parameter
    ///
    /// # Example
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// let mut kuma = Kumaraswamy::new(1.0, 5.0).unwrap();
    /// assert_eq!(kuma.a(), 1.0);
    ///
    /// kuma.set_a(2.3).unwrap();
    /// assert_eq!(kuma.a(), 2.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// # let mut kuma = Kumaraswamy::new(1.0, 5.0).unwrap();
    /// assert!(kuma.set_a(2.3).is_ok());
    /// assert!(kuma.set_a(0.0).is_err());
    /// assert!(kuma.set_a(std::f64::INFINITY).is_err());
    /// assert!(kuma.set_a(std::f64::NEG_INFINITY).is_err());
    /// assert!(kuma.set_a(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_a(&mut self, a: f64) -> Result<(), KumaraswamyError> {
        if a <= 0.0 {
            Err(KumaraswamyError::ATooLow { a })
        } else if !a.is_finite() {
            Err(KumaraswamyError::ANotFinite { a })
        } else {
            self.set_a_unchecked(a);
            Ok(())
        }
    }

    /// Set the value of the a parameter without input validation
    #[inline]
    pub fn set_a_unchecked(&mut self, a: f64) {
        self.a = a;
        self.ab_ln = OnceCell::new();
    }

    /// Set the value of the b parameter
    ///
    /// # Example
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// let mut kuma = Kumaraswamy::new(1.0, 5.0).unwrap();
    /// assert_eq!(kuma.b(), 5.0);
    ///
    /// kuma.set_b(2.3).unwrap();
    /// assert_eq!(kuma.b(), 2.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Kumaraswamy;
    /// # let mut kuma = Kumaraswamy::new(1.0, 5.0).unwrap();
    /// assert!(kuma.set_b(2.3).is_ok());
    /// assert!(kuma.set_b(0.0).is_err());
    /// assert!(kuma.set_b(std::f64::INFINITY).is_err());
    /// assert!(kuma.set_b(std::f64::NEG_INFINITY).is_err());
    /// assert!(kuma.set_b(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_b(&mut self, b: f64) -> Result<(), KumaraswamyError> {
        if b <= 0.0 {
            Err(KumaraswamyError::BTooLow { b })
        } else if !b.is_finite() {
            Err(KumaraswamyError::BNotFinite { b })
        } else {
            self.set_b_unchecked(b);
            Ok(())
        }
    }

    /// Set the value of the b parameter without input validation
    #[inline]
    pub fn set_b_unchecked(&mut self, b: f64) {
        self.b = b;
        self.ab_ln = OnceCell::new();
    }

    /// Evaluate or fetch cached ln(a*b)
    #[inline]
    fn ab_ln(&self) -> f64 {
        *self.ab_ln.get_or_init(|| self.a.ln() + self.b.ln())
    }
}

#[inline]
fn invcdf(p: f64, a: f64, b: f64) -> f64 {
    (1.0 - (1.0 - p).powf(b.recip())).powf(a.recip())
}

macro_rules! impl_kumaraswamy {
    ($kind: ty) => {
        impl Rv<$kind> for Kumaraswamy {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xf = *x as f64;
                let a = self.a;
                let b = self.b;
                (b - 1.0).mul_add(
                    (1.0 - xf.powf(a)).ln(),
                    (a - 1.0).mul_add(xf.ln(), self.ab_ln()),
                )
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let p: f64 = rng.gen();
                invcdf(p, self.a, self.b) as $kind
            }
        }

        impl Support<$kind> for Kumaraswamy {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite() && 0.0 < *x && *x < 1.0
            }
        }

        impl ContinuousDistr<$kind> for Kumaraswamy {}

        impl Cdf<$kind> for Kumaraswamy {
            fn cdf(&self, x: &$kind) -> f64 {
                1.0 - (1.0 - (*x as f64).powf(self.a)).powf(self.b)
            }
        }

        impl InverseCdf<$kind> for Kumaraswamy {
            fn invcdf(&self, p: f64) -> $kind {
                invcdf(p, self.a, self.b) as $kind
            }
        }

        impl Mean<$kind> for Kumaraswamy {
            fn mean(&self) -> Option<$kind> {
                let b = self.b;
                let ar1 = 1.0 + self.a.recip();
                let mean = b * ar1.gamma() * b.gamma() / (ar1 + b).gamma();
                Some(mean as $kind)
            }
        }

        impl Median<$kind> for Kumaraswamy {
            fn median(&self) -> Option<$kind> {
                let median =
                    (1.0 - (-self.b.recip()).exp2()).powf(self.a.recip());
                Some(median as $kind)
            }
        }

        impl Mode<$kind> for Kumaraswamy {
            fn mode(&self) -> Option<$kind> {
                if self.a < 1.0 || self.b < 1.0 {
                    None
                } else if (self.a - 1.0).abs() < f64::EPSILON
                    && (self.b - 1.0).abs() < f64::EPSILON
                {
                    None
                } else {
                    let mode = ((self.a - 1.0) / (self.a * self.b - 1.0))
                        .powf(self.a.recip());
                    Some(mode as $kind)
                }
            }
        }
    };
}

impl_kumaraswamy!(f64);
impl_kumaraswamy!(f32);

impl Entropy for Kumaraswamy {
    fn entropy(&self) -> f64 {
        // Harmonic function for reals see:
        // https://en.wikipedia.org/wiki/Harmonic_number#Harmonic_numbers_for_real_and_complex_values
        let hb = self.b.digamma() + EULER_MASCERONI;
        (1.0 - self.a.recip()).mul_add(hb, 1.0 - self.b.recip()) - self.ab_ln()
    }
}

impl std::error::Error for KumaraswamyError {}

impl fmt::Display for KumaraswamyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ATooLow { a } => {
                write!(f, "a ({}) must be greater than zero", a)
            }
            Self::ANotFinite { a } => write!(f, "non-finite a: {}", a),
            Self::BTooLow { b } => {
                write!(f, "b ({}) must be greater than zero", b)
            }
            Self::BNotFinite { b } => write!(f, "non-finite b: {}", b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{Beta, Gamma, LogNormal};
    use crate::misc::ks_test;
    use crate::misc::quad;
    use crate::test_basic_impls;

    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!([continuous] Kumaraswamy::centered(1.2).unwrap());

    #[test]
    fn cdf_uniform_midpoint() {
        let kuma = Kumaraswamy::uniform();
        assert::close(kuma.cdf(&0.5), 0.5, 1E-8);
    }

    #[test]
    fn draw_should_resturn_values_within_0_to_1() {
        let mut rng = rand::thread_rng();
        let kuma = Kumaraswamy::default();
        for _ in 0..100 {
            let x = kuma.draw(&mut rng);
            assert!(0.0 < x && x < 1.0);
        }
    }

    #[test]
    fn sample_returns_the_correct_number_draws() {
        let mut rng = rand::thread_rng();
        let kuma = Kumaraswamy::default();
        let xs: Vec<f64> = kuma.sample(103, &mut rng);
        assert_eq!(xs.len(), 103);
    }

    #[test]
    fn draws_from_correct_distribution() {
        let lognormal = LogNormal::new(0.0, 0.25).unwrap();
        let mut rng = rand::thread_rng();

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let a = lognormal.draw(&mut rng);
            let b = lognormal.draw(&mut rng);
            let kuma = Kumaraswamy::new(a, b).unwrap();

            let cdf = |x: f64| kuma.cdf(&x);
            let xs: Vec<f64> = kuma.sample(1000, &mut rng);
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
    fn pdf_quad_and_cdf_agree() {
        // create a Kumaraswamy distr with median at 0.5
        let kuma = Kumaraswamy::centered(2.0).unwrap();
        let intergral = quad(|x| kuma.f(&x), 0.0, 0.5);
        assert::close(intergral, 0.5, 1E-6);
    }

    #[test]
    fn median_for_centered_should_be_one_half() {
        fn k_centered(a: f64) -> impl Median<f64> {
            Kumaraswamy::centered(a).unwrap()
        }
        assert::close(k_centered(2.0).median().unwrap(), 0.5, 1E-10);
        assert::close(k_centered(1.0).median().unwrap(), 0.5, 1E-10);
        assert::close(k_centered(0.5).median().unwrap(), 0.5, 1E-10);
        assert::close(k_centered(1.2).median().unwrap(), 0.5, 1E-10);
        assert::close(k_centered(16.2).median().unwrap(), 0.5, 1E-10);
    }

    #[test]
    fn mean_for_uniform_should_be_one_half() {
        let mean: f64 = Kumaraswamy::uniform().mean().unwrap();
        assert::close(mean, 0.5, 1E-10)
    }

    #[test]
    fn equivalent_mean_to_beta_when_a_or_b_is_1() {
        let mut rng = rand::thread_rng();

        // K(a, 1) = B(a, 1) and K(1, b) = B(1, b)
        fn equiv(p: f64) {
            {
                let kuma_m: f64 =
                    Kumaraswamy::new(1.0, p).unwrap().mean().unwrap();
                let beta_m: f64 = Beta::new(1.0, p).unwrap().mean().unwrap();

                assert::close(kuma_m, beta_m, 1E-10)
            }
            {
                let kuma_m: f64 =
                    Kumaraswamy::new(p, 1.0).unwrap().mean().unwrap();
                let beta_m: f64 = Beta::new(p, 1.0).unwrap().mean().unwrap();

                assert::close(kuma_m, beta_m, 1E-10)
            }
        }

        Gamma::new(2.0, 2.0)
            .unwrap()
            .sample(100, &mut rng)
            .iter()
            .for_each(|&p| equiv(p))
    }

    #[test]
    fn equivalent_mode_to_beta_when_a_or_b_is_1() {
        let mut rng = rand::thread_rng();

        // K(a, 1) = B(a, 1) and K(1, b) = B(1, b)
        fn equiv(p: f64) {
            {
                let kuma_m: f64 =
                    Kumaraswamy::new(1.0, p).unwrap().mean().unwrap();
                let beta_m: f64 = Beta::new(1.0, p).unwrap().mean().unwrap();

                assert::close(kuma_m, beta_m, 1E-10)
            }
            {
                let kuma_m: f64 =
                    Kumaraswamy::new(p, 1.0).unwrap().mean().unwrap();
                let beta_m: f64 = Beta::new(p, 1.0).unwrap().mean().unwrap();

                assert::close(kuma_m, beta_m, 1E-10)
            }
        }

        // there is no mode if a or b < 1.0, so we'll add 1 to the parameter
        Gamma::new(2.0, 2.0)
            .unwrap()
            .sample(100, &mut rng)
            .iter()
            .for_each(|p: &f64| equiv(p + 1_f64))
    }

    #[test]
    fn no_mode_for_a_or_b_less_than_1() {
        fn mode(a: f64, b: f64) -> Option<f64> {
            Kumaraswamy::new(a, b).unwrap().mode()
        }
        assert!(mode(0.5, 2.0).is_none());
        assert!(mode(2.0, 0.99999).is_none());
        assert!(mode(1.0, 0.99999).is_none());
    }

    #[test]
    fn no_mode_for_a_and_b_equal_to_1() {
        let mode: Option<f64> = Kumaraswamy::new(1.0, 1.0).unwrap().mode();
        assert!(mode.is_none());
    }

    #[test]
    fn uniform_entropy_should_be_higheest() {
        // XXX: This doesn't test values
        let kuma_u = Kumaraswamy::uniform();
        let kuma_m = Kumaraswamy::centered(3.0).unwrap();

        assert!(kuma_u.entropy() > kuma_m.entropy());
    }

    #[test]
    fn set_a() {
        let mut kuma = Kumaraswamy::uniform();
        assert::close(kuma.pdf(&0.3), 1.0, 1E-10);

        kuma.set_a(2.0).unwrap();
        assert::close(
            kuma.pdf(&0.3),
            Beta::new(2.0, 1.0).unwrap().pdf(&0.3_f64),
            1E-10,
        );
    }

    #[test]
    fn set_b() {
        let mut kuma = Kumaraswamy::uniform();
        assert::close(kuma.pdf(&0.3), 1.0, 1E-10);

        kuma.set_b(2.0).unwrap();
        assert::close(
            kuma.pdf(&0.3),
            Beta::new(1.0, 2.0).unwrap().pdf(&0.3_f64),
            1E-10,
        );
    }

    #[test]
    #[cfg(feature = "serde1")]
    fn should_deserialize_without_ab_ln() {
        use indoc::indoc;

        let yaml = indoc!(
            "
            ---
            a: 2.0
            b: 3.0
            "
        );

        let kuma_1: Kumaraswamy = serde_yaml::from_str(&yaml).unwrap();
        let kuma_2 = Kumaraswamy::new(2.0, 3.0).unwrap();

        assert::close(kuma_1.f(&0.5_f64), kuma_2.f(&0.5_f64), 1E-12);
    }
}
