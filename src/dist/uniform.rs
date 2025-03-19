//! Continuous uniform distribution, U(a, b) on the interval x in [a, b]
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::traits::*;
use rand::Rng;
use std::f64;
use std::fmt;
use std::sync::OnceLock;

/// [Continuous uniform distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)),
/// U(a, b) on the interval x in [a, b]
///
/// # Example
///
/// The Uniform CDF is a line
///
/// ```
/// use rv::prelude::*;
///
/// let u = Uniform::new(2.0, 4.0).unwrap();
///
/// // A line representing the CDF
/// let y = |x: f64| { 0.5 * x - 1.0 };
///
/// assert!((u.cdf(&3.0_f64) - y(3.0)).abs() < 1E-12);
/// assert!((u.cdf(&3.2_f64) - y(3.2)).abs() < 1E-12);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Uniform {
    a: f64,
    b: f64,
    /// Cached value of the ln(PDF)
    #[cfg_attr(feature = "serde1", serde(skip))]
    lnf: OnceLock<f64>,
}

impl Shiftable for Uniform {
    type Output = Uniform;
    type Error = UniformError;

    fn shifted(self, dx: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Uniform::new(self.a() + dx, self.b() + dx)
    }

    fn shifted_unchecked(self, dx: f64) -> Self::Output
    where
        Self: Sized,
    {
        Uniform::new_unchecked(self.a() + dx, self.b() + dx)
    }
}

impl Parameterized for Uniform {
    type Parameters = (f64, f64);

    fn emit_params(&self) -> Self::Parameters {
        (self.a(), self.b())
    }

    fn from_params((a, b): Self::Parameters) -> Self {
        Self::new_unchecked(a, b)
    }
}

impl PartialEq for Uniform {
    fn eq(&self, other: &Uniform) -> bool {
        self.a == other.a && self.b == other.b
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum UniformError {
    /// A >= B
    InvalidInterval { a: f64, b: f64 },
    /// A was infinite or NaN
    ANotFinite { a: f64 },
    /// B was infinite or NaN
    BNotFinite { b: f64 },
}

impl Uniform {
    /// Create a new uniform distribution on [a, b]
    #[inline]
    pub fn new(a: f64, b: f64) -> Result<Self, UniformError> {
        if a >= b {
            Err(UniformError::InvalidInterval { a, b })
        } else if !a.is_finite() {
            Err(UniformError::ANotFinite { a })
        } else if !b.is_finite() {
            Err(UniformError::BNotFinite { b })
        } else {
            Ok(Uniform::new_unchecked(a, b))
        }
    }

    /// Creates a new Uniform without checking whether the parameters are
    /// valid.
    #[inline]
    pub fn new_unchecked(a: f64, b: f64) -> Self {
        Uniform {
            a,
            b,
            lnf: OnceLock::new(),
        }
    }

    /// Get the lower bound, a
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::Uniform;
    /// let u = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(u.a(), 0.0);
    /// ```
    #[inline]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Set the value of a
    pub fn set_a(&mut self, a: f64) -> Result<(), UniformError> {
        if !a.is_finite() {
            Err(UniformError::ANotFinite { a })
        } else if a >= self.b {
            Err(UniformError::InvalidInterval { a, b: self.b })
        } else {
            self.set_a_unchecked(a);
            Ok(())
        }
    }

    /// Set the value of a without checking if a is valid
    pub fn set_a_unchecked(&mut self, a: f64) {
        self.lnf = OnceLock::new();
        self.a = a
    }

    /// Get the upper bound, b
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::Uniform;
    /// let u = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(u.b(), 1.0);
    /// ```
    #[inline]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Set the value of b
    pub fn set_b(&mut self, b: f64) -> Result<(), UniformError> {
        if !b.is_finite() {
            Err(UniformError::BNotFinite { b })
        } else if self.a >= b {
            Err(UniformError::InvalidInterval { a: self.a, b })
        } else {
            self.set_b_unchecked(b);
            Ok(())
        }
    }

    /// Set the value of b without checking if b is valid
    pub fn set_b_unchecked(&mut self, b: f64) {
        self.lnf = OnceLock::new();
        self.b = b
    }

    #[inline]
    fn lnf(&self) -> f64 {
        *self.lnf.get_or_init(|| -(self.b - self.a).ln())
    }
}

impl Default for Uniform {
    fn default() -> Self {
        Uniform::new_unchecked(0.0, 1.0)
    }
}

impl From<&Uniform> for String {
    fn from(u: &Uniform) -> String {
        format!("U({}, {})", u.a, u.b)
    }
}

impl_display!(Uniform);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for Uniform {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xf = f64::from(*x);
                if self.a <= xf && xf <= self.b {
                    // call the lnf cache field
                    self.lnf()
                } else {
                    f64::NEG_INFINITY
                }
            }
        }

        impl Sampleable<$kind> for Uniform {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let u = rand_distr::Uniform::new(self.a, self.b);
                rng.sample(u) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let u = rand_distr::Uniform::new(self.a, self.b);
                (0..n).map(|_| rng.sample(u) as $kind).collect()
            }
        }

        #[allow(clippy::cmp_owned)]
        impl Support<$kind> for Uniform {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite()
                    && self.a <= f64::from(*x)
                    && f64::from(*x) <= self.b
            }
        }

        impl ContinuousDistr<$kind> for Uniform {}

        impl Mean<$kind> for Uniform {
            fn mean(&self) -> Option<$kind> {
                let m = (self.b + self.a) / 2.0;
                Some(m as $kind)
            }
        }

        impl Median<$kind> for Uniform {
            fn median(&self) -> Option<$kind> {
                let m = (self.b + self.a) / 2.0;
                Some(m as $kind)
            }
        }

        impl Variance<$kind> for Uniform {
            fn variance(&self) -> Option<$kind> {
                let diff = self.b - self.a;
                let v = diff * diff / 12.0;
                Some(v as $kind)
            }
        }

        impl Cdf<$kind> for Uniform {
            fn cdf(&self, x: &$kind) -> f64 {
                let xf = f64::from(*x);
                if xf < self.a {
                    0.0
                } else if xf >= self.b {
                    1.0
                } else {
                    (xf - self.a) / (self.b - self.a)
                }
            }
        }

        impl InverseCdf<$kind> for Uniform {
            fn invcdf(&self, p: f64) -> $kind {
                let x = p.mul_add(self.b - self.a, self.a);
                x as $kind
            }
        }
    };
}

impl Skewness for Uniform {
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl Kurtosis for Uniform {
    fn kurtosis(&self) -> Option<f64> {
        Some(-1.2)
    }
}

impl Entropy for Uniform {
    fn entropy(&self) -> f64 {
        (self.b - self.a).ln()
    }
}

impl_traits!(f64);
impl_traits!(f32);

impl std::error::Error for UniformError {}

impl fmt::Display for UniformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInterval { a, b } => {
                write!(f, "invalid interval: (a, b) = ({}, {})", a, b)
            }
            Self::ANotFinite { a } => write!(f, "non-finite a: {}", a),
            Self::BNotFinite { b } => write!(f, "non-finite b: {}", b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!(f64, Uniform);

    #[test]
    fn new() {
        let u = Uniform::new(0.0, 1.0).unwrap();
        assert::close(u.a, 0.0, TOL);
        assert::close(u.b, 1.0, TOL);
    }

    #[test]
    fn new_rejects_a_equal_to_b() {
        assert!(Uniform::new(1.0, 1.0).is_err());
    }

    #[test]
    fn new_rejects_a_gt_b() {
        assert!(Uniform::new(2.0, 1.0).is_err());
    }

    #[test]
    fn new_rejects_non_finite_a_or_b() {
        assert!(Uniform::new(f64::NEG_INFINITY, 1.0).is_err());
        assert!(Uniform::new(f64::NAN, 1.0).is_err());
        assert!(Uniform::new(0.0, f64::INFINITY).is_err());
        assert!(Uniform::new(0.0, f64::NAN).is_err());
    }

    #[test]
    fn mean() {
        let m: f64 = Uniform::new(2.0, 4.0).unwrap().mean().unwrap();
        assert::close(m, 3.0, TOL);
    }

    #[test]
    fn median() {
        let m: f64 = Uniform::new(2.0, 4.0).unwrap().median().unwrap();
        assert::close(m, 3.0, TOL);
    }

    #[test]
    fn variance() {
        let v: f64 = Uniform::new(2.0, 4.0).unwrap().variance().unwrap();
        assert::close(v, 2.0 / 6.0, TOL);
    }

    #[test]
    fn entropy() {
        let h: f64 = Uniform::new(2.0, 4.0).unwrap().entropy();
        assert::close(h, std::f64::consts::LN_2, TOL);
    }

    #[test]
    fn ln_pdf() {
        let u = Uniform::new(2.0, 4.0).unwrap();
        assert::close(u.ln_pdf(&2.0_f64), -std::f64::consts::LN_2, TOL);
        assert::close(u.ln_pdf(&2.3_f64), -std::f64::consts::LN_2, TOL);
        assert::close(u.ln_pdf(&3.3_f64), -std::f64::consts::LN_2, TOL);
        assert::close(u.ln_pdf(&4.0_f64), -std::f64::consts::LN_2, TOL);
    }

    #[test]
    fn cdf() {
        let u = Uniform::new(2.0, 4.0).unwrap();
        assert::close(u.cdf(&2.0_f64), 0.0, TOL);
        assert::close(u.cdf(&2.3_f64), 0.149_999_999_999_999_9, TOL);
        assert::close(u.cdf(&3.3_f64), 0.649_999_999_999_999_9, TOL);
        assert::close(u.cdf(&4.0_f64), 1.0, TOL);
    }

    #[test]
    fn cdf_inv_cdf_ident() {
        let mut rng = rand::thread_rng();
        let ru = rand::distributions::Uniform::new(1.2, 3.4);
        let u = Uniform::new(1.2, 3.4).unwrap();
        for _ in 0..100 {
            let x: f64 = rng.sample(ru);
            let cdf = u.cdf(&x);
            let y: f64 = u.invcdf(cdf);
            assert::close(x, y, 1E-8);
        }
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let u = Uniform::new(1.2, 3.4).unwrap();
        let cdf = |x: f64| u.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = u.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });
        assert!(passes > 0);
    }

    use crate::test_shiftable_cdf;
    use crate::test_shiftable_density;
    use crate::test_shiftable_entropy;
    use crate::test_shiftable_invcdf;
    use crate::test_shiftable_method;

    test_shiftable_method!(Uniform::new(2.0, 4.0).unwrap(), mean);
    test_shiftable_method!(Uniform::new(2.0, 4.0).unwrap(), median);
    test_shiftable_method!(Uniform::new(2.0, 4.0).unwrap(), variance);
    test_shiftable_method!(Uniform::new(2.0, 4.0).unwrap(), skewness);
    test_shiftable_method!(Uniform::new(2.0, 4.0).unwrap(), kurtosis);
    test_shiftable_density!(Uniform::new(2.0, 4.0).unwrap());
    test_shiftable_entropy!(Uniform::new(2.0, 4.0).unwrap());
    test_shiftable_cdf!(Uniform::new(2.0, 4.0).unwrap());
    test_shiftable_invcdf!(Uniform::new(2.0, 4.0).unwrap());
}
