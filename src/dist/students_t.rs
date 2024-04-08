#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::misc::ln_gammafn;
use crate::traits::*;
use rand::Rng;
use std::f64::consts::PI;
use std::fmt;

/// [Student's T distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
/// over x in (-∞, ∞).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct StudentsT {
    /// Degrees of freedom, ν, in (0, ∞)
    v: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum StudentsTError {
    /// The v parameter is infinite or NaN
    VNotFinite { v: f64 },
    /// The v parameter is less than or equal to zero
    VTooLow { v: f64 },
}

impl StudentsT {
    /// Create a new Student's T distribtuion with degrees of freedom, v.
    #[inline]
    pub fn new(v: f64) -> Result<Self, StudentsTError> {
        if v <= 0.0 {
            Err(StudentsTError::VTooLow { v })
        } else if !v.is_finite() {
            Err(StudentsTError::VNotFinite { v })
        } else {
            Ok(StudentsT { v })
        }
    }

    /// Creates a new StudentsT without checking whether the parameter is
    /// valid.
    #[inline]
    pub fn new_unchecked(v: f64) -> Self {
        StudentsT { v }
    }

    /// Get the degrees of freedom, v
    #[inline]
    pub fn v(&self) -> f64 {
        self.v
    }

    /// Set the value of v
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::StudentsT;
    ///
    /// let mut t = StudentsT::new(1.2).unwrap();
    /// assert_eq!(t.v(), 1.2);
    ///
    /// t.set_v(4.3).unwrap();
    /// assert_eq!(t.v(), 4.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::StudentsT;
    /// # let mut t = StudentsT::new(1.2).unwrap();
    /// assert!(t.set_v(2.1).is_ok());
    ///
    /// // must be greater than zero
    /// assert!(t.set_v(0.0).is_err());
    /// assert!(t.set_v(-1.0).is_err());
    ///
    ///
    /// assert!(t.set_v(std::f64::INFINITY).is_err());
    /// assert!(t.set_v(f64::NEG_INFINITY).is_err());
    /// assert!(t.set_v(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_v(&mut self, v: f64) -> Result<(), StudentsTError> {
        if !v.is_finite() {
            Err(StudentsTError::VNotFinite { v })
        } else if v <= 0.0 {
            Err(StudentsTError::VTooLow { v })
        } else {
            self.set_v_unchecked(v);
            Ok(())
        }
    }

    /// Set the value of v without input validation
    #[inline]
    pub fn set_v_unchecked(&mut self, v: f64) {
        self.v = v;
    }
}

impl Default for StudentsT {
    fn default() -> Self {
        StudentsT { v: 2.0 }
    }
}

impl From<&StudentsT> for String {
    fn from(t: &StudentsT) -> String {
        format!("Student's({})", t.v)
    }
}

impl_display!(StudentsT);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for StudentsT {
            fn ln_f(&self, x: &$kind) -> f64 {
                // TODO: could cache ln(pi*v) and ln_gamma(v/2)
                let vp1 = (self.v + 1.0) / 2.0;
                let xf = f64::from(*x);
                let xterm = -vp1 * (xf * xf / self.v).ln_1p();
                let zterm = 0.5_f64.mul_add(
                    -(self.v * PI).ln(),
                    ln_gammafn(vp1) - ln_gammafn(self.v / 2.0),
                );
                zterm + xterm
            }
        }

        impl Sampleable<$kind> for StudentsT {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let t = rand_distr::StudentT::new(self.v).unwrap();
                rng.sample(t) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let t = rand_distr::StudentT::new(self.v).unwrap();
                (0..n).map(|_| rng.sample(t) as $kind).collect()
            }
        }

        impl Support<$kind> for StudentsT {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite()
            }
        }

        impl ContinuousDistr<$kind> for StudentsT {}

        impl Mean<$kind> for StudentsT {
            fn mean(&self) -> Option<$kind> {
                if self.v > 1.0 {
                    Some(0.0)
                } else {
                    None
                }
            }
        }

        impl Median<$kind> for StudentsT {
            fn median(&self) -> Option<$kind> {
                Some(0.0)
            }
        }

        impl Mode<$kind> for StudentsT {
            fn mode(&self) -> Option<$kind> {
                Some(0.0)
            }
        }

        impl Variance<$kind> for StudentsT {
            fn variance(&self) -> Option<$kind> {
                if self.v > 2.0 {
                    Some((self.v / (self.v - 2.0)) as $kind)
                } else {
                    None
                }
            }
        }
    };
}

impl Skewness for StudentsT {
    fn skewness(&self) -> Option<f64> {
        if self.v > 3.0 {
            Some(0.0)
        } else {
            None
        }
    }
}

impl Kurtosis for StudentsT {
    fn kurtosis(&self) -> Option<f64> {
        if self.v > 4.0 {
            Some(6.0 / (self.v - 4.0))
        } else if self.v > 2.0 {
            Some(f64::INFINITY)
        } else {
            None
        }
    }
}

impl_traits!(f64);
impl_traits!(f32);

impl std::error::Error for StudentsTError {}

impl fmt::Display for StudentsTError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VNotFinite { v } => write!(f, "non-finite v: {}", v),
            Self::VTooLow { v } => {
                write!(f, "v ({}) must be greater than zero", v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_basic_impls;
    use std::f64;

    const TOL: f64 = 1E-12;

    test_basic_impls!([continuous] StudentsT::default());

    #[test]
    fn new() {
        let t = StudentsT::new(2.3).unwrap();
        assert::close(t.v, 2.3, TOL);
    }

    #[test]
    fn new_should_reject_v_leq_zero() {
        assert!(StudentsT::new(f64::MIN_POSITIVE).is_ok());
        assert!(StudentsT::new(0.0).is_err());
        assert!(StudentsT::new(-f64::MIN_POSITIVE).is_err());
        assert!(StudentsT::new(-1.0).is_err());
    }

    #[test]
    fn new_should_reject_non_finite_v() {
        assert!(StudentsT::new(f64::INFINITY).is_err());
        assert!(StudentsT::new(-f64::NAN).is_err());
        assert!(StudentsT::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn ln_pdf() {
        let t = StudentsT::new(2.3).unwrap();
        assert::close(t.ln_pdf(&0.0_f64), -1.024_744_023_893_756_6, TOL);
        assert::close(t.ln_pdf(&1.0_f64), -1.620_416_044_030_352, TOL);
        assert::close(t.ln_pdf(&2.5_f64), -3.191_230_587_916_138, TOL);
        assert::close(t.ln_pdf(&-2.5_f64), -3.191_230_587_916_138, TOL);
    }

    #[test]
    fn variance() {
        let v: f64 = StudentsT::new(2.3).unwrap().variance().unwrap();
        assert::close(v, 7.666_666_666_666_670_5, TOL);
    }

    #[test]
    fn median() {
        let m: f64 = StudentsT::new(2.3).unwrap().median().unwrap();
        assert::close(m, 0.0, TOL);
    }

    #[test]
    fn mode() {
        let m: f64 = StudentsT::new(2.3).unwrap().mode().unwrap();
        assert::close(m, 0.0, TOL);
    }
}
