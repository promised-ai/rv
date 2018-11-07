extern crate rand;
extern crate special;

use std::f64::consts::PI;
use std::f64::INFINITY;

use self::rand::distributions;
use self::rand::Rng;
use self::special::Gamma as SGamma;

use result;
use traits::*;

/// [Student's T distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
/// over x in (-∞, ∞).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct StudentsT {
    /// Degrees of freedom, ν, in (0, ∞)
    pub v: f64,
}

impl StudentsT {
    pub fn new(v: f64) -> result::Result<Self> {
        if v > 0.0 && v.is_finite() {
            Ok(StudentsT { v })
        } else {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "v must be finite and greater than 0";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        }
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for StudentsT {
            fn ln_f(&self, x: &$kind) -> f64 {
                let vp1 = (self.v + 1.0) / 2.0;
                let xterm = -vp1 * (1.0 + f64::from(*x).powi(2) / self.v).ln();
                let zterm = vp1.ln_gamma().0
                    - (self.v / 2.0).ln_gamma().0
                    - 0.5 * (self.v * PI).ln();
                zterm + xterm
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let t = distributions::StudentT::new(self.v);
                rng.sample(t) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let t = distributions::StudentT::new(self.v);
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
            Some(INFINITY)
        } else {
            None
        }
    }
}

impl_traits!(f64);
impl_traits!(f32);

#[cfg(test)]
mod tests {
    use super::*;
    extern crate assert;
    use std::f64;

    const TOL: f64 = 1E-12;

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
        assert::close(t.ln_pdf(&0.0_f64), -1.0247440238937566, TOL);
        assert::close(t.ln_pdf(&1.0_f64), -1.6204160440303521, TOL);
        assert::close(t.ln_pdf(&2.5_f64), -3.191230587916138, TOL);
        assert::close(t.ln_pdf(&-2.5_f64), -3.191230587916138, TOL);
    }

    #[test]
    fn variance() {
        let v: f64 = StudentsT::new(2.3).unwrap().variance().unwrap();
        assert::close(v, 7.6666666666666705, TOL);
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
