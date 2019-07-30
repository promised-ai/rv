//! Cauchy distribution over x in (-∞, ∞)
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::consts::LN_PI;
use crate::impl_display;
use crate::misc::logsumexp;
use crate::result;
use crate::traits::*;
use getset::Setters;
use rand::distributions::Cauchy as RCauchy;
use rand::Rng;
use std::f64::consts::PI;

/// [Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
/// over x in (-∞, ∞).
///
/// # Example
/// ```
/// use rv::prelude::*;
///
/// let cauchy = Cauchy::new(1.2, 3.4).expect("Invalid params");
/// let ln_fx = cauchy.ln_pdf(&0.2_f64); // -2.4514716152673368
///
/// assert!((ln_fx + 2.4514716152673368).abs() < 1E-12);
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Setters)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Cauchy {
    /// location, x<sub>0</sub>, in (-∞, ∞)
    #[set = "pub"]
    loc: f64,
    /// scale, γ, in (0, ∞)
    #[set = "pub"]
    scale: f64,
}

impl Cauchy {
    /// Createa a new Cauchy distribution
    ///
    /// # Arguments
    /// - loc: location, x<sub>0</sub>, in (-∞, ∞)
    /// - scale: scale, γ, in (0, ∞)
    pub fn new(loc: f64, scale: f64) -> result::Result<Self> {
        let loc_ok = loc.is_finite();
        let scale_ok = scale > 0.0 && scale.is_finite();
        if loc_ok && scale_ok {
            Ok(Cauchy { loc, scale })
        } else if !loc_ok {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let err = result::Error::new(err_kind, "loc must be finite");
            Err(err)
        } else {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "scale must be finite and greater than zero";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        }
    }

    /// Get the location parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Cauchy;
    /// let c = Cauchy::new(0.1, 1.0).unwrap();
    /// assert_eq!(c.loc(), 0.1);
    /// ```
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// Get the scale parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Cauchy;
    /// let c = Cauchy::new(0.1, 1.0).unwrap();
    /// assert_eq!(c.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Default for Cauchy {
    fn default() -> Self {
        Cauchy::new(0.0, 1.0).unwrap()
    }
}

impl From<&Cauchy> for String {
    fn from(cauchy: &Cauchy) -> String {
        format!("Cauchy(loc: {}, scale: {})", cauchy.loc, cauchy.scale)
    }
}

impl_display!(Cauchy);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Cauchy {
            fn ln_f(&self, x: &$kind) -> f64 {
                let ln_scale = self.scale.ln();
                let term = ln_scale
                    + 2.0 * ((f64::from(*x) - self.loc).abs().ln() - ln_scale);
                // TODO: make a logaddexp method for two floats
                -logsumexp(&[ln_scale, term]) - LN_PI
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let cauchy = RCauchy::new(self.loc, self.scale);
                rng.sample(cauchy) as $kind
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let cauchy = RCauchy::new(self.loc, self.scale);
                (0..n).map(|_| rng.sample(cauchy) as $kind).collect()
            }
        }

        impl Support<$kind> for Cauchy {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite()
            }
        }

        impl ContinuousDistr<$kind> for Cauchy {}

        impl Cdf<$kind> for Cauchy {
            fn cdf(&self, x: &$kind) -> f64 {
                PI.recip() * ((f64::from(*x) - self.loc) / self.scale).atan()
                    + 0.5
            }
        }

        impl InverseCdf<$kind> for Cauchy {
            fn invcdf(&self, p: f64) -> $kind {
                (self.loc + self.scale * (PI * (p - 0.5)).tan()) as $kind
            }
        }

        impl Median<$kind> for Cauchy {
            fn median(&self) -> Option<$kind> {
                Some(self.loc as $kind)
            }
        }

        impl Mode<$kind> for Cauchy {
            fn mode(&self) -> Option<$kind> {
                Some(self.loc as $kind)
            }
        }
    };
}

impl Entropy for Cauchy {
    fn entropy(&self) -> f64 {
        (4.0 * PI * self.scale).ln()
    }
}

impl_traits!(f64);
impl_traits!(f32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;
    use std::f64;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    #[test]
    fn ln_pdf_loc_zero() {
        let c = Cauchy::new(0.0, 1.0).unwrap();
        assert::close(c.ln_pdf(&0.2), -1.1839505990026815, TOL);
    }

    #[test]
    fn ln_pdf_loc_nonzero() {
        let c = Cauchy::new(1.2, 3.4).unwrap();
        assert::close(c.ln_pdf(&0.2), -2.4514716152673368, TOL);
    }

    #[test]
    fn cdf_at_loc() {
        let c = Cauchy::new(1.2, 3.4).unwrap();
        assert::close(c.cdf(&1.2), 0.5, TOL);
    }

    #[test]
    fn cdf_off_loc() {
        let c = Cauchy::new(1.2, 3.4).unwrap();
        assert::close(c.cdf(&2.2), 0.59105300185574883, TOL);
        assert::close(c.cdf(&0.2), 1.0 - 0.59105300185574883, TOL);
    }

    #[test]
    fn inv_cdf_ident() {
        let mut rng = rand::thread_rng();
        let c = Cauchy::default();
        for _ in 0..100 {
            let x: f64 = c.draw(&mut rng);
            let p: f64 = c.cdf(&x);
            let y: f64 = c.invcdf(p);
            assert::close(x, y, 1E-8); // allow a little more error
        }
    }

    #[test]
    fn inv_cdf() {
        let c = Cauchy::new(1.2, 3.4).unwrap();
        let lower: f64 = c.invcdf(0.4);
        let upper: f64 = c.invcdf(0.6);
        assert::close(lower, 0.095273032808118607, TOL);
        assert::close(upper, 2.3047269671918813, TOL);
    }

    #[test]
    fn median_should_be_loc() {
        let m: f64 = Cauchy::new(1.2, 3.4).unwrap().median().unwrap();
        assert::close(m, 1.2, TOL);
    }

    #[test]
    fn mode_should_be_loc() {
        let m: f64 = Cauchy::new(-0.2, 3.4).unwrap().median().unwrap();
        assert::close(m, -0.2, TOL);
    }

    #[test]
    fn finite_numbers_should_be_in_support() {
        let c = Cauchy::default();
        assert!(c.supports(&0.0_f64));
        assert!(c.supports(&f64::MIN_POSITIVE));
        assert!(c.supports(&f64::MAX));
        assert!(c.supports(&f64::MIN));
    }

    #[test]
    fn non_finite_numbers_should_not_be_in_support() {
        let c = Cauchy::default();
        assert!(!c.supports(&f64::INFINITY));
        assert!(!c.supports(&f64::NEG_INFINITY));
        assert!(!c.supports(&f64::NAN));
    }

    #[test]
    fn entropy() {
        let c = Cauchy::new(1.2, 3.4).unwrap();
        assert::close(c.entropy(), 3.7547996785914064, TOL);
    }

    #[test]
    fn loc_does_not_affect_entropy() {
        let c1 = Cauchy::new(1.2, 3.4).unwrap();
        let c2 = Cauchy::new(-99999.9, 3.4).unwrap();
        assert::close(c1.entropy(), c2.entropy(), TOL);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let c = Cauchy::new(1.2, 3.4).unwrap();
        let cdf = |x: f64| c.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = c.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }
}
