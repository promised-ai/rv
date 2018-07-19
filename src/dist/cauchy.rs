//! Cauchy distribution over x in (-∞, ∞)
extern crate rand;

use self::rand::distributions::Cauchy as RCauchy;
use self::rand::Rng;
use consts::LN_PI;
use std::f64::consts::PI;
use std::io;
use traits::*;
use utils::logsumexp;

/// Cauchy distribution over x in (-∞, ∞)
///
/// # Example
/// ```
/// # extern crate rv;
/// use rv::prelude::*;
///
/// let cauchy = Cauchy::new(1.2, 3.4).expect("Invalid params");
/// let ln_fx = cauchy.ln_pdf(&0.2_f64); // -2.4514716152673368
///
/// assert!((ln_fx + 2.4514716152673368).abs() < 1E-12);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cauchy {
    /// location, x<sub>0</sub>, in (-∞, ∞)
    pub loc: f64,
    /// location, γ, in (0, ∞)
    pub scale: f64,
}

impl Cauchy {
    pub fn new(loc: f64, scale: f64) -> io::Result<Self> {
        let loc_ok = loc.is_finite();
        let scale_ok = scale > 0.0 && scale.is_finite();
        if loc_ok && scale_ok {
            Ok(Cauchy { loc, scale })
        } else if !loc_ok {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "loc must be finite");
            Err(err)
        } else {
            let err_kind = io::ErrorKind::InvalidInput;
            let msg = "scale must be finite and greater than zero";
            let err = io::Error::new(err_kind, msg);
            Err(err)
        }
    }
}

impl Default for Cauchy {
    fn default() -> Self {
        Cauchy::new(0.0, 1.0).unwrap()
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Cauchy {
            fn ln_f(&self, x: &$kind) -> f64 {
                let ln_scale = self.scale.ln();
                let term = ln_scale
                    + 2.0 * ((f64::from(*x) - self.loc).abs().ln() - ln_scale);
                // TODO: make a logaddexp method for two floats
                -logsumexp(&vec![ln_scale, term])
            }

            fn ln_normalizer(&self) -> f64 {
                LN_PI
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
            fn contains(&self, x: &$kind) -> bool {
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
    extern crate assert;
    use super::*;
    use std::f64;

    const TOL: f64 = 1E-12;

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
        assert!(c.contains(&0.0_f64));
        assert!(c.contains(&f64::MIN_POSITIVE));
        assert!(c.contains(&f64::MAX));
        assert!(c.contains(&f64::MIN));
    }

    #[test]
    fn non_finite_numbers_should_not_be_in_support() {
        let c = Cauchy::default();
        assert!(!c.contains(&f64::INFINITY));
        assert!(!c.contains(&f64::NEG_INFINITY));
        assert!(!c.contains(&f64::NAN));
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
}
