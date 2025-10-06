use crate::{
    impl_scalable, impl_shiftable,
    traits::{
        Cdf, ContinuousDistr, HasDensity, Mean, Median, Mode, Sampleable,
        Scalable, Shiftable, Support,
    },
};

/// A Dirac delta distribution
///
/// This is not a traditional distribution but rather a helpful device, as a dirac delta will
/// always draw a value of 0 and has a cdf characterised by the Heaviside function centered at zero.
///
/// # Example
/// ```rust
/// use rv::prelude::*;
/// use rv::dist::Delta;
/// use rand;
///
/// let mut rng = rand::rng();
/// let x: f64 = Delta.draw(&mut rng);
///
/// assert_eq!(x, 0.0);
/// assert_eq!(Delta.cdf(&-1.0), 0.0);
/// assert_eq!(Delta.cdf(&1.0), 1.0);
/// ```
pub struct Delta;

macro_rules! trait_impls {
    ($t: ty) => {
        impl Sampleable<$t> for Delta {
            fn draw<R: rand::Rng>(&self, _rng: &mut R) -> $t {
                0.0
            }
        }

        impl Cdf<$t> for Delta {
            fn cdf(&self, x: &$t) -> f64 {
                if x < &0.0 { 0.0 } else { 1.0 }
            }
        }

        impl HasDensity<$t> for Delta {
            fn ln_f(&self, x: &$t) -> f64 {
                if x == &0.0 { f64::INFINITY } else { 0.0 }
            }
        }

        impl Support<$t> for Delta {
            fn supports(&self, _x: &$t) -> bool {
                true
            }
        }

        impl ContinuousDistr<$t> for Delta {}

        impl Mean<$t> for Delta {
            fn mean(&self) -> Option<$t> {
                Some(0.0)
            }
        }

        impl Mode<$t> for Delta {
            fn mode(&self) -> Option<$t> {
                Some(0.0)
            }
        }

        impl Median<$t> for Delta {
            fn median(&self) -> Option<$t> {
                Some(0.0)
            }
        }
    };
}

trait_impls!(f64);
trait_impls!(f32);

impl_shiftable!(Delta);
impl_scalable!(Delta);
