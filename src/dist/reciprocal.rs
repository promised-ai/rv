use rand::prelude::Rng;

use crate::prelude::{Cdf, InverseCdf, Mean, Median, Rv, Support, Variance};

/// The Reciprocal Distribution
///
/// # Probability Density Function
/// The PDF for the `Reciprocal` distribution is
///
/// ```math
/// f(x; a, b) = \frac{1}{x \left[ \ln(b) - \ln(a) \right]}.
/// ```
///
#[derive(Clone, Copy, Debug)]
pub struct Reciprocal {
    /// Left bound of the reciprocal distribution.
    a: f64,
    /// Right bound of the reciprocal distribution.
    b: f64,
    ln_b_a: f64,
}

impl Reciprocal {
    /// Create a new `Reciprocal` Distribution with bounds `a` and `b`.
    ///
    /// # Errors
    /// * `ReciprocalError::ANotLessThanB` - Parameter `a` must be less than `b`.
    /// * `ReciprocalError::ANotGreaterThanZero` - Parameter `a` must be greater than 0.
    pub fn new(a: f64, b: f64) -> Result<Self, ReciprocalError> {
        if a <= 0.0 {
            Err(ReciprocalError::ANotLessThanB)
        } else if a > b {
            Err(ReciprocalError::ANotGreaterThanZero)
        } else {
            let ln_b_a = b.ln() - a.ln();
            Ok(Self { a, b, ln_b_a })
        }
    }

    /// Create a new `Reciprocal` Distribution with bounds `a` and `b` without checking for
    /// erroneous parameters.
    pub fn new_unchecked(a: f64, b: f64) -> Self {
        let ln_b_a = b.ln() - a.ln();
        Self { a, b, ln_b_a }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ReciprocalError {
    /// `a` must be greater than 0.
    ANotGreaterThanZero,
    /// Upon Creation, `a` must be less than `b`.
    ANotLessThanB,
}

macro_rules! impl_rv {
    ($X: ty) => {
        impl Rv<$X> for Reciprocal {
            fn ln_f(&self, x: &$X) -> f64 {
                let xf = f64::from(*x);
                if xf < self.a || xf > self.b {
                    f64::NEG_INFINITY
                } else {
                    -(xf * self.ln_b_a).ln()
                }
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $X {
                (self.a * (self.b / self.a).powf(rng.gen())) as $X
            }
        }

        impl Cdf<$X> for Reciprocal {
            fn cdf(&self, x: &$X) -> f64 {
                let xf = f64::from(*x);
                if xf < self.a {
                    0.0
                } else if xf > self.b {
                    1.0
                } else {
                    ((xf / self.a).ln()) / self.ln_b_a
                }
            }
        }

        impl InverseCdf<$X> for Reciprocal {
            fn invcdf(&self, p: f64) -> $X {
                (self.a * (self.b / self.a).powf(p)) as $X
            }
        }

        impl Support<$X> for Reciprocal {
            fn supports(&self, x: &$X) -> bool {
                let xf = f64::from(*x);
                xf >= self.a && xf <= self.b
            }
        }

        impl Mean<$X> for Reciprocal {
            fn mean(&self) -> Option<$X> {
                Some(((self.b - self.a) / self.ln_b_a) as $X)
            }
        }

        impl Median<$X> for Reciprocal {
            fn median(&self) -> Option<$X> {
                Some((self.a * self.b).sqrt() as $X)
            }
        }

        impl Variance<$X> for Reciprocal {
            fn variance(&self) -> Option<$X> {
                Some(
                    ((self.b.powi(2) - self.a.powi(2)) / (2.0 * self.ln_b_a)
                        - ((self.b - self.a) / self.ln_b_a).powi(2))
                        as $X,
                )
            }
        }
    };
}

impl_rv!(f64);
impl_rv!(f32);
