use crate::dist::Poisson;
use crate::misc::ln_binom;
use crate::traits::*;
use once_cell::sync::OnceCell;
use rand::Rng;

#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

/// Negative Binomial distribution errors
pub enum NegBinomialError {
    /// The probability parameter, p, is not in [0, 1]
    POutOfRangeError,
    /// R is less that 1.0
    RLessThanOneError,
}

/// Negative Binomial distribution
pub struct NegBinomial {
    r: f64,
    p: f64,
    // ln((1-p)^r)
    r_ln_1mp: OnceCell<f64>,
    // ln(p)
    ln_p: OnceCell<f64>,
}

impl NegBinomial {
    pub fn new(r: f64, p: f64) -> Result<Self, NegBinomialError> {
        if r < 1.0 {
            Err(NegBinomialError::RLessThanOneError)
        } else if 1.0 < p || p < 0.0 {
            Err(NegBinomialError::POutOfRangeError)
        } else {
            Ok(Self::new_unchecked(r, p))
        }
    }

    pub fn new_unchecked(r: f64, p: f64) -> Self {
        NegBinomial {
            r,
            p,
            r_ln_1mp: OnceCell::new(),
            ln_p: OnceCell::new(),
        }
    }

    #[inline]
    pub fn r(&self) -> f64 {
        self.r
    }

    #[inline]
    pub fn set_r(&mut self, r: f64) -> Result<(), NegBinomialError> {
        if r < 1.0 {
            Err(NegBinomialError::RLessThanOneError)
        } else {
            self.r = r;
            Ok(())
        }
    }

    #[inline]
    pub fn p(&self) -> f64 {
        self.p
    }

    #[inline]
    pub fn set_p(&mut self, p: f64) -> Result<(), NegBinomialError> {
        if 1.0 < p || p < 0.0 {
            Err(NegBinomialError::POutOfRangeError)
        } else {
            self.p = p;
            Ok(())
        }
    }

    #[inline]
    fn r_ln_1mp(&self) -> f64 {
        *self.r_ln_1mp.get_or_init(|| self.r * (1.0 - self.p).ln())
    }

    #[inline]
    fn ln_p(&self) -> f64 {
        *self.ln_p.get_or_init(|| self.p.ln())
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for NegBinomial {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xf = (*x) as f64;
                ln_binom(xf + self.r - 1.0, xf)
                    + self.r_ln_1mp()
                    + xf * self.ln_p()
            }

            fn draw<R: Rng>(&self, mut rng: &mut R) -> $kind {
                let scale = self.p / (1.0 - self.p);
                let gamma = rand_distr::Gamma::new(self.r, scale).unwrap();
                let pois_rate = rng.sample(gamma);
                Poisson::new_unchecked(pois_rate).draw(&mut rng)
            }

            fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<$kind> {
                let scale = self.p / (1.0 - self.p);
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
            fn supports(&self, x: &$kind) -> bool {
                *x > 1
            }
        }

        impl Mode<$kind> for NegBinomial {
            fn mode(&self) -> Option<$kind> {
                if self.r <= 1.0 {
                    Some(0)
                } else {
                    let m = self.p * (self.r - 1.0) / (1.0 - self.p);
                    // take the floor through integer conversion
                    Some(m as $kind)
                }
            }
        }
    };
}

impl Mean<f64> for NegBinomial {
    fn mean(&self) -> Option<f64> {
        Some((self.p * self.r) / (1.0 - self.p))
    }
}

impl Variance<f64> for NegBinomial {
    fn variance(&self) -> Option<f64> {
        Some((self.p * self.r) / (1.0 - self.p).powi(2))
    }
}

impl Skewness for NegBinomial {
    fn skewness(&self) -> Option<f64> {
        Some((1.0 + self.p) / (self.p * self.r).sqrt())
    }
}

impl Kurtosis for NegBinomial {
    fn kurtosis(&self) -> Option<f64> {
        Some(6.0 / self.r + (1.0 - self.p).powi(2) / (self.p * self.r))
    }
}

impl_traits!(u8);
impl_traits!(u16);
impl_traits!(u32);
