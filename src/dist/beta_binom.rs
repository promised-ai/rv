//! Beta Binomial distribution of x in {0, ..., n}
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::impl_display;
use crate::misc::{ln_binom, ln_pflip};
use crate::traits::*;
use getset::Setters;
use rand::Rng;
use special::Beta as _;
use std::f64;

/// [Beta Binomial distribution](https://en.wikipedia.org/wiki/Beta-binomial_distribution)
/// over k in {0, ..., n}
///
/// # Example
///
/// ```
/// use std::f64;
/// use rv::prelude::*;
///
/// let a = 3.0;
/// let b = 2.0;
/// let n = 20;
///
/// let beta = Beta::new(a, b).unwrap();
/// let beta_binom = BetaBinomial::new(n, a, b).unwrap();
///
/// let beta_mean: f64 = beta.mean().unwrap();
/// let beta_binom_mean: f64 = beta_binom.mean().unwrap();
/// assert!( (beta_mean * f64::from(n) - beta_binom_mean).abs() < 1E-12 );
/// ```
///
/// Some functions will panic when given data outside the supported range:
/// [0, n]
///
/// ```
/// # use rv::prelude::*;
/// let beta_binom = BetaBinomial::new(20, 3.0, 2.0).unwrap();
/// assert!(!beta_binom.supports(&21_u32));
/// ```
///
/// The following will panic because 21 is out of the support
///
/// ```should_panic
/// # use rv::prelude::*;
/// # let beta_binom = BetaBinomial::new(20, 3.0, 2.0).unwrap();
/// beta_binom.pmf(&21_u32); // panics
/// ```

#[derive(Debug, Clone, PartialEq, PartialOrd, Setters)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct BetaBinomial {
    /// Total number of trials
    #[set = "pub"]
    n: u32,
    /// Analogous to Beta Distribution α parameter.
    #[set = "pub"]
    alpha: f64,
    /// Analogous to Beta Distribution β parameter
    #[set = "pub"]
    beta: f64,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum BetaBinomialError {
    /// The alpha parameter is less than zero
    AlphaLessThanZeroError,
    /// The alpha parameter is infinite or NaN
    AlphaNotFiniteError,
    /// The beta parameter is less than zero
    BetaLessThanZeroError,
    /// The beta parameter is infinite or NaN
    BetaNotFiniteError,
    /// The number of trails is zero
    NIsZeroError,
}

impl BetaBinomial {
    /// Create a beta-binomal distirbution
    ///
    /// # Arguments
    ///
    /// - n: the total number of trials
    /// - alpha: the prior pseudo obersvations of success
    /// - beta: the prior pseudo obersvations of failure
    pub fn new(
        n: u32,
        alpha: f64,
        beta: f64,
    ) -> Result<Self, BetaBinomialError> {
        if alpha < 0.0 {
            Err(BetaBinomialError::AlphaLessThanZeroError)
        } else if !alpha.is_finite() {
            Err(BetaBinomialError::AlphaNotFiniteError)
        } else if beta < 0.0 {
            Err(BetaBinomialError::BetaLessThanZeroError)
        } else if !beta.is_finite() {
            Err(BetaBinomialError::BetaNotFiniteError)
        } else if n == 0 {
            Err(BetaBinomialError::NIsZeroError)
        } else {
            Ok(BetaBinomial { n, alpha, beta })
        }
    }

    /// Creates a new BetaBinomial without checking whether the parameters are
    /// valid.
    pub fn new_unchecked(n: u32, alpha: f64, beta: f64) -> Self {
        BetaBinomial { n, alpha, beta }
    }

    /// Get `n`, the number of trials.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::BetaBinomial;
    /// let bb = BetaBinomial::new(10, 1.0, 2.0).unwrap();
    /// assert_eq!(bb.n(), 10);
    /// ```
    pub fn n(&self) -> u32 {
        self.n
    }

    /// Get the `alpha` parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::BetaBinomial;
    /// let bb = BetaBinomial::new(10, 1.0, 2.0).unwrap();
    /// assert_eq!(bb.alpha(), 1.0);
    /// ```
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the `beta` parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::BetaBinomial;
    /// let bb = BetaBinomial::new(10, 1.0, 2.0).unwrap();
    /// assert_eq!(bb.beta(), 2.0);
    /// ```
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl From<&BetaBinomial> for String {
    fn from(bb: &BetaBinomial) -> String {
        format!("BetaBinomial({}; α: {}, β: {})", bb.n, bb.alpha, bb.beta)
    }
}

impl_display!(BetaBinomial);

macro_rules! impl_int_traits {
    ($kind:ty) => {
        impl Rv<$kind> for BetaBinomial {
            fn ln_f(&self, k: &$kind) -> f64 {
                let nf = f64::from(self.n);
                let kf = *k as f64;
                ln_binom(nf, kf)
                    + (kf + self.alpha).ln_beta(nf - kf + self.beta)
                    - self.alpha.ln_beta(self.beta)
            }

            fn draw<R: Rng>(&self, mut rng: &mut R) -> $kind {
                self.sample(1, &mut rng)[0]
            }

            fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<$kind> {
                // TODO: Could speed this up if we didn't compute the
                // k-independent terms in ln_f
                let ln_weights: Vec<f64> =
                    (0..=self.n).map(|x| self.ln_f(&x)).collect();

                ln_pflip(&ln_weights, n, true, &mut rng)
                    .iter()
                    .map(|k| *k as $kind)
                    .collect()
            }
        }

        impl Support<$kind> for BetaBinomial {
            #[allow(unused_comparisons)]
            fn supports(&self, k: &$kind) -> bool {
                *k >= 0 && *k <= self.n as $kind
            }
        }

        impl DiscreteDistr<$kind> for BetaBinomial {}

        impl Cdf<$kind> for BetaBinomial {
            fn cdf(&self, k: &$kind) -> f64 {
                // XXX: Slow and awful.
                // TODO: could make this faster with hypergeometric function,
                // but the `special` crate doesn't implement it...yet (take
                // the hint).
                (0..=*k).fold(0.0, |acc, x| acc + self.pmf(&x))
            }
        }
    };
}

impl Mean<f64> for BetaBinomial {
    fn mean(&self) -> Option<f64> {
        let nf = f64::from(self.n);
        let m = self.alpha / (self.alpha + self.beta);
        Some(nf * m)
    }
}

impl Variance<f64> for BetaBinomial {
    fn variance(&self) -> Option<f64> {
        let nf = f64::from(self.n);
        let apb = self.alpha + self.beta;
        let v_numer = nf * self.alpha * self.beta * (apb + nf);
        let v_denom = apb * apb * (apb + 1.0);
        Some(v_numer / v_denom)
    }
}

impl_int_traits!(u8);
impl_int_traits!(u16);
impl_int_traits!(u32);
impl_int_traits!(u64);
impl_int_traits!(usize);

impl_int_traits!(i8);
impl_int_traits!(i16);
impl_int_traits!(i32);
impl_int_traits!(i64);

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;

    const TOL: f64 = 1E-12;

    #[test]
    fn new() {
        let beta_binom = BetaBinomial::new(10, 0.1, 0.2).unwrap();
        assert_eq!(beta_binom.n, 10);
        assert::close(beta_binom.alpha(), 0.1, TOL);
        assert::close(beta_binom.beta(), 0.2, TOL);
    }

    #[test]
    fn pmf() {
        let beta_binom = BetaBinomial::new(10, 0.5, 2.0).unwrap();
        // Values from wolfram alpha
        let target = vec![
            0.387765, 0.176257, 0.118973, 0.0881283, 0.0674732, 0.0520508,
            0.039761, 0.0295368, 0.020768, 0.0130762, 0.00621118,
        ];
        let pmfs: Vec<f64> = (0..=10).map(|k| beta_binom.pmf(&k)).collect();
        assert::close(pmfs, target, 1E-6);
    }
}
