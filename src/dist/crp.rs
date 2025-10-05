//! Chinese Restaurant Process
//!
//! [The Chinese Restaurant Process](https://en.wikipedia.org/wiki/Chinese_restaurant_process) (CRP)
//! is a distribution over partitions of items. The CRP defines a process by
//! which entities are assigned to an unknown number of partition.
//!
//! The CRP is parameterized CRP(α) where α is the 'discount' parameter in
//! (0, ∞). Higher α causes there to be more partitions, as it encourages new
//! entries to create new partitions.
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::Partition;
use crate::impl_display;
use crate::misc::ln_gammafn;
use crate::misc::pflip;
use crate::traits::{HasDensity, Parameterized, Sampleable, Support};
use rand::Rng;
use std::fmt;

/// Parameters for the Chinese Restaurant Process distribution
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct CrpParameters {
    /// Discount parameter
    pub alpha: f64,
    /// number of items in the partition
    pub n: usize,
}

/// [Chinese Restaurant Process](https://en.wikipedia.org/wiki/Chinese_restaurant_process),
/// a distribution over partitions.
///
/// # Example
///
/// ```
/// use::rv::prelude::*;
///
/// let mut rng = rand::rng();
///
/// let crp = Crp::new(1.0, 10).expect("Invalid parameters");
/// let partition = crp.draw(&mut rng);
///
/// assert_eq!(partition.len(), 10);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Crp {
    /// Discount parameter
    alpha: f64,
    /// number of items in the partition
    n: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum CrpError {
    /// n parameter is zero
    NIsZero,
    /// alpha parameter is less than or equal to zero
    AlphaTooLow { alpha: f64 },
    /// alpha parameter is infinite or NaN
    AlphaNotFinite { alpha: f64 },
}

impl Crp {
    /// Create an empty `Crp` with parameter alpha
    ///
    /// # Arguments
    /// - alpha: Discount parameter in (0, Infinity)
    /// - n: the number of items in the partition
    pub fn new(alpha: f64, n: usize) -> Result<Self, CrpError> {
        if n == 0 {
            Err(CrpError::NIsZero)
        } else if alpha <= 0.0 {
            Err(CrpError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(CrpError::AlphaNotFinite { alpha })
        } else {
            Ok(Crp { alpha, n })
        }
    }

    /// Create a new Crp without checking whether the parameters are valid.
    ///
    /// ```rust
    /// use rv::dist::Crp;
    ///
    /// let crp = Crp::new_unchecked(3.0, 10);
    ///
    /// assert_eq!(crp.alpha(), 3.0);
    /// assert_eq!(crp.n(), 10);
    /// ```
    #[inline]
    #[must_use]
    pub fn new_unchecked(alpha: f64, n: usize) -> Self {
        Crp { alpha, n }
    }

    /// Get the discount parameter, `alpha`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Crp;
    /// let crp = Crp::new(1.0, 12).unwrap();
    /// assert_eq!(crp.alpha(), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set the value of alpha
    ///
    /// # Example
    /// ```rust
    /// # use rv::dist::Crp;
    /// let mut crp = Crp::new(1.1, 20).unwrap();
    /// assert_eq!(crp.alpha(), 1.1);
    ///
    /// crp.set_alpha(2.3).unwrap();
    /// assert_eq!(crp.alpha(), 2.3);
    /// ```
    ///
    /// Will error for invalid parameters
    ///
    /// ```rust
    /// # use rv::dist::Crp;
    /// # let mut crp = Crp::new(1.1, 20).unwrap();
    /// assert!(crp.set_alpha(0.5).is_ok());
    /// assert!(crp.set_alpha(0.0).is_err());
    /// assert!(crp.set_alpha(-1.0).is_err());
    /// assert!(crp.set_alpha(f64::INFINITY).is_err());
    /// assert!(crp.set_alpha(f64::NEG_INFINITY).is_err());
    /// assert!(crp.set_alpha(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_alpha(&mut self, alpha: f64) -> Result<(), CrpError> {
        if alpha <= 0.0 {
            Err(CrpError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(CrpError::AlphaNotFinite { alpha })
        } else {
            self.set_alpha_unchecked(alpha);
            Ok(())
        }
    }

    /// Set the value of alpha without input validation
    #[inline]
    pub fn set_alpha_unchecked(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    /// Get the number of entries in the partition, `n`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Crp;
    /// let crp = Crp::new(1.0, 12).unwrap();
    /// assert_eq!(crp.n(), 12);
    /// ```
    #[inline]
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Set the value of n
    ///
    /// # Example
    /// ```rust
    /// # use rv::dist::Crp;
    /// let mut crp = Crp::new(1.1, 20).unwrap();
    /// assert_eq!(crp.n(), 20);
    ///
    /// crp.set_n(11).unwrap();
    /// assert_eq!(crp.n(), 11);
    /// ```
    ///
    /// Will error for invalid parameters
    ///
    /// ```rust
    /// # use rv::dist::Crp;
    /// # let mut crp = Crp::new(1.1, 20).unwrap();
    /// assert!(crp.set_n(5).is_ok());
    /// assert!(crp.set_n(1).is_ok());
    /// assert!(crp.set_n(0).is_err());
    /// ```
    #[inline]
    pub fn set_n(&mut self, n: usize) -> Result<(), CrpError> {
        if n == 0 {
            Err(CrpError::NIsZero)
        } else {
            self.set_n_unchecked(n);
            Ok(())
        }
    }

    /// Set the value of alpha without input validation
    #[inline]
    pub fn set_n_unchecked(&mut self, n: usize) {
        self.n = n;
    }
}

impl From<&Crp> for String {
    fn from(crp: &Crp) -> String {
        format!("CRP({}; α: {})", crp.n, crp.alpha)
    }
}

impl_display!(Crp);

impl HasDensity<Partition> for Crp {
    fn ln_f(&self, x: &Partition) -> f64 {
        let gsum = x
            .counts()
            .iter()
            .fold(0.0, |acc, ct| acc + ln_gammafn(*ct as f64));

        // TODO: could cache ln(alpha) and ln_gamma(alpha)
        (x.k() as f64).mul_add(self.alpha.ln(), gsum) + ln_gammafn(self.alpha)
            - ln_gammafn(x.len() as f64 + self.alpha)
    }
}

impl Sampleable<Partition> for Crp {
    fn draw<R: Rng>(&self, rng: &mut R) -> Partition {
        let mut k = 1;
        // TODO: Set capacity according to
        // https://www.cs.princeton.edu/courses/archive/fall07/cos597C/scribe/20070921.pdf
        let mut weights: Vec<f64> = vec![1.0];
        let mut sum = 1.0 + self.alpha;
        let mut z: Vec<usize> = Vec::with_capacity(self.n);
        z.push(0);

        for _ in 1..self.n {
            weights.push(self.alpha);
            let zi = pflip(&weights, Some(sum), rng);
            z.push(zi);

            if zi == k {
                weights[zi] = 1.0;
                k += 1;
            } else {
                weights.truncate(k);
                weights[zi] += 1.0;
            }
            sum += 1.0;
        }
        // convert weights to counts, correcting for possible floating point
        // errors
        // TODO: Is this right? Wouldn't this be the _expected_ counts?
        let counts: Vec<usize> =
            weights.iter().map(|w| (w + 0.5) as usize).collect();

        Partition::new_unchecked(z, counts)
    }
}

impl Support<Partition> for Crp {
    #[inline]
    fn supports(&self, _x: &Partition) -> bool {
        true
    }
}

impl std::error::Error for CrpError {}

impl fmt::Display for CrpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlphaTooLow { alpha } => {
                write!(f, "alpha ({alpha}) must be greater than zero")
            }
            Self::AlphaNotFinite { alpha } => {
                write!(f, "alpha ({alpha}) was non-finite")
            }
            Self::NIsZero => write!(f, "n must be greater than zero"),
        }
    }
}

impl Parameterized for Crp {
    type Parameters = CrpParameters;

    fn emit_params(&self) -> Self::Parameters {
        CrpParameters {
            alpha: self.alpha,
            n: self.n,
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.alpha, params.n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn new() {
        let crp = Crp::new(1.2, 808).unwrap();
        assert::close(crp.alpha, 1.2, TOL);
        assert_eq!(crp.n, 808);
    }

    #[test]
    fn params() {
        let crp = Crp::new(1.2, 808).unwrap();
        let params = crp.emit_params();

        let new_crp = Crp::from_params(params);
        assert_eq!(crp, new_crp);
    }

    // TODO: More tests!
}
