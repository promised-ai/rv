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
use serde_derive::{Deserialize, Serialize};

use crate::data::Partition;
use crate::impl_display;
use crate::misc::pflip;
use crate::traits::*;
use rand::Rng;
use special::Gamma as _;
use std::fmt;

/// [Chinese Restaurant Process](https://en.wikipedia.org/wiki/Chinese_restaurant_process),
/// a distribution over partitions.
///
/// # Example
///
/// ```
/// use::rv::prelude::*;
///
/// let mut rng = rand::thread_rng();
///
/// let crp = Crp::new(1.0, 10).expect("Invalid parameters");
/// let partition = crp.draw(&mut rng);
///
/// assert_eq!(partition.len(), 10);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Crp {
    /// Discount parameter
    alpha: f64,
    /// number of items in the partition
    n: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
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

    /// Create a new Crp without checking whether the parametes are valid.
    #[inline]
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
    /// assert!(crp.set_alpha(std::f64::INFINITY).is_err());
    /// assert!(crp.set_alpha(std::f64::NEG_INFINITY).is_err());
    /// assert!(crp.set_alpha(std::f64::NAN).is_err());
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

impl Rv<Partition> for Crp {
    fn ln_f(&self, x: &Partition) -> f64 {
        let gsum = x
            .counts()
            .iter()
            .fold(0.0, |acc, ct| acc + (*ct as f64).ln_gamma().0);

        // TODO: could cache ln(alpha) and ln_gamma(alpha)
        gsum + (x.k() as f64) * self.alpha.ln() + self.alpha.ln_gamma().0
            - (x.len() as f64 + self.alpha).ln_gamma().0
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> Partition {
        let mut k = 1;
        let mut weights: Vec<f64> = vec![1.0];
        let mut z: Vec<usize> = Vec::with_capacity(self.n);
        z.push(0);

        for _ in 1..self.n {
            weights.push(self.alpha);
            let zi = pflip(&weights, 1, rng)[0];
            z.push(zi);

            if zi == k {
                weights[zi] = 1.0;
                k += 1;
            } else {
                weights.truncate(k);
                weights[zi] += 1.0;
            }
        }
        // convert weights to counts, correcting for possible floating point
        // errors
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
                write!(f, "alpha ({}) must be greater than zero", alpha)
            }
            Self::AlphaNotFinite { alpha } => {
                write!(f, "alpha ({}) was non-finite", alpha)
            }
            Self::NIsZero => write!(f, "n must be greater than zero"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;

    test_basic_impls!(
        Crp::new(1.0, 10).unwrap(),
        Partition::new_unchecked(vec![0; 10], vec![10])
    );

    #[test]
    fn new() {
        let crp = Crp::new(1.2, 808).unwrap();
        assert::close(crp.alpha, 1.2, TOL);
        assert_eq!(crp.n, 808);
    }

    // TODO: More tests!
}
