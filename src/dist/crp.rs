//! Chinese Restaurant Process
//!
//! [The Chinese Restaurant Process](https://en.wikipedia.org/wiki/Chinese_restaurant_process) (CRP)
//! is a distribution over partitions of items. The CRP defines a process by
//! which entities are assigned to an unknown number of partition.
//!
//! The CRP is parameterized CRP(α) where α is the 'discount' parameter in
//! (0, ∞). Higher α causes there to be more partitions, as it encourages new
//! entries to create new partitions.
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::data::Partition;
use crate::impl_display;
use crate::misc::pflip;
use crate::result;
use crate::traits::*;
use getset::Setters;
use rand::Rng;
use special::Gamma as _;

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
#[derive(Debug, Clone, Setters)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Crp {
    /// Discount parameter
    #[set = "pub"]
    alpha: f64,
    /// number of items in the partition
    #[set = "pub"]
    n: usize,
}

impl Crp {
    /// Create an empty `Crp` with parameter alpha
    ///
    /// # Arguments
    /// - alpha: Discount parameter in (0, Infinity)
    /// - n: the number of items in the partition
    pub fn new(alpha: f64, n: usize) -> result::Result<Self> {
        let alpha_ok = alpha > 0.0 && alpha.is_finite();
        let n_ok = n > 0;
        if !alpha_ok {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "α must be greater than zero and finite";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        } else if !n_ok {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "n must be greater than zero";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        } else {
            Ok(Crp { alpha, n })
        }
    }

    /// Create a new Crp without checking whether the parametes are valid.
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
    pub fn alpha(&self) -> f64 {
        self.alpha
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
    pub fn n(&self) -> usize {
        self.n
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

    // TODO: More tests!
}
