use peroxide::fuga::Algorithm;
use rand::seq::SliceRandom;
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use super::StickSequence;
use crate::traits::*;

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
/// A "Stick-breaking discrete" distribution parameterized by a StickSequence.
pub struct Sbd {
    pub sticks: StickSequence,
}

/// Struct representing the Sbd (Sticks-based distribution) type.
/// Sbd is used to generate random numbers based on a given StickSequence.
impl Sbd {
    /// Creates a new instance of Sbd with the specified StickSequence.
    ///
    /// # Arguments
    ///
    /// * `sticks` - The StickSequence used for generating random numbers.
    ///
    /// # Returns
    ///
    /// A new instance of Sbd.
    pub fn new(sticks: StickSequence) -> Sbd {
        Self { sticks }
    }

    /// Calculates the inverse complementary cumulative distribution function
    /// (invccdf) of the Sbd. Sbd is based around the ccdf instead of the cdf
    /// because this allows for more precision in the tails.
    ///
    /// # Arguments
    ///
    /// * `p` - The value for which to calculate the invccdf.
    ///
    /// # Returns
    ///
    /// The index of the first element in the StickSequence whose value is less
    /// than `u`.
    pub fn invccdf(&self, p: f64) -> usize {
        self.sticks.extendmap_ccdf(
            |ccdf| ccdf.last().unwrap() < &p,
            |ccdf| ccdf.iter().position(|q| *q < p).unwrap() - 1,
        )
    }

    /// Calculates the inverse cumulative distribution function (invccdf) of the
    /// Sbd for multiple values, which are assumed to be already sorted. The
    /// returned vector contains the indices of the StickSequence elements whose
    /// values are less than the corresponding values in `ps`.
    ///
    /// # Arguments
    ///
    /// * `ps` - The values for which to calculate the invccdf.
    ///
    /// # Returns
    ///
    /// A vector containing the indices of the StickSequence elements whose
    /// values are less than the corresponding values in `ps`.
    pub fn multi_invccdf_sorted(&self, ps: &[f64]) -> Vec<usize> {
        let n = ps.len();
        self.sticks.extendmap_ccdf(
            // Note that ccdf is decreasing, but xs is increasing
            |ccdf| ccdf.last().unwrap() < ps.first().unwrap(),
            |ccdf| {
                let mut result: Vec<usize> = Vec::with_capacity(n);

                // We'll start at the end of the sorted uniforms (the largest value)
                let mut i: usize = n - 1;
                for q in ccdf.iter().skip(1).enumerate() {
                    while ps[i] > *q.1 {
                        result.push(q.0);
                        if i == 0 {
                            break;
                        } else {
                            i -= 1;
                        }
                    }
                }
                result
            },
        )
    }
}

/// Implementation of the `Support` trait for `Sbd`.
impl Support<usize> for Sbd {
    /// Checks if the given value is supported by `Sbd`.
    ///
    /// # Arguments
    ///
    /// * `x` - The value to be checked.
    ///
    /// # Returns
    ///
    /// Returns `true` if the value is greater than or equal to zero, `false` otherwise.
    fn supports(&self, _: &usize) -> bool {
        true
    }
}

/// Implementation of the `Cdf` trait for `Sbd`.
impl Cdf<usize> for Sbd {
    /// Calculates the survival function (SF) for a given value `x`.
    ///
    /// The survival function is defined as 1 minus the cumulative distribution function (CDF).
    /// It represents the probability that a random variable is greater than `x`.
    ///
    /// # Arguments
    ///
    /// * `x` - The value for which to calculate the survival function.
    ///
    /// # Returns
    ///
    /// The calculated survival function value as a `f64`.
    fn sf(&self, x: &usize) -> f64 {
        self.sticks.ccdf(x + 1)
    }

    /// Calculates the cumulative distribution function (CDF) for a given value `x`.
    ///
    /// The cumulative distribution function (CDF) represents the probability that a random variable
    /// is less than or equal to `x`.
    ///
    /// # Arguments
    ///
    /// * `x` - The value for which to calculate the cumulative distribution function.
    ///
    /// # Returns
    ///
    /// The calculated cumulative distribution function value as a `f64`.
    fn cdf(&self, x: &usize) -> f64 {
        1.0 - self.sf(x)
    }
}

impl InverseCdf<usize> for Sbd {
    fn invcdf(&self, p: f64) -> usize {
        self.invccdf(1.0 - p)
    }
}

impl DiscreteDistr<usize> for Sbd {}

impl Mode<usize> for Sbd {
    fn mode(&self) -> Option<usize> {
        let w0 = self.sticks.weight(0);
        // Once the unallocated mass is less than that of first stick, the
        // allocated mass is guaranteed to contain the mode.
        let n = self.sticks.extendmap_ccdf(
            |ccdf| ccdf.last().unwrap() < &w0,
            |ccdf| {
                let weights: Vec<f64> =
                    ccdf.windows(2).map(|qs| qs[0] - qs[1]).collect();
                weights.arg_max()
            },
        );
        Some(n)
    }
}

/// Generate a vector of sorted uniform random variables.
///
/// # Arguments
///     
/// * `n` - The number of random variables to generate.
///
/// * `rng` - A mutable reference to the random number generator.
///
/// # Returns
///
/// A vector of sorted uniform random variables.
///
/// # Example
///
/// ```
/// use rand::thread_rng;
/// use rv::experimental::sbd::sorted_uniforms;
///    
/// let mut rng = thread_rng();
/// let n = 10000;
/// let xs = sorted_uniforms(n, &mut rng);
/// assert!(xs.len() == n);
///
/// // Result is sorted and in the unit interval
/// assert!(&0.0 < xs.first().unwrap());
/// assert!(xs.last().unwrap() < &1.0);
/// assert!(xs.windows(2).all(|w| w[0] <= w[1]));
///
/// // Mean is 1/2
/// let mean = xs.iter().sum::<f64>() / n as f64;
/// assert!(mean > 0.49 && mean < 0.51);
///
/// // Variance is 1/12
/// let var = xs.iter().map(|x| (x - 0.5).powi(2)).sum::<f64>() / n as f64;
/// assert!(var > 0.08 && var < 0.09);
/// ```
pub fn sorted_uniforms<R: Rng>(n: usize, rng: &mut R) -> Vec<f64> {
    let mut xs: Vec<_> = (0..n)
        .map(|_| -rng.gen::<f64>().ln())
        .scan(0.0, |state, x| {
            *state += x;
            Some(*state)
        })
        .collect();
    let max = *xs.last().unwrap() - rng.gen::<f64>().ln();
    (0..n).for_each(|i| xs[i] /= max);
    xs
}

impl HasDensity<usize> for Sbd {
    fn f(&self, n: &usize) -> f64 {
        let sticks = &self.sticks;
        sticks.weight(*n)
    }

    fn ln_f(&self, n: &usize) -> f64 {
        self.f(n).ln()
    }
}

impl Sampleable<usize> for Sbd {
    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        let u: f64 = rng.gen();
        self.invccdf(u)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<usize> {
        let ps = sorted_uniforms(n, &mut rng);
        let mut result = self.multi_invccdf_sorted(&ps);

        // At this point `result` is sorted, so we need to shuffle it.
        // Note that shuffling is O(n) but sorting is O(n log n)
        result.shuffle(&mut rng);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::UnitPowerLaw;
    use rand::thread_rng;

    #[test]
    fn test_sorted_uniforms() {
        let mut rng = thread_rng();
        let n = 1000;
        let xs = sorted_uniforms(n, &mut rng);
        assert_eq!(xs.len(), n);

        // Result is sorted and in the unit interval
        assert!(&0.0 < xs.first().unwrap());
        assert!(xs.last().unwrap() < &1.0);
        assert!(xs.windows(2).all(|w| w[0] <= w[1]));

        // Mean is 1/2
        let mean = xs.iter().sum::<f64>() / n as f64;
        assert!((0.49..0.51).contains(&mean));

        // Variance is 1/12
        let var = xs.iter().map(|x| (x - 0.5).powi(2)).sum::<f64>() / n as f64;
        assert!((0.08..0.09).contains(&var))
    }

    #[test]
    fn test_multi_invccdf_sorted() {
        let sticks = StickSequence::new(UnitPowerLaw::new(10.0).unwrap(), None);
        let sbd = Sbd::new(sticks);
        let ps = sorted_uniforms(5, &mut thread_rng());
        assert_eq!(
            sbd.multi_invccdf_sorted(&ps),
            ps.iter().rev().map(|p| sbd.invccdf(*p)).collect::<Vec<_>>()
        )
    }
}
