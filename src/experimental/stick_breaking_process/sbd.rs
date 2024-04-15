use super::StickSequence;
use crate::dist::Mixture;
use crate::misc::ConvergentSequence;
use crate::traits::*;
use rand::seq::SliceRandom;
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
/// A "Stick-breaking discrete" distribution parameterized by a StickSequence.
pub struct StickBreakingDiscrete {
    sticks: StickSequence,
}

impl StickBreakingDiscrete {
    /// Creates a new instance of StickBreakingDiscrete with the specified StickSequence.
    ///
    /// # Arguments
    ///
    /// * `sticks` - The StickSequence used for generating random numbers.
    ///
    /// # Returns
    ///
    /// A new instance of StickBreakingDiscrete.
    pub fn new(sticks: StickSequence) -> StickBreakingDiscrete {
        Self { sticks }
    }

    /// Calculates the inverse complementary cumulative distribution function
    /// (invccdf) for the StickBreakingDiscrete distribution. This method is preferred over the
    /// traditional cumulative distribution function (cdf) as it provides higher precision in the
    /// tail regions of the distribution.
    ///
    /// # Arguments
    ///
    /// * `p` - The probability value for which to calculate the invccdf.
    ///
    /// # Returns
    ///
    /// The index of the first element in the StickSequence whose cumulative probability is less
    /// than `p`.
    pub fn invccdf(&self, p: f64) -> usize {
        debug_assert!(p > 0.0 && p < 1.0);
        self.sticks.extendmap_ccdf(
            |ccdf| ccdf.last().unwrap() < &p,
            |ccdf| ccdf.iter().position(|q| *q < p).unwrap() - 1,
        )
    }

    /// Provides a reference to the StickSequence used by the StickBreakingDiscrete distribution.
    ///
    /// # Returns
    ///
    /// A reference to the StickSequence.
    pub fn stick_sequence(&self) -> &StickSequence {
        &self.sticks
    }

    /// Calculates the inverse complementary cumulative distribution function (invccdf) for
    /// multiple sorted values. This method is useful for efficiently computing the invccdf for a
    /// sequence of values that are already sorted in ascending order. The returned vector contains
    /// the indices of the StickSequence elements whose cumulative probabilities are less than the
    /// corresponding values in `ps`.
    ///
    /// # Arguments
    ///
    /// * `ps` - A slice of probability values for which to calculate the invccdf. The values must
    ///   be sorted in ascending order.
    ///
    /// # Returns
    ///
    /// A vector containing the indices of the StickSequence elements whose cumulative probabilities
    /// are less than the corresponding values in `ps`.
    pub fn multi_invccdf_sorted(&self, ps: &[f64]) -> Vec<usize> {
        let n = ps.len();
        self.sticks.extendmap_ccdf(
            // Note that ccdf is decreasing, but ps is increasing
            |ccdf| ccdf.last().unwrap() < ps.first().unwrap(),
            |ccdf| {
                let mut result: Vec<usize> = Vec::with_capacity(n);

                // Start at the end of the sorted probability values (the largest value)
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

/// Implementation of the `Support` trait for `StickBreakingDiscrete`.
impl Support<usize> for StickBreakingDiscrete {
    /// Checks if the given value is supported by `StickBreakingDiscrete`.
    ///
    /// # Arguments
    ///
    /// * `x` - The value to be checked.
    ///
    /// # Returns
    ///
    /// Returns `true` for all values as `StickBreakingDiscrete` supports all `usize` values, `false` otherwise.
    fn supports(&self, _: &usize) -> bool {
        true
    }
}

/// Implementation of the `Cdf` trait for `StickBreakingDiscrete`.
impl Cdf<usize> for StickBreakingDiscrete {
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
        self.sticks.ccdf(*x + 1)
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

impl InverseCdf<usize> for StickBreakingDiscrete {
    /// Calculates the inverse cumulative distribution function (invcdf) for a given probability `p`.
    ///
    /// The inverse cumulative distribution function (invcdf) represents the value below which a random variable
    /// falls with probability `p`.
    ///
    /// # Arguments
    ///
    /// * `p` - The probability value for which to calculate the invcdf.
    ///
    /// # Returns
    ///
    /// The calculated invcdf value as a `usize`.
    fn invcdf(&self, p: f64) -> usize {
        self.invccdf(1.0 - p)
    }
}

impl DiscreteDistr<usize> for StickBreakingDiscrete {}

impl Mode<usize> for StickBreakingDiscrete {
    /// Calculates the mode of the `StickBreakingDiscrete` distribution.
    ///
    /// The mode is the value that appears most frequently in a data set or probability distribution.
    ///
    /// # Returns
    ///
    /// The mode of the distribution as an `Option<usize>`. Returns `None` if the mode cannot be determined.
    fn mode(&self) -> Option<usize> {
        let w0 = self.sticks.weight(0);
        // Once the unallocated mass is less than that of the first stick, the
        // allocated mass is guaranteed to contain the mode.
        let n = self.sticks.extendmap_ccdf(
            |ccdf| ccdf.last().unwrap() < &w0,
            |ccdf| {
                let weights: Vec<f64> =
                    ccdf.windows(2).map(|qs| qs[0] - qs[1]).collect();
                weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
            },
        );
        n
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
/// use rv::experimental::stick_breaking_process::sbd::sorted_uniforms;
///    
/// let mut rng = thread_rng();
/// let n = 10000;
/// let xs = sorted_uniforms(n, &mut rng);
/// assert_eq!(xs.len(), n);
///
/// // Result is sorted and in the unit interval
/// assert!(xs.first().map_or(false, |&first| first > 0.0));
/// assert!(xs.last().map_or(false, |&last| last < 1.0));
/// assert!(xs.windows(2).all(|w| w[0] <= w[1]));
///
/// // Mean is approximately 1/2
/// let mean = xs.iter().sum::<f64>() / n as f64;
/// assert!(mean > 0.49 && mean < 0.51);
///
/// // Variance is approximately 1/12
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
/// Provides density and log-density functions for StickBreakingDiscrete.
impl HasDensity<usize> for StickBreakingDiscrete {
    /// Computes the density of a given stick index.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the stick.
    ///
    /// # Returns
    ///
    /// The density of the stick at index `n`.
    fn f(&self, n: &usize) -> f64 {
        let sticks = &self.sticks;
        sticks.weight(*n)
    }

    /// Computes the natural logarithm of the density of a given stick index.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the stick.
    ///
    /// # Returns
    ///
    /// The natural logarithm of the density of the stick at index `n`.
    fn ln_f(&self, n: &usize) -> f64 {
        self.f(n).ln()
    }
}

/// Enables sampling from StickBreakingDiscrete.
impl Sampleable<usize> for StickBreakingDiscrete {
    /// Draws a single sample from the distribution.
    ///
    /// # Type Parameters
    ///
    /// * `R` - The random number generator type.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to the random number generator.
    ///
    /// # Returns
    ///
    /// A single sample as a usize.
    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        let u: f64 = rng.gen();
        self.invccdf(u)
    }

    /// Draws multiple samples from the distribution and shuffles them.
    ///
    /// # Type Parameters
    ///
    /// * `R` - The random number generator type.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of samples to draw.
    /// * `rng` - A mutable reference to the random number generator.
    ///
    /// # Returns
    ///
    /// A vector of usize samples, shuffled.
    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<usize> {
        let ps = sorted_uniforms(n, &mut rng);
        let mut result = self.multi_invccdf_sorted(&ps);

        // At this point `result` is sorted, so we need to shuffle it.
        // Note that shuffling is O(n) but sorting is O(n log n)
        result.shuffle(&mut rng);
        result
    }
}

impl Entropy for StickBreakingDiscrete {
    fn entropy(&self) -> f64 {
        let probs = (0..).map(|n| self.f(&n));
        probs
            .map(|p| p * p.ln())
            .scan(0.0, |state, x| {
                *state -= x;
                Some(*state)
            })
            .limit(1e-10)
    }
}

impl Entropy for &Mixture<StickBreakingDiscrete> {
    fn entropy(&self) -> f64 {
        let probs = (0..).map(|n| self.f(&n));
        probs
            .map(|p| p * p.ln())
            .scan(0.0, |state, x| {
                *state -= x;
                Some(*state)
            })
            .limit(1e-10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
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

        // t will aggregate our chi-squared test statistic
        let mut t = 0.0;

        {
            // We'll build a histogram and count the bin populations, aggregating
            // the chi-squared statistic as we go
            let mut next_bin = 0.01;
            let mut bin_pop = 0;

            for x in xs.iter() {
                bin_pop += 1;
                if *x > next_bin {
                    let obs = bin_pop as f64;
                    let exp = n as f64 / 100.0;
                    t += (obs - exp).powi(2) / exp;
                    bin_pop = 0;
                    next_bin += 0.01;
                }
            }

            // The last bin
            let obs = bin_pop as f64;
            let exp = n as f64 / 100.0;
            t += (obs - exp).powi(2) / exp;
        }

        let alpha = 0.001;

        // dof = number of bins minus one
        let chi2 = ChiSquared::new(99.0).unwrap();
        let p = chi2.sf(&t);
        assert!(p > alpha);
    }

    #[test]
    fn test_multi_invccdf_sorted() {
        let sticks = StickSequence::new(UnitPowerLaw::new(10.0).unwrap(), None);
        let sbd = StickBreakingDiscrete::new(sticks);
        let ps = sorted_uniforms(5, &mut thread_rng());
        assert_eq!(
            sbd.multi_invccdf_sorted(&ps),
            ps.iter().rev().map(|p| sbd.invccdf(*p)).collect::<Vec<_>>()
        )
    }
}
