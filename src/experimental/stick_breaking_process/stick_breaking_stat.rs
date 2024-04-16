use crate::experimental::stick_breaking_process::stick_breaking::StickBreaking;
use crate::traits::{HasSuffStat, SuffStat};

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Represents the sufficient statistics for a stick-breaking process.
///
/// This struct is used to accumulate statistics from a stick-breaking process,
/// such as the number of breaks and the sum of the logarithms of the remaining stick lengths.
///
/// # Fields
///
/// * `n` - The total number of observations.
/// * `num_breaks` - The number of breaks observed.
/// * `sum_log_q` - The sum of the logarithms of the remaining stick lengths after each break.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct StickBreakingSuffStat {
    n: usize,
    num_breaks: usize,
    sum_log_q: f64,
}

impl Default for StickBreakingSuffStat {
    /// Provides a default instance of `StickBreakingSuffStat` with zeroed statistics.
    ///
    /// # Returns
    ///
    /// A new instance of `StickBreakingSuffStat` with all fields set to zero.
    fn default() -> Self {
        Self::new()
    }
}

impl StickBreakingSuffStat {
    /// Constructs a new `StickBreakingSuffStat`.
    ///
    /// Initializes a new instance of `StickBreakingSuffStat` with all fields set to zero,
    /// representing the start of a new stick-breaking process.
    ///
    /// # Returns
    ///
    /// A new instance of `StickBreakingSuffStat`.
    pub fn new() -> Self {
        Self {
            n: 0,
            num_breaks: 0,
            sum_log_q: 0.0,
        }
    }

    /// Returns the number of breaks observed in the stick-breaking process.
    pub fn num_breaks(&self) -> usize {
        self.num_breaks
    }

    /// Returns the sum of the logarithms of the remaining stick lengths after
    /// each break.
    pub fn sum_log_q(&self) -> f64 {
        self.sum_log_q
    }
}

impl From<&&[f64]> for StickBreakingSuffStat {
    /// Constructs a `StickBreakingSuffStat` from a slice of floating-point numbers.
    ///
    /// This conversion allows for directly observing a slice of stick lengths
    /// and accumulating their statistics into a `StickBreakingSuffStat`.
    ///
    /// # Arguments
    ///
    /// * `x` - A reference to a slice of floating-point numbers representing stick lengths.
    ///
    /// # Returns
    ///
    /// A new instance of `StickBreakingSuffStat` with observed statistics.
    fn from(x: &&[f64]) -> Self {
        let mut stat = StickBreakingSuffStat::new();
        stat.observe(x);
        stat
    }
}

// TODO: Generalize the above, something like
// impl<Stat,X> From<&X> for Stat
//     where Stat: SuffStat<X>
//     {
//     fn from(x: &X) -> Self {
//         let mut stat = Stat::new();
//         stat.observe(x);
//         stat
//     }
// }

/// Computes the sufficient statistic for a UnitPowerLaw distribution from a sequence of stick lengths.
///
/// This function processes a sequence of stick lengths resulting from a stick-breaking process
/// parameterized by a UnitPowerLaw(α), which is equivalent to a Beta(α,1) distribution. It calculates
/// the sufficient statistic for this distribution, which is necessary for further statistical analysis
/// or parameter estimation.
///
/// # Arguments
///
/// * `sticks` - A slice of floating-point numbers representing the lengths of the sticks.
///
/// # Returns
///
/// A tuple containing:
/// - The number of breaks (`usize`).
/// - The natural logarithm of the product of (1 - pᵢ) for each stick length pᵢ (`f64`).
fn stick_stat_unit_powerlaw(sticks: &[f64]) -> (usize, f64) {
    // First we need to find the sequence of remaining stick lengths. Because we
    // broke the sticks left-to-right, we need to reverse the sequence.
    let remaining = sticks.iter().rev().scan(0.0, |acc, &x| {
        *acc += x;
        Some(*acc)
    });

    let qs = sticks
        .iter()
        // Reversing `remaining` would force us to collect the intermediate
        // result e.g. into a `Vec`. Instead, we can reverse the sequence of
        // stick lengths to match.
        .rev()
        // Now zip the sequences together and do the main computation we're interested in.
        .zip(remaining)
        // In theory the broken stick lengths should all be less than what was
        // remaining before the break. In practice, numerical instabilities can
        // cause problems. So we filter to be sure we only consider valid
        // values.
        .filter(|(&len, remaining)| len < *remaining)
        .map(|(&len, remaining)| 1.0 - len / remaining);

    // The sufficient statistic is (n, ∑ᵢ log(1 - pᵢ)) == (n, log ∏ᵢ(1 - pᵢ)).
    // First we compute `n` and `∏ᵢ(1 - pᵢ)`
    let (num_breaks, prod_q) =
        qs.fold((0, 1.0), |(n, prod_q), q| (n + 1, prod_q * q));

    (num_breaks, prod_q.ln())
}

/// Implementation of `HasSuffStat` for `StickBreaking` with stick lengths as input.
impl HasSuffStat<&[f64]> for StickBreaking {
    type Stat = StickBreakingSuffStat;

    /// Creates an empty sufficient statistic for stick breaking.
    ///
    /// # Returns
    ///
    /// A new instance of `StickBreakingSuffStat` with zeroed statistics.
    fn empty_suffstat(&self) -> Self::Stat {
        Self::Stat::new()
    }

    /// Computes the natural logarithm of the likelihood function given the sufficient statistic.
    ///
    /// # Arguments
    ///
    /// * `stat` - A reference to the sufficient statistic of stick lengths.
    ///
    /// # Returns
    ///
    /// The natural logarithm of the likelihood function.
    fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
        let alpha = self.alpha();
        let alpha_ln = self.break_tail().alpha_ln();
        (stat.num_breaks as f64)
            .mul_add(alpha_ln, (alpha - 1.0) * stat.sum_log_q)
    }
}

/// Implementation of `SuffStat` for `StickBreakingSuffStat` with stick lengths as input.
impl SuffStat<&[f64]> for StickBreakingSuffStat {
    /// Returns the total number of observations.
    ///
    /// # Returns
    ///
    /// The total number of observations.
    fn n(&self) -> usize {
        self.n
    }

    /// Observes a sequence of stick lengths and updates the sufficient statistic.
    ///
    /// # Arguments
    ///
    /// * `sticks` - A reference to a slice of floating-point numbers representing stick lengths.
    fn observe(&mut self, sticks: &&[f64]) {
        let (num_breaks, sum_log_q) = stick_stat_unit_powerlaw(sticks);
        self.n += 1;
        self.num_breaks += num_breaks;
        self.sum_log_q += sum_log_q;
    }

    /// Reverses the observation of a sequence of stick lengths and updates the sufficient statistic.
    ///
    /// # Arguments
    ///
    /// * `sticks` - A reference to a slice of floating-point numbers representing stick lengths.
    fn forget(&mut self, sticks: &&[f64]) {
        let (num_breaks, sum_log_q) = stick_stat_unit_powerlaw(sticks);
        self.n -= 1;
        self.num_breaks -= num_breaks;
        self.sum_log_q -= sum_log_q;
    }
}
