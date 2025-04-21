use crate::data::{DataOrSuffStat, ShiftedSuffStat};
use crate::dist::Shifted;
use crate::traits::*;
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::marker::PhantomData;

/// A wrapper for priors that shifts the output distribution
///
/// If drawing a `Pr` gives a distribution `Fx`, then drawing `ShiftedPrior<Pr>`
/// will produce a `Shifted<Fx>`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct ShiftedPrior<Pr, Fx>
where
    Pr: Sampleable<Fx>,
    Fx: Shiftable,
{
    parent: Pr,
    shift: f64,
    _phantom: PhantomData<Fx>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ShiftedPriorError {
    /// The shift parameter must be a finite number
    NonFiniteShift(f64),
}

impl std::error::Error for ShiftedPriorError {}

impl fmt::Display for ShiftedPriorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteShift(shift) => {
                write!(f, "non-finite shift: {}", shift)
            }
        }
    }
}

impl<Pr, Fx> ShiftedPrior<Pr, Fx>
where
    Pr: Sampleable<Fx>,
    Fx: Shiftable,
{
    /// Creates a new shifted prior with the given parent prior and shift value.
    ///
    /// # Errors
    /// Returns `ShiftedPriorError::NonFiniteShift` if the shift parameter is not
    /// a finite number (i.e., if it's infinite or NaN).
    pub fn new(parent: Pr, shift: f64) -> Result<Self, ShiftedPriorError> {
        if !shift.is_finite() {
            Err(ShiftedPriorError::NonFiniteShift(shift))
        } else {
            Ok(ShiftedPrior {
                parent,
                shift,
                _phantom: PhantomData,
            })
        }
    }

    /// Creates a new shifted prior with the given parent prior and shift value,
    /// without checking the shift parameter.
    ///
    /// # Safety
    /// The shift parameter must be a finite number.
    pub fn new_unchecked(parent: Pr, shift: f64) -> Self {
        ShiftedPrior {
            parent,
            shift,
            _phantom: PhantomData,
        }
    }

    pub fn parent(&self) -> &Pr {
        &self.parent
    }

    pub fn parent_mut(&mut self) -> &mut Pr {
        &mut self.parent
    }

    pub fn shift(&self) -> f64 {
        self.shift
    }
}

impl<Pr, Fx> Sampleable<Shifted<Fx>> for ShiftedPrior<Pr, Fx>
where
    Pr: Sampleable<Fx>,
    Fx: Shiftable,
{
    fn draw<R: Rng>(&self, rng: &mut R) -> Shifted<Fx> {
        let fx = self.parent.draw(rng);
        Shifted::new_unchecked(fx, self.shift)
    }
}

pub struct ShiftedPriorParameters<Pr: Parameterized> {
    parent: Pr::Parameters,
    shift: f64,
}

impl<Pr, Fx> Parameterized for ShiftedPrior<Pr, Fx>
where
    Pr: Sampleable<Fx> + Parameterized,
    Fx: Shiftable,
{
    type Parameters = ShiftedPriorParameters<Pr>;

    fn emit_params(&self) -> Self::Parameters {
        ShiftedPriorParameters {
            parent: self.parent.emit_params(),
            shift: self.shift,
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        let parent = Pr::from_params(params.parent);
        Self::new_unchecked(parent, params.shift)
    }
}

impl<Pr, Fx> ConjugatePrior<f64, Shifted<Fx>> for ShiftedPrior<Pr, Fx>
where
    Pr: ConjugatePrior<f64, Fx, Posterior = Pr>,
    Fx: HasSuffStat<f64> + Shiftable + HasDensity<f64>,
    Shifted<Fx>: HasSuffStat<f64, Stat = ShiftedSuffStat<Fx::Stat>>,
{
    type Posterior = Self;
    type MCache = Pr::MCache;
    type PpCache = Pr::PpCache;

    fn empty_stat(&self) -> ShiftedSuffStat<Fx::Stat> {
        let parent_stat = self.parent.empty_stat();
        ShiftedSuffStat::new(parent_stat, self.shift)
    }

    fn posterior(
        &self,
        x: &DataOrSuffStat<f64, Shifted<Fx>>,
    ) -> Self::Posterior {
        // For now, we'll just compute a new posterior with the same parameters
        // In the future, we should implement proper handling of the data
        let data: Vec<f64> = match x {
            DataOrSuffStat::Data(xs) => {
                xs.iter().map(|&x| x - self.shift).collect()
            }
            DataOrSuffStat::SuffStat(_) => vec![], // Not handling suffstat for now
        };

        let posterior_parent =
            self.parent.posterior(&DataOrSuffStat::Data(&data));
        Self::new_unchecked(posterior_parent, self.shift)
    }

    fn ln_m_cache(&self) -> Self::MCache {
        self.parent.ln_m_cache()
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::MCache,
        x: &DataOrSuffStat<f64, Shifted<Fx>>,
    ) -> f64 {
        // For now, we'll just compute from data
        let data: Vec<f64> = match x {
            DataOrSuffStat::Data(xs) => {
                xs.iter().map(|&x| x - self.shift).collect()
            }
            DataOrSuffStat::SuffStat(_) => vec![], // Not handling suffstat for now
        };

        self.parent
            .ln_m_with_cache(cache, &DataOrSuffStat::Data(&data))
    }

    fn ln_pp_cache(
        &self,
        x: &DataOrSuffStat<f64, Shifted<Fx>>,
    ) -> Self::PpCache {
        // For now, we'll just compute from data
        let data: Vec<f64> = match x {
            DataOrSuffStat::Data(xs) => {
                xs.iter().map(|&x| x - self.shift).collect()
            }
            DataOrSuffStat::SuffStat(_) => vec![], // Not handling suffstat for now
        };

        self.parent.ln_pp_cache(&DataOrSuffStat::Data(&data))
    }

    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &f64) -> f64 {
        // Shift y back to the parent distribution's space
        let shifted_y = *y - self.shift;
        // Compute the log posterior predictive using the parent
        self.parent.ln_pp_with_cache(cache, &shifted_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataOrSuffStat;
    use crate::dist::{Gaussian, NormalInvChiSquared, Shifted};
    use crate::traits::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn test_shifted_prior_draw() {
        let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 2.0, 1.0);
        let shifted_prior = ShiftedPrior::new(prior, 2.0).unwrap();

        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let dist = shifted_prior.draw(&mut rng);

        assert_eq!(dist.shift(), 2.0);
    }

    #[test]
    fn test_shifted_prior_conjugate() {
        let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 2.0, 1.0);
        let shifted_prior = ShiftedPrior::new(prior, 2.0).unwrap();

        // Create an empty stat to test
        let stat = shifted_prior.empty_stat();
        assert_eq!(stat.shift(), 2.0);

        // Test posterior with empty data
        let data: Vec<f64> = Vec::new();
        // Manually create DataOrSuffStat instead of using .into()
        let dos = DataOrSuffStat::Data(&data);
        let posterior = shifted_prior.posterior(&dos);

        // Shift should persist through posterior computation
        assert_eq!(posterior.shift(), 2.0);
    }

    #[test]
    fn test_shifted_prior_with_data() {
        let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 2.0, 1.0);
        let shifted_prior = ShiftedPrior::new(prior, 2.0).unwrap();

        // Create some data - will be shifted by -2.0 internally for parent calculations
        let data = vec![2.0, 4.0, 6.0];

        // Manually create DataOrSuffStat instead of using .into()
        let dos = DataOrSuffStat::Data(&data);

        // Compute posterior
        let posterior = shifted_prior.posterior(&dos);

        // Shift should persist through posterior computation
        assert_eq!(posterior.shift(), 2.0);

        // Verify ln_m and ln_pp work
        let ln_m = shifted_prior.ln_m(&dos);
        let ln_pp = shifted_prior.ln_pp(&2.0, &dos);

        // Values should be finite (actual values will depend on implementation)
        assert!(ln_m.is_finite());
        assert!(ln_pp.is_finite());
    }
}
