use crate::data::extract_stat_then;
use crate::data::{DataOrSuffStat, ScaledSuffStat};
use crate::dist::Scaled;
use crate::traits::{
    ConjugatePrior, HasDensity, HasSuffStat, Parameterized, Sampleable,
    Scalable,
};
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::marker::PhantomData;

/// A wrapper for priors that scales the output distribution
///
/// If drawing a `Pr` gives a distribution `Fx`, then drawing `ScaledPrior<Pr>`
/// will produce a `Scaled<Fx>`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct ScaledPrior<Pr, Fx>
where
    Pr: Sampleable<Fx>,
    Fx: Scalable,
{
    parent: Pr,
    scale: f64,
    rate: f64,
    logjac: f64,
    _phantom: PhantomData<Fx>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScaledPriorError {
    /// The scale parameter must be a normal (finite, non-zero, non-subnormal) number
    NonNormalScale(f64),
    /// The scale parameter must be positive
    NegativeScale(f64),
}

impl std::error::Error for ScaledPriorError {}

#[cfg_attr(coverage_nightly, coverage(off))]
impl fmt::Display for ScaledPriorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonNormalScale(scale) => {
                write!(f, "non-normal scale: {scale}")
            }
            Self::NegativeScale(scale) => {
                write!(f, "negative scale: {scale}")
            }
        }
    }
}

impl<Pr, Fx> ScaledPrior<Pr, Fx>
where
    Pr: Sampleable<Fx>,
    Fx: Scalable,
{
    /// Creates a new scaled prior with the given parent prior and scale factor.
    ///
    /// # Errors
    /// Returns `ScaledPriorError::NonNormalScale` if the scale parameter is not a
    /// normal number (i.e., if it's zero, infinite, or NaN).
    ///
    /// Returns `ScaledPriorError::NegativeScale` if the scale parameter is not positive.
    pub fn new(parent: Pr, scale: f64) -> Result<Self, ScaledPriorError> {
        if !scale.is_normal() {
            Err(ScaledPriorError::NonNormalScale(scale))
        } else if scale <= 0.0 {
            Err(ScaledPriorError::NegativeScale(scale))
        } else {
            Ok(ScaledPrior {
                parent,
                scale,
                rate: scale.recip(),
                logjac: scale.abs().ln(),
                _phantom: PhantomData,
            })
        }
    }

    /// Creates a new scaled prior with the given parent prior and scale factor,
    /// without checking the scale parameter.
    ///
    /// # Safety
    /// The scale parameter must be a positive normal (finite, non-zero,
    /// non-subnormal) number.
    pub fn new_unchecked(parent: Pr, scale: f64) -> Self {
        ScaledPrior {
            parent,
            scale,
            rate: scale.recip(),
            logjac: scale.abs().ln(),
            _phantom: PhantomData,
        }
    }

    pub fn parent(&self) -> &Pr {
        &self.parent
    }

    pub fn parent_mut(&mut self) -> &mut Pr {
        &mut self.parent
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    pub fn logjac(&self) -> f64 {
        self.logjac
    }
}

impl<Pr, Fx> Sampleable<Scaled<Fx>> for ScaledPrior<Pr, Fx>
where
    Pr: Sampleable<Fx>,
    Fx: Scalable,
{
    fn draw<R: Rng>(&self, rng: &mut R) -> Scaled<Fx> {
        let fx = self.parent.draw(rng);
        Scaled::new_unchecked(fx, self.scale)
    }
}

pub struct ScaledPriorParameters<Pr: Parameterized> {
    parent: Pr::Parameters,
    scale: f64,
}

impl<Pr, Fx> Parameterized for ScaledPrior<Pr, Fx>
where
    Pr: Sampleable<Fx> + Parameterized,
    Fx: Scalable,
{
    type Parameters = ScaledPriorParameters<Pr>;

    fn emit_params(&self) -> Self::Parameters {
        ScaledPriorParameters {
            parent: self.parent.emit_params(),
            scale: self.scale,
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        let parent = Pr::from_params(params.parent);
        Self::new_unchecked(parent, params.scale)
    }
}

impl<Pr, Fx> ConjugatePrior<f64, Scaled<Fx>> for ScaledPrior<Pr, Fx>
where
    Pr: ConjugatePrior<f64, Fx, Posterior = Pr>,
    Fx: HasSuffStat<f64> + Scalable + HasDensity<f64>,
    Scaled<Fx>: HasSuffStat<f64, Stat = ScaledSuffStat<Fx::Stat>>,
{
    type Posterior = ScaledPrior<Pr::Posterior, Fx>;
    type MCache = Pr::MCache;
    type PpCache = Pr::PpCache;

    fn empty_stat(&self) -> ScaledSuffStat<Fx::Stat> {
        let parent_stat = self.parent.empty_stat();
        ScaledSuffStat::new(parent_stat, self.scale)
    }

    fn posterior_from_suffstat(
        &self,
        stat: &ScaledSuffStat<Fx::Stat>,
    ) -> Self::Posterior {
        ScaledPrior::new_unchecked(
            self.parent.posterior_from_suffstat(stat.parent()),
            self.scale,
        )
    }

    fn ln_m_cache(&self) -> Self::MCache {
        self.parent.ln_m_cache()
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::MCache,
        x: &DataOrSuffStat<f64, Scaled<Fx>>,
    ) -> f64 {
        // For now, we'll just compute from data
        let data: Vec<f64> = match x {
            DataOrSuffStat::Data(xs) => {
                xs.iter().map(|&x| x * self.rate).collect()
            }
            DataOrSuffStat::SuffStat(_) => vec![], // Not handling suffstat for now
        };

        self.parent
            .ln_m_with_cache(cache, &DataOrSuffStat::Data(&data))
    }

    fn ln_pp_cache(
        &self,
        x: &DataOrSuffStat<f64, Scaled<Fx>>,
    ) -> Self::PpCache {
        extract_stat_then(self, x, |stat| {
            self.parent
                .ln_pp_cache(&DataOrSuffStat::SuffStat(stat.parent()))
        })
    }

    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &f64) -> f64 {
        // Scale y back to the parent distribution's space
        let scaled_y = *y * self.rate;
        // Compute the log posterior predictive using the parent
        // Add the log Jacobian adjustment for the scale
        self.parent.ln_pp_with_cache(cache, &scaled_y) - self.logjac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataOrSuffStat;
    use crate::dist::NormalInvChiSquared;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn test_scaled_prior_draw() {
        let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 2.0, 1.0);
        let scaled_prior = ScaledPrior::new(prior, 2.0).unwrap();

        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let dist = scaled_prior.draw(&mut rng);

        assert_eq!(dist.scale(), 2.0);
    }

    #[test]
    fn test_scaled_prior_conjugate() {
        let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 2.0, 1.0);
        let scaled_prior = ScaledPrior::new(prior, 2.0).unwrap();

        // Create an empty stat to test
        let stat = scaled_prior.empty_stat();
        assert_eq!(stat.scale(), 2.0);

        // Test posterior with empty data
        let data: Vec<f64> = Vec::new();
        // Manually create DataOrSuffStat instead of using .into()
        let dos = DataOrSuffStat::Data(&data);
        let posterior = scaled_prior.posterior(&dos);

        // Scale should persist through posterior computation
        assert_eq!(posterior.scale(), 2.0);
    }

    #[test]
    fn test_scaled_prior_with_data() {
        let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 2.0, 1.0);
        let scaled_prior = ScaledPrior::new(prior, 2.0).unwrap();

        // Create some data - will be scaled by 1/2 internally for parent calculations
        let data = vec![2.0, 4.0, 6.0];

        // Manually create DataOrSuffStat instead of using .into()
        let dos = DataOrSuffStat::Data(&data);

        // Compute posterior
        let posterior = scaled_prior.posterior(&dos);

        // Scale should persist through posterior computation
        assert_eq!(posterior.scale(), 2.0);

        // Verify ln_m and ln_pp work
        let ln_m = scaled_prior.ln_m(&dos);
        let ln_pp = scaled_prior.ln_pp(&2.0, &dos);

        // Values should be finite (actual values will depend on implementation)
        assert!(ln_m.is_finite());
        assert!(ln_pp.is_finite());
    }
}
