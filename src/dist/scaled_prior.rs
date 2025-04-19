use crate::dist::Scaled;
use crate::traits::*;
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::marker::PhantomData;
use crate::data::{DataOrSuffStat, ScaledSuffStat};

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

impl fmt::Display for ScaledPriorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonNormalScale(scale) => {
                write!(f, "non-normal scale: {}", scale)
            }
            Self::NegativeScale(scale) => {
                write!(f, "negative scale: {}", scale)
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

/// Helper trait to convert between scaled and unscaled data
/// 
/// This is used internally by the ConjugatePrior implementation for ScaledPrior.
trait ScaleData<X> {
    /// Scale the data by the given factor
    fn scale_data(&self, scale: f64) -> Self;
}

impl<X: std::ops::Mul<f64, Output = X> + Clone> ScaleData<X> for X {
    fn scale_data(&self, scale: f64) -> Self {
        self.clone() * scale
    }
}

impl<X: std::ops::Mul<f64, Output = X> + Clone> ScaleData<X> for Vec<X> {
    fn scale_data(&self, scale: f64) -> Self {
        self.iter().map(|x| x.clone() * scale).collect()
    }
}

impl<X: std::ops::Mul<f64, Output = X> + Clone> ScaleData<X> for &[X] {
    fn scale_data(&self, scale: f64) -> Self {
        &self.iter().map(|x| x.clone() * scale).collect::<Vec<_>>()[..]
    }
}

impl<Pr, X, Fx> ConjugatePrior<X, Scaled<Fx>> for ScaledPrior<Pr, Fx>
where
    Pr: ConjugatePrior<X, Fx>,
    Fx: HasDensity<X> + HasSuffStat<X> + Scalable,
    X: std::ops::Mul<f64, Output = X> + Clone,
{
    type Posterior = ScaledPrior<Pr::Posterior, Fx>;
    type MCache = Pr::MCache;
    type PpCache = (Pr::PpCache, f64); // Parent cache and scale

    fn empty_stat(&self) -> <Scaled<Fx> as HasSuffStat<X>>::Stat {
        // For a Scaled<Fx>, the Stat is ScaledSuffStat<Fx::Stat>
        ScaledSuffStat::new(self.parent.empty_stat(), self.scale)
    }

    fn posterior(&self, x: &DataOrSuffStat<X, Scaled<Fx>>) -> Self::Posterior {
        // We need to convert the data for Scaled<Fx> into data for Fx
        // This means scaling by rate = 1/scale
        let parent_data = match x {
            DataOrSuffStat::Data(data) => {
                // Scale the data by rate
                let scaled_data = data.scale_data(self.rate);
                DataOrSuffStat::Data(&scaled_data)
            }
            DataOrSuffStat::SuffStat(stat) => {
                // The stat is already set up to handle the scaling
                // Just access the parent stat directly
                DataOrSuffStat::SuffStat(stat.parent())
            }
        };
        
        // Get posterior from parent
        let parent_posterior = self.parent.posterior(&parent_data);
        
        // Wrap in ScaledPrior with the same scale
        ScaledPrior::new_unchecked(parent_posterior, self.scale)
    }

    fn ln_m_cache(&self) -> Self::MCache {
        self.parent.ln_m_cache()
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::MCache,
        x: &DataOrSuffStat<X, Scaled<Fx>>,
    ) -> f64 {
        // We need to convert the data for Scaled<Fx> into data for Fx
        let parent_data = match x {
            DataOrSuffStat::Data(data) => {
                // Scale the data by rate
                let scaled_data = data.scale_data(self.rate);
                DataOrSuffStat::Data(&scaled_data)
            }
            DataOrSuffStat::SuffStat(stat) => {
                // The stat is already set up to handle the scaling
                // Just access the parent stat directly
                DataOrSuffStat::SuffStat(stat.parent())
            }
        };
        
        // Use parent's ln_m_with_cache
        self.parent.ln_m_with_cache(cache, &parent_data)
    }

    fn ln_pp_cache(&self, x: &DataOrSuffStat<X, Scaled<Fx>>) -> Self::PpCache {
        // We need to convert the data for Scaled<Fx> into data for Fx
        let parent_data = match x {
            DataOrSuffStat::Data(data) => {
                // Scale the data by rate
                let scaled_data = data.scale_data(self.rate);
                DataOrSuffStat::Data(&scaled_data)
            }
            DataOrSuffStat::SuffStat(stat) => {
                // The stat is already set up to handle the scaling
                // Just access the parent stat directly
                DataOrSuffStat::SuffStat(stat.parent())
            }
        };
        
        // Get cache from parent and save our scale
        (self.parent.ln_pp_cache(&parent_data), self.scale)
    }

    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &X) -> f64 {
        // Unpack the cache
        let (parent_cache, scale) = cache;
        
        // Scale y by rate to get into parent's space
        let scaled_y = y.clone() * self.rate;
        
        // Use parent's ln_pp_with_cache
        self.parent.ln_pp_with_cache(parent_cache, &scaled_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{Gaussian, NormalInvChiSquared};
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
    fn test_scale_data() {
        let x = 2.0;
        let scaled = x.scale_data(3.0);
        assert_eq!(scaled, 6.0);
        
        let vec = vec![1.0, 2.0, 3.0];
        let scaled_vec = vec.scale_data(2.0);
        assert_eq!(scaled_vec, vec![2.0, 4.0, 6.0]);
    }
} 