//! Trait definitions
pub use crate::data::DataOrSuffStat;
use rand::Rng;

pub trait Parameterized: Sized {
    type Parameters;

    fn emit_params(&self) -> Self::Parameters;

    fn from_params(params: Self::Parameters) -> Self;

    fn map_params(
        &self,
        f: impl Fn(Self::Parameters) -> Self::Parameters,
    ) -> Self {
        let params = self.emit_params();
        let new_params = f(params);
        Self::from_params(new_params)
    }
}

pub trait Sampleable<X> {
    /// Single draw from the `Rv`
    ///
    /// # Example
    ///
    /// Flip a coin
    ///
    /// ```
    /// use rv::dist::Bernoulli;
    /// use rv::traits::*;
    ///
    /// let b = Bernoulli::uniform();
    /// let mut rng = rand::thread_rng();
    /// let x: bool = b.draw(&mut rng); // could be true, could be false.
    /// ```
    fn draw<R: Rng>(&self, rng: &mut R) -> X;

    /// Multiple draws of the `Rv`
    ///
    /// # Example
    ///
    /// Flip a lot of coins
    ///
    /// ```
    /// use rv::dist::Bernoulli;
    /// use rv::traits::*;
    ///
    /// let b = Bernoulli::uniform();
    /// let mut rng = rand::thread_rng();
    /// let xs: Vec<bool> = b.sample(22, &mut rng);
    ///
    /// assert_eq!(xs.len(), 22);
    /// ```
    ///
    /// Estimate Gaussian mean
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::*;
    ///
    /// let gauss = Gaussian::standard();
    /// let mut rng = rand::thread_rng();
    /// let xs: Vec<f64> = gauss.sample(100_000, &mut rng);
    ///
    /// assert::close(xs.iter().sum::<f64>()/100_000.0, 0.0, 1e-2);
    /// ```
    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<X> {
        (0..n).map(|_| self.draw(&mut rng)).collect()
    }

    /// Create a never-ending iterator of samples
    ///
    /// # Example
    ///
    /// Estimate the mean of a Gamma distribution
    ///
    /// ```
    /// use rv::traits::*;
    /// use rv::dist::Gamma;
    ///
    /// let mut rng = rand::thread_rng();
    ///
    /// let gamma = Gamma::new(2.0, 1.0).unwrap();
    ///
    /// let n = 1_000_000_usize;
    /// let mean = <Gamma as Sampleable<f64>>::sample_stream(&gamma, &mut rng)
    ///     .take(n)
    ///     .sum::<f64>() / n as f64;;
    ///
    /// assert::close(mean, 2.0, 1e-2);
    /// ```
    fn sample_stream<'r, R: Rng>(
        &'r self,
        mut rng: &'r mut R,
    ) -> Box<dyn Iterator<Item = X> + 'r> {
        Box::new(std::iter::repeat_with(move || self.draw(&mut rng)))
    }
}

pub trait HasDensity<X> {
    /// Probability function
    ///
    /// # Example
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::*;
    ///
    /// let g = Gaussian::standard();
    /// assert!(g.f(&0.0_f64) > g.f(&0.1_f64));
    /// assert!(g.f(&0.0_f64) > g.f(&-0.1_f64));
    /// ```
    fn f(&self, x: &X) -> f64 {
        self.ln_f(x).exp()
    }

    /// Probability function
    ///
    /// # Example
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::*;
    ///
    /// let g = Gaussian::standard();
    /// assert!(g.ln_f(&0.0_f64) > g.ln_f(&0.1_f64));
    /// assert!(g.ln_f(&0.0_f64) > g.ln_f(&-0.1_f64));
    /// ```
    fn ln_f(&self, x: &X) -> f64;
}

/// Random variable
///
/// Contains the minimal functionality that a random object must have to be
/// useful: a function defining the un-normalized density/mass at a point,
/// and functions to draw samples from the distribution.
pub trait Rv<X>: Sampleable<X> + HasDensity<X> {}

impl<X, T> Rv<X> for T where T: Sampleable<X> + HasDensity<X> {}

/// Stochastic process
///
pub trait Process<S, O>: Sampleable<S> + HasDensity<O> {}

impl<S, O, T> Process<S, O> for T where T: Sampleable<S> + HasDensity<O> {}

/// Identifies the support of the Rv
pub trait Support<X> {
    /// Returns `true` if `x` is in the support of the `Rv`
    ///
    /// # Example
    ///
    /// ```
    /// use rv::dist::Uniform;
    /// use rv::traits::Support;
    ///
    /// // Create uniform with support on the interval [0, 1]
    /// let u = Uniform::new(0.0, 1.0).unwrap();
    ///
    /// assert!(u.supports(&0.5_f64));
    /// assert!(!u.supports(&-0.1_f64));
    /// assert!(!u.supports(&1.1_f64));
    /// ```
    fn supports(&self, x: &X) -> bool;
}

/// Is a continuous probability distributions
///
/// This trait uses the `Rv<X>` and `Support<X>` implementations to implement
/// itself.
pub trait ContinuousDistr<X>: HasDensity<X> + Support<X> {
    /// The value of the Probability Density Function (PDF) at `x`
    ///
    /// # Example
    ///
    /// Compute the Gaussian PDF, f(x)
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::ContinuousDistr;
    ///
    /// let g = Gaussian::standard();
    ///
    /// let f_mean = g.pdf(&0.0_f64);
    /// let f_low = g.pdf(&-1.0_f64);
    /// let f_high = g.pdf(&1.0_f64);
    ///
    /// assert!(f_mean > f_low);
    /// assert!(f_mean > f_high);
    /// assert!((f_low - f_high).abs() < 1E-12);
    /// ```
    ///
    /// Returns 0 if x is not in support
    ///
    /// ```
    /// # use rv::traits::ContinuousDistr;
    /// use rv::dist::Exponential;
    ///
    /// let expon = Exponential::new(1.0).unwrap();
    /// let f = expon.pdf(&-1.0_f64);
    /// assert_eq!(f, 0.0);
    /// ```
    fn pdf(&self, x: &X) -> f64 {
        self.f(x)
    }

    /// The value of the log Probability Density Function (PDF) at `x`
    ///
    /// # Example
    ///
    /// Compute the natural logarithm of the Gaussian PDF, ln(f(x))
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::ContinuousDistr;
    ///
    /// let g = Gaussian::standard();
    ///
    /// let lnf_mean = g.ln_pdf(&0.0_f64);
    /// let lnf_low = g.ln_pdf(&-1.0_f64);
    /// let lnf_high = g.ln_pdf(&1.0_f64);
    ///
    /// assert!(lnf_mean > lnf_low);
    /// assert!(lnf_mean > lnf_high);
    /// assert!((lnf_low - lnf_high).abs() < 1E-12);
    /// ```
    ///
    /// Returns -inf if x is not in support
    ///
    /// ```
    /// # use rv::traits::ContinuousDistr;
    /// use rv::dist::Exponential;
    ///
    /// let expon = Exponential::new(1.0).unwrap();
    /// let f = expon.ln_pdf(&-1.0_f64);
    /// assert_eq!(f, f64::NEG_INFINITY);
    /// ```
    fn ln_pdf(&self, x: &X) -> f64 {
        if self.supports(x) {
            self.ln_f(x)
        } else {
            f64::NEG_INFINITY
        }
    }
}

/// Has a cumulative distribution function (CDF)
pub trait Cdf<X>: HasDensity<X> {
    /// The value of the Cumulative Density Function at `x`
    ///
    /// # Example
    ///
    /// The proportion of probability in (-∞, μ) in N(μ, σ) is 50%
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::Cdf;
    ///
    /// let g = Gaussian::new(1.0, 1.5).unwrap();
    ///
    /// assert!((g.cdf(&1.0_f64) - 0.5).abs() < 1E-12);
    /// ```
    fn cdf(&self, x: &X) -> f64;

    /// Survival function, `1 - CDF(x)`
    fn sf(&self, x: &X) -> f64 {
        1.0 - self.cdf(x)
    }
}

/// Has an inverse-CDF / quantile function
pub trait InverseCdf<X>: HasDensity<X> + Support<X> {
    /// The value of the `x` at the given probability in the CDF
    ///
    /// # Example
    ///
    /// The CDF identity: p = CDF(x) => x = CDF<sup>-1</sup>(p)
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::Cdf;
    /// use rv::traits::InverseCdf;
    ///
    /// let g = Gaussian::standard();
    ///
    /// let x: f64 = 1.2;
    /// let p: f64 = g.cdf(&x);
    /// let y: f64 = g.invcdf(p);
    ///
    /// // x and y should be about the same
    /// assert!((x - y).abs() < 1E-12);
    /// ```
    fn invcdf(&self, p: f64) -> X;

    /// Alias for `invcdf`
    fn quantile(&self, p: f64) -> X {
        self.invcdf(p)
    }

    /// Interval containing `p` proportion for the probability
    ///
    /// # Example
    ///
    /// Confidence interval
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::InverseCdf;
    ///
    /// let g = Gaussian::new(100.0, 15.0).unwrap();
    /// let ci: (f64, f64) = g.interval(0.68268949213708585);  // one stddev
    /// assert!( (ci.0 - 85.0).abs() < 1E-12);
    /// assert!( (ci.1 - 115.0).abs() < 1E-12);
    /// ```
    fn interval(&self, p: f64) -> (X, X) {
        let pt = (1.0 - p) / 2.0;
        (self.quantile(pt), self.quantile(p + pt))
    }
}

/// Is a discrete probability distribution
pub trait DiscreteDistr<X>: Rv<X> + Support<X> {
    /// Probability mass function (PMF) at `x`
    ///
    /// # Panics
    ///
    /// If `x` is not supported
    ///
    /// # Example
    ///
    /// The probability of a fair coin coming up heads in 0.5
    ///
    /// ```
    /// use rv::dist::Bernoulli;
    /// use rv::traits::DiscreteDistr;
    ///
    /// // Fair coin (p = 0.5)
    /// let b = Bernoulli::uniform();
    ///
    /// assert!( (b.pmf(&true) - 0.5).abs() < 1E-12);
    /// ```
    fn pmf(&self, x: &X) -> f64 {
        self.ln_pmf(x).exp()
    }

    /// Natural logarithm of the probability mass function (PMF)
    ///
    /// # Example
    ///
    /// The probability of a fair coin coming up heads in 0.5
    ///
    /// ```
    /// use rv::dist::Bernoulli;
    /// use rv::traits::DiscreteDistr;
    ///
    /// // Fair coin (p = 0.5)
    /// let b = Bernoulli::uniform();
    ///
    /// assert!( (b.ln_pmf(&true) - 0.5_f64.ln()).abs() < 1E-12);
    /// ```
    fn ln_pmf(&self, x: &X) -> f64 {
        if self.supports(x) {
            self.ln_f(x)
        } else {
            f64::NEG_INFINITY
        }
    }
}

/// Defines the distribution mean
pub trait Mean<X> {
    /// Returns `None` if the mean is undefined
    fn mean(&self) -> Option<X>;
}

/// Defines the distribution median
pub trait Median<X> {
    /// Returns `None` if the median is undefined
    fn median(&self) -> Option<X>;
}

/// Defines the distribution mode
pub trait Mode<X> {
    /// Returns `None` if the mode is undefined or is not a single value
    fn mode(&self) -> Option<X>;
}

/// Defines the distribution variance
pub trait Variance<X> {
    /// Returns `None` if the variance is undefined
    fn variance(&self) -> Option<X>;
}

/// Defines the distribution entropy
pub trait Entropy {
    /// The entropy, *H(X)*
    fn entropy(&self) -> f64;
}

pub trait Skewness {
    fn skewness(&self) -> Option<f64>;
}

pub trait Kurtosis {
    fn kurtosis(&self) -> Option<f64>;
}

/// KL divergences
pub trait KlDivergence {
    /// The KL divergence, KL(P|Q) between this distribution, P, and another, Q
    ///
    /// # Example
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::KlDivergence;
    ///
    /// let g1 = Gaussian::new(1.0, 1.0).unwrap();
    /// let g2 = Gaussian::new(-1.0, 2.0).unwrap();
    ///
    /// let kl_self = g1.kl(&g1);
    /// let kl_other = g1.kl(&g2);
    ///
    /// // KL(P|P) = 0
    /// assert!( kl_self < 1E-12 );
    ///
    /// // KL(P|Q) > 0 if P ≠ Q
    /// assert!( kl_self < kl_other );
    /// ```
    fn kl(&self, other: &Self) -> f64;

    /// Symmetrized divergence, KL(P|Q) + KL(Q|P)
    ///
    /// # Example
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::KlDivergence;
    ///
    /// let g1 = Gaussian::new(1.0, 1.0).unwrap();
    /// let g2 = Gaussian::new(-1.0, 2.0).unwrap();
    ///
    /// let kl_12 = g1.kl(&g2);
    /// let kl_21 = g2.kl(&g1);
    ///
    /// let kl_sym = g1.kl_sym(&g2);
    ///
    /// assert!( (kl_12 + kl_21 - kl_sym).abs() < 1E-10 );
    /// ```
    fn kl_sym(&self, other: &Self) -> f64 {
        self.kl(other) + other.kl(self)
    }
}

/// A prior on `Fx` that induces a posterior that is the same form as the prior
///
/// # Example
///
/// Conjugate analysis of coin flips using Bernoulli with a Beta prior on the
/// success probability.
///
/// ```
/// use rv::traits::LegacyConjugatePrior;
/// use rv::dist::{Bernoulli, Beta};
///
/// let flips = vec![true, false, false];
/// let prior = Beta::jeffreys();
///
/// // If we observe more false than true, the posterior predictive
/// // probability of true decreases.
/// let pp_no_obs = prior.pp(&true, &(&vec![]).into());
/// let pp_obs = prior.pp(&true, &(&flips).into());
///
/// assert!(pp_obs < pp_no_obs);
/// ```
///
/// Use a cache to speed up repeated computations.
///
/// ```
/// # use rv::traits::LegacyConjugatePrior;
/// use rv::traits::*;
/// use rv::traits::SuffStat;
/// use rv::dist::{Categorical, SymmetricDirichlet};
/// use rv::data::{CategoricalSuffStat, DataOrSuffStat};
/// use std::time::Instant;
///
/// let ncats = 10;
/// let symdir = SymmetricDirichlet::jeffreys(ncats).unwrap();
/// let mut suffstat = CategoricalSuffStat::new(10);
/// let mut rng = rand::thread_rng();
///
/// Categorical::new(&vec![1.0, 1.0, 5.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0])
///     .unwrap()
///     .sample_stream(&mut rng)
///     .take(1000)
///     .for_each(|x: u8| suffstat.observe(&x));
///
///
/// let stat = DataOrSuffStat::SuffStat(&suffstat);
///
/// // Get predictions from predictive distribution using the cache
/// let t_cache = {
///     let cache = symdir.ln_pp_cache(&stat);
///     let t_start = Instant::now();
///     // Argmax
///     let k_max = (0..ncats).fold((0, f64::NEG_INFINITY), |(ix, f), y| {
///             let f_r = symdir.ln_pp_with_cache(&cache, &y);
///             if f_r > f {
///                 (y, f_r)
///             } else {
///                 (ix, f)
///             }
///
///         });
///
///     assert_eq!(k_max.0, 2);
///     t_start.elapsed()
/// };
///
/// // Get predictions from predictive distribution w/o cache
/// let t_no_cache = {
///     let t_start = Instant::now();
///     // Argmax
///     let k_max = (0..ncats).fold((0, f64::NEG_INFINITY), |(ix, f), y| {
///             let f_r = symdir.ln_pp(&y, &stat);
///             if f_r > f {
///                 (y, f_r)
///             } else {
///                 (ix, f)
///             }
///
///         });
///
///     assert_eq!(k_max.0, 2);
///     t_start.elapsed()
/// };
///
/// // Using cache improves runtime
/// assert!(t_no_cache.as_nanos() > t_cache.as_nanos());
/// ```
pub trait LegacyConjugatePrior<X, Fx>: Sampleable<Fx>
where
    Fx: HasDensity<X> + HasSuffStat<X>,
{
    /// Type of the posterior distribution
    type Posterior: Sampleable<Fx>;
    /// Type of the cache for the marginal likelihood
    type MCache;
    /// Type of the cache for the posterior predictive
    type PpCache;

    /// Generate and empty sufficient statistic
    fn empty_stat(&self) -> Fx::Stat;

    /// Computes the posterior distribution from the data
    fn posterior_from_suffstat(&self, stat: &Fx::Stat) -> Self::Posterior {
        self.posterior(&DataOrSuffStat::SuffStat(stat))
    }

    fn posterior(&self, x: &DataOrSuffStat<X, Fx>) -> Self::Posterior;

    /// Compute the cache for the log marginal likelihood.
    fn ln_m_cache(&self) -> Self::MCache;

    /// Log marginal likelihood with supplied cache.
    fn ln_m_with_cache(
        &self,
        cache: &Self::MCache,
        x: &DataOrSuffStat<X, Fx>,
    ) -> f64;

    /// The log marginal likelihood
    fn ln_m(&self, x: &DataOrSuffStat<X, Fx>) -> f64 {
        let cache = self.ln_m_cache();
        self.ln_m_with_cache(&cache, x)
    }

    /// Compute the cache for the Log posterior predictive of y given x.
    ///
    /// The cache should encompass all information about `x`.
    fn ln_pp_cache(&self, x: &DataOrSuffStat<X, Fx>) -> Self::PpCache;

    /// Log posterior predictive of y given x with supplied ln(norm)
    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &X) -> f64;

    /// Log posterior predictive of y given x
    fn ln_pp(&self, y: &X, x: &DataOrSuffStat<X, Fx>) -> f64 {
        let cache = self.ln_pp_cache(x);
        self.ln_pp_with_cache(&cache, y)
    }

    /// Marginal likelihood of x
    fn m(&self, x: &DataOrSuffStat<X, Fx>) -> f64 {
        self.ln_m(x).exp()
    }

    fn pp_with_cache(&self, cache: &Self::PpCache, y: &X) -> f64 {
        self.ln_pp_with_cache(cache, y).exp()
    }

    /// Posterior Predictive distribution
    fn pp(&self, y: &X, x: &DataOrSuffStat<X, Fx>) -> f64 {
        self.ln_pp(y, x).exp()
    }
}

/// Get the quad bounds of a univariate real distribution
pub trait QuadBounds {
    fn quad_bounds(&self) -> (f64, f64);
}

/// The data for this distribution can be summarized by a statistic
pub trait HasSuffStat<X> {
    type Stat: SuffStat<X>;

    fn empty_suffstat(&self) -> Self::Stat;

    /// Return the log likelihood for the data represented by the sufficient
    /// statistic.
    fn ln_f_stat(&self, stat: &Self::Stat) -> f64;
}

/// Is a [sufficient statistic](https://en.wikipedia.org/wiki/Sufficient_statistic) for a
/// distribution.
///
/// # Examples
///
/// Basic suffstat usage.
///
/// ```
/// use rv::data::BernoulliSuffStat;
/// use rv::traits::SuffStat;
///
/// // Bernoulli sufficient statistics are the number of observations, n, and
/// // the number of successes, k.
/// let mut stat = BernoulliSuffStat::new();
///
/// assert!(stat.n() == 0 && stat.k() == 0);
///
/// stat.observe(&true);  // observe `true`
/// assert!(stat.n() == 1 && stat.k() == 1);
///
/// stat.observe(&false);  // observe `false`
/// assert!(stat.n() == 2 && stat.k() == 1);
///
/// stat.forget_many(&vec![false, true]);  // forget `true` and `false`
/// assert!(stat.n() == 0 && stat.k() == 0);
/// ```
///
/// Conjugate analysis of coin flips using Bernoulli with a Beta prior on the
/// success probability.
///
/// ```
/// use rv::traits::SuffStat;
/// use rv::traits::LegacyConjugatePrior;
/// use rv::data::BernoulliSuffStat;
/// use rv::dist::{Bernoulli, Beta};
///
/// let flips = vec![true, false, false];
///
/// // Pack the data into a sufficient statistic that holds the number of
/// // trials and the number of successes
/// let mut stat = BernoulliSuffStat::new();
/// stat.observe_many(&flips);
///
/// let prior = Beta::jeffreys();
///
/// // If we observe more false than true, the posterior predictive
/// // probability of true decreases.
/// let pp_no_obs = prior.pp(&true, &(&BernoulliSuffStat::new()).into());
/// let pp_obs = prior.pp(&true, &(&flips).into());
///
/// assert!(pp_obs < pp_no_obs);
/// ```
pub trait SuffStat<X> {
    /// Returns the number of observations
    fn n(&self) -> usize;

    /// Assimilate the datum `x` into the statistic
    fn observe(&mut self, x: &X);

    /// Remove the datum `x` from the statistic
    fn forget(&mut self, x: &X);

    /// Assimilate several observations
    fn observe_many(&mut self, xs: &[X]) {
        xs.iter().for_each(|x| self.observe(x));
    }

    /// Forget several observations
    fn forget_many(&mut self, xs: &[X]) {
        xs.iter().for_each(|x| self.forget(x));
    }

    /// Combine sufficient statistics
    fn merge(&mut self, other: Self);
}

/// Trait for distributions that can be shifted by a constant value
pub trait Shiftable {
    type Output;
    type Error;

    fn shifted(self, shift: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized;

    fn shifted_unchecked(self, shift: f64) -> Self::Output
    where
        Self: Sized;
}

/// Macro to implement Shiftable for a distribution type
///
/// This macro automatically implements the Shiftable trait for a given type,
/// using the default Shifted<T> as the Output type.
#[macro_export]
macro_rules! impl_shiftable {
    // Simple case for non-generic types
    ($type:ty) => {
        use $crate::prelude::Shifted;
        use $crate::prelude::ShiftedError;

        impl Shiftable for $type {
            type Output = Shifted<Self>;
            type Error = ShiftedError;

            fn shifted(self, shift: f64) -> Result<Self::Output, Self::Error>
            where
                Self: Sized,
            {
                Shifted::new(self, shift)
            }

            fn shifted_unchecked(self, shift: f64) -> Self::Output
            where
                Self: Sized,
            {
                Shifted::new_unchecked(self, shift)
            }
        }
    };
}

/// A distribution that can absorb scaling into its parameters
pub trait Scalable {
    type Output;
    type Error;

    fn scaled(self, scale: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized;

    fn scaled_unchecked(self, scale: f64) -> Self::Output
    where
        Self: Sized;
}

/// Macro to implement Scalable for a distribution type
///
/// This macro automatically implements the Scalable trait for a given type,
/// using the default Scaled<T> as the Output type.
#[macro_export]
macro_rules! impl_scalable {
    // Simple case for non-generic types
    ($type:ty) => {
        use $crate::prelude::Scaled;
        use $crate::prelude::ScaledError;

        impl Scalable for $type {
            type Output = Scaled<Self>;
            type Error = ScaledError;

            fn scaled(self, scale: f64) -> Result<Self::Output, Self::Error>
            where
                Self: Sized,
            {
                Scaled::new(self, scale)
            }

            fn scaled_unchecked(self, scale: f64) -> Self::Output
            where
                Self: Sized,
            {
                Scaled::new_unchecked(self, scale)
            }
        }
    };
}

#[cfg(test)]
mod test {
    #[macro_export]
    macro_rules! test_shiftable_mean {
    ($expr:expr) => {
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn shiftable_mean(shift in -100.0..100.0) {
                let dist = $expr;
                let shifted = dist.clone().shifted_unchecked(shift);
                let manual = Shifted::new_unchecked(dist, shift);

                let mean_shifted = shifted.mean();
                let mean_manual = manual.mean();
                match (mean_shifted, mean_manual) {
                    (Some(mean_shifted), Some(mean_manual)) => {
                        let mean_shifted: f64 = mean_shifted;
                        let mean_manual: f64 = mean_manual;
                        prop_assert!($crate::misc::eq_or_close(mean_shifted, mean_manual, 1e-10), "means differ: {} vs {}", mean_shifted, mean_manual);
                    }
                    (None, None) => {},
                    _ => {
                        prop_assert!(false, "Shifting should not affect existence of mean");
                    }
                }
            }
        }
    };
}

    #[macro_export]
    macro_rules! test_shiftable_method {
        // Base case with no extension
        ($expr:expr, $ident:ident) => {
            test_shiftable_method!($expr, $ident, );
        };

        // Main implementation
        ($expr:expr, $ident:ident, $($ext:ident)?) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<shiftable_ $ident $(_ $ext)?>](shift in -100.0..100.0) {
                        let dist = $expr;
                        let shifted = dist.clone().shifted_unchecked(shift).$ident();
                        let manual = $crate::prelude::Shifted::new_unchecked(dist, shift).$ident();

                        match (shifted, manual) {
                            (Some(shifted), Some(manual)) => {
                                let shifted: f64 = shifted;
                                let manual: f64 = manual;
                                proptest::prop_assert!($crate::misc::eq_or_close(shifted, manual, 1e-10),
                                    "{}s differ: {} vs {}", stringify!($ident), shifted, manual);
                            }
                            (None, None) => {},
                            _ => {
                                proptest::prop_assert!(false, "Shifting should not affect existence of {}",
                                    stringify!($ident));
                            }
                        }
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_shiftable_density {
        ($expr:expr) => {
            test_shiftable_density!($expr, );
        };

        ($expr:expr, $($ext:ident)?) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<shiftable_density $(_ $ext)?>](y in -100.0..100.0, shift in -100.0..100.0) {
                        let dist = $expr;
                        let shifted: f64 = dist.clone().shifted_unchecked(shift).ln_f(&y);
                        let manual: f64 = $crate::prelude::Shifted::new_unchecked(dist, shift).ln_f(&y);
                        proptest::prop_assert!($crate::misc::eq_or_close(shifted, manual, 1e-10),
                            "densities differ: {} vs {}", shifted, manual);
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_shiftable_cdf {
        ($expr:expr) => {
            test_shiftable_cdf!($expr, );
        };

        ($expr:expr, $($ext:ident)?) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<shiftable_cdf $(_ $ext)?>](x in -100.0..100.0, shift in -100.0..100.0) {
                        let dist = $expr;
                        let shifted: f64 = dist.clone().shifted_unchecked(shift).cdf(&x);
                        let manual: f64 = $crate::prelude::Shifted::new_unchecked(dist, shift).cdf(&x);
                        proptest::prop_assert!($crate::misc::eq_or_close(shifted, manual, 1e-10),
                            "cdfs differ: {} vs {}", shifted, manual);
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_shiftable_invcdf {
        ($expr:expr) => {
            test_shiftable_invcdf!($expr, );
        };

        ($expr:expr, $($ext:ident)?) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<shiftable_invcdf $(_ $ext)?>](p in 0.0..1.0, shift in -100.0..100.0) {
                        let dist = $expr;
                        let shifted: f64 = dist.clone().shifted_unchecked(shift).invcdf(p);
                        let manual: f64 = $crate::prelude::Shifted::new_unchecked(dist, shift).invcdf(p);
                        proptest::prop_assert!($crate::misc::eq_or_close(shifted, manual, 1e-10),
                            "invcdfs differ: {} vs {}", shifted, manual);
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_shiftable_entropy {
        ($expr:expr) => {
            test_shiftable_entropy!($expr, );
        };

        ($expr:expr, $($ext:ident)?) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<shiftable_entropy $(_ $ext)?>](shift in -100.0..100.0) {
                        let dist = $expr;
                        let shifted: f64 = dist.clone().shifted_unchecked(shift).entropy();
                        let manual: f64 = $crate::prelude::Shifted::new_unchecked(dist, shift).entropy();
                        proptest::prop_assert!($crate::misc::eq_or_close(shifted, manual, 1e-10),
                            "entropies differ: {} vs {}", shifted, manual);
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_scalable_mean {
        ($expr:expr) => {
            use proptest::prelude::*;

            proptest! {
                #[test]
                fn scalable_mean(scale in -100.0..100.0) {
                    let dist = $expr;
                    let scaled = dist.clone().scaled(scale);
                    let manual = Scaled::new(dist, scale);

                    let mean_scaled = scaled.mean();
                    let mean_manual = manual.mean();
                    match (mean_scaled, mean_manual) {
                        (Some(mean_scaled), Some(mean_manual)) => {
                            let mean_scaled: f64 = mean_scaled;
                            let mean_manual: f64 = mean_manual;
                            prop_assert!($crate::misc::eq_or_close(mean_scaled, mean_manual, 1e-10), "means differ: {} vs {}", mean_scaled, mean_manual);
                        }
                        (None, None) => {},
                        _ => {
                            prop_assert!(false, "Shifting should not affect existence of mean");
                        }
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_scalable_method {
        // Base case with no extension
        ($expr:expr, $ident:ident) => {
            test_scalable_method!($expr, $ident, );
        };

        // Main implementation
        ($expr:expr, $ident:ident, $($ext:ident)?) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<scalable_ $ident $(_ $ext)?>](scale in 1e-10..100.0) {
                        let dist = $expr;
                        let scaled = dist.clone().scaled_unchecked(scale).$ident();
                        let manual = $crate::prelude::Scaled::new_unchecked(dist, scale).$ident();

                        match (scaled, manual) {
                            (Some(scaled), Some(manual)) => {
                                let scaled: f64 = scaled;
                                let manual: f64 = manual;
                                proptest::prop_assert!($crate::misc::eq_or_close(scaled, manual, 1e-10),
                                    "{}s differ: {} vs {}", stringify!($ident), scaled, manual);
                            }
                            (None, None) => {},
                            _ => {
                                proptest::prop_assert!(false, "Scaling should not affect existence of {}",
                                    stringify!($ident));
                            }
                        }
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_scalable_density {
        ($expr:expr) => {
            test_scalable_density!($expr, );
        };

        ($expr:expr, $($ext:ident)?) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<scalable_density $(_ $ext)?>](y in -100.0..100.0, scale in 1e-10..100.0) {
                        let dist = $expr;
                        let scaled: f64 = dist.clone().scaled_unchecked(scale).ln_f(&y);
                        let manual: f64 = $crate::prelude::Scaled::new_unchecked(dist, scale).ln_f(&y);
                        proptest::prop_assert!($crate::misc::eq_or_close(scaled, manual, 1e-10),
                            "densities differ: {} vs {}", scaled, manual);
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_scalable_cdf {
        ($expr:expr) => {
            test_scalable_cdf!($expr, );
        };

        ($expr:expr, $($ext:ident)?) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<scalable_cdf $(_ $ext)?>](x in -100.0..100.0, scale in 1e-10..100.0) {
                        let dist = $expr;
                        let scaled: f64 = dist.clone().scaled_unchecked(scale).cdf(&x);
                        let manual: f64 = $crate::prelude::Scaled::new_unchecked(dist, scale).cdf(&x);
                        proptest::prop_assert!($crate::misc::eq_or_close(scaled, manual, 1e-10),
                            "cdfs differ: {} vs {}", scaled, manual);
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_scalable_invcdf {
        ($expr:expr) => {
            test_scalable_invcdf!($expr, );
        };

        ($expr:expr, $($ext:ident)?) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<scalable_invcdf $(_ $ext)?>](p in 0.0..1.0, scale in 1e-10..100.0) {
                        let dist = $expr;
                        let scaled: f64 = dist.clone().scaled_unchecked(scale).invcdf(p);
                        let manual: f64 = $crate::prelude::Scaled::new_unchecked(dist, scale).invcdf(p);
                        proptest::prop_assert!($crate::misc::eq_or_close(scaled, manual, 1e-10),
                            "invcdfs differ: {} vs {}", scaled, manual);
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! test_scalable_entropy {
        ($expr:expr) => {
            test_scalable_entropy!($expr, );
        };

        ($expr:expr, $($ext:ident)?) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<scalable_entropy $(_ $ext)?>](scale in 1e-10..100.0) {
                        let dist = $expr;
                        let scaled: f64 = dist.clone().scaled_unchecked(scale).entropy();
                        let manual: f64 = $crate::prelude::Scaled::new_unchecked(dist, scale).entropy();
                        proptest::prop_assert!($crate::misc::eq_or_close(scaled, manual, 1e-10),
                            "entropies differ: {} vs {}", scaled, manual);
                    }
                }
            }
        };
    }
}
