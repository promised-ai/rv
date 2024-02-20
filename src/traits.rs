//! Trait definitions
use crate::data::DataOrSuffStat;
use crate::suffstat_traits::HasSuffStat;
use rand::Rng;

/// Random variable
///
/// Contains the minimal functionality that a random object must have to be
/// useful: a function defining the un-normalized density/mass at a point,
/// and functions to draw samples from the distribution.
pub trait Rv<X> {
    /// Probability function
    ///
    /// # Example
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::Rv;
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
    /// use rv::traits::Rv;
    ///
    /// let g = Gaussian::standard();
    /// assert!(g.ln_f(&0.0_f64) > g.ln_f(&0.1_f64));
    /// assert!(g.ln_f(&0.0_f64) > g.ln_f(&-0.1_f64));
    /// ```
    fn ln_f(&self, x: &X) -> f64;

    /// Single draw from the `Rv`
    ///
    /// # Example
    ///
    /// Flip a coin
    ///
    /// ```
    /// use rv::dist::Bernoulli;
    /// use rv::traits::Rv;
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
    /// use rv::traits::Rv;
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
    /// use rv::traits::Rv;
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
    /// use rv::traits::Rv;
    /// use rv::dist::Gamma;
    ///
    /// let mut rng = rand::thread_rng();
    ///
    /// let gamma = Gamma::new(2.0, 1.0).unwrap();
    ///
    /// let n = 1_000_000_usize;
    /// let mean = <Gamma as Rv<f64>>::sample_stream(&gamma, &mut rng)
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
pub trait ContinuousDistr<X>: Rv<X> + Support<X> {
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
        self.ln_pdf(x).exp()
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
    /// assert_eq!(f, std::f64::NEG_INFINITY);
    /// ```
    fn ln_pdf(&self, x: &X) -> f64 {
        if self.supports(x) {
            self.ln_f(x)
        } else {
            std::f64::NEG_INFINITY
        }
    }
}

/// Has a cumulative distribution function (CDF)
pub trait Cdf<X>: Rv<X> {
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
pub trait InverseCdf<X>: Rv<X> + Support<X> {
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
            std::f64::NEG_INFINITY
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

    /// Symmetrised divergence, KL(P|Q) + KL(Q|P)
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
/// use rv::traits::ConjugatePrior;
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
/// # use rv::traits::ConjugatePrior;
/// use rv::traits::Rv;
/// use rv::suffstat_traits::SuffStat;
/// use rv::dist::{Categorical, SymmetricDirichlet};
/// use rv::data::{CategoricalSuffStat, DataOrSuffStat};
/// use std::time::Instant;
///
/// let ncats = 10;
/// let symdir = SymmetricDirichlet::jeffreys(ncats).unwrap();
/// let mut suffstat = CategoricalSuffStat::new(ncats);
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
///     let t_start = Instant::now();
///     let cache = symdir.ln_pp_cache(&stat);
///     // Argmax
///     let k_max = (0..ncats).fold((0, std::f64::NEG_INFINITY), |(ix, f), y| {
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
///     let k_max = (0..ncats).fold((0, std::f64::NEG_INFINITY), |(ix, f), y| {
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
pub trait ConjugatePrior<X, Fx>: Rv<Fx>
where
    Fx: Rv<X> + HasSuffStat<X>,
{
    /// Type of the posterior distribution
    type Posterior: Rv<Fx>;
    /// Type of the `ln_m` cache
    type LnMCache;
    /// Type of the `ln_pp` cache
    type LnPpCache;

    /// Computes the posterior distribution from the data
    fn posterior(&self, x: &DataOrSuffStat<X, Fx>) -> Self::Posterior;

    /// Compute the cache for the log marginal likelihood.
    fn ln_m_cache(&self) -> Self::LnMCache;

    /// Log marginal likelihood with supplied cache.
    fn ln_m_with_cache(
        &self,
        cache: &Self::LnMCache,
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
    fn ln_pp_cache(&self, x: &DataOrSuffStat<X, Fx>) -> Self::LnPpCache;

    /// Log posterior predictive of y given x with supplied ln(norm)
    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &X) -> f64;

    /// Log posterior predictive of y given x
    fn ln_pp(&self, y: &X, x: &DataOrSuffStat<X, Fx>) -> f64 {
        let cache = self.ln_pp_cache(x);
        self.ln_pp_with_cache(&cache, y)
    }

    /// Marginal likelihood of x
    fn m(&self, x: &DataOrSuffStat<X, Fx>) -> f64 {
        self.ln_m(x).exp()
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
