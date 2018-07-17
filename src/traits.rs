extern crate rand;

use self::rand::Rng;

/// Random variable
///
/// Contains the minimal functionality that a random object must have to be
/// useful: a function defining the un-normalized density/mass at a point,
/// and functions to draw samples from the distribution.
pub trait Rv<X> {
    /// Un-normalized probability function
    fn f(&self, x: &X) -> f64 {
        self.ln_f(x).exp()
    }

    /// Un-normalized probability function
    fn ln_f(&self, x: &X) -> f64;

    /// The constant term in the PDF/PMF. Should not be a function of any of
    /// the parameters.
    fn normalizer(&self) -> f64 {
        self.ln_normalizer().exp()
    }

    /// The log of the constant term in the PDF/PMF. Should not be a function of
    /// any of the parameters.
    fn ln_normalizer(&self) -> f64;

    /// Single draw from the `Rv`
    fn draw<R: Rng>(&self, rng: &mut R) -> X;

    /// Multiple draws of the `Rv`
    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<X> {
        (0..n).map(|_| self.draw(&mut rng)).collect()
    }
}

/// Identifies the support of the Rv
pub trait Support<X> {
    /// Returns `true` if `x` is in the support of the `Rv`
    fn contains(&self, x: &X) -> bool;
}

/// Is a continuous probability distributions
pub trait ContinuousDistr<X>: Rv<X> + Support<X> {
    /// The value of the Probability Density Function (PDF) at `x`
    fn pdf(&self, x: &X) -> f64 {
        self.ln_pdf(x).exp()
    }

    /// The value of the log Probability Density Function (PDF) at `x`
    fn ln_pdf(&self, x: &X) -> f64 {
        if !self.contains(&x) {
            panic!("x not in support");
        }
        self.ln_f(x) - self.ln_normalizer()
    }
}

/// Has a cumulative distribution function (CDF)
pub trait Cdf<X>: Rv<X> {
    /// The value of the Cumulative Density Function at `x`
    fn cdf(&self, x: &X) -> f64;

    /// Survival function
    fn sf(&self, x: &X) -> f64 {
        1.0 - self.cdf(x)
    }
}

/// Has an inverse-CDF / quantile function
pub trait InverseCdf<X>: Rv<X> + Support<X> {
    /// The value of the `x` at the given probability in the CDF
    fn invcdf(&self, p: f64) -> X;

    /// Alias for `invcdf`
    fn quantile(&self, p: f64) -> X {
        self.invcdf(p)
    }
}

/// Is a discrete probability distribution
pub trait DiscreteDistr<X>: Rv<X> + Support<X> {
    fn pmf(&self, x: &X) -> f64 {
        self.ln_pmf(x).exp()
    }

    fn ln_pmf(&self, x: &X) -> f64 {
        if !self.contains(&x) {
            panic!("x not in support");
        }
        self.ln_f(x) - self.ln_normalizer()
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
    fn kl(&self, other: &Self) -> f64;

    /// Symmetrised divergence, KL(P|Q) + KL(Q|P)
    fn kl_sym(&self, other: &Self) -> f64 {
        self.kl(&other) + other.kl(&self)
    }
}

/// The data for this distribution can be summarized by a statistic
pub trait HasSuffStat<X> {
    type Stat: SuffStat<X>;
    fn empty_suffstat(&self) -> Self::Stat;
}

/// Is a sufccicent statistic for a distribution
pub trait SuffStat<X> {
    fn observe(&mut self, x: &X);
    fn forget(&mut self, x: &X);
}

pub enum DataOrSuffStat<'a, X, Fx>
where
    X: 'a,
    Fx: 'a + HasSuffStat<X>,
{
    Data(&'a Vec<X>),
    SuffStat(&'a Fx::Stat),
}

/// A prior on `Fx` that induces a posterior that is the same form as the prior
pub trait ConjugatePrior<X, Fx>: Rv<Fx>
where
    Fx: Rv<X> + HasSuffStat<X>,
{
    // TODO: Might it make sense to add an associated type `Posterior`, in the
    // event that the posterior is slightly different? For performance reasons
    // we might want the prior to be a SymmetricDirichlet type, which would
    // only have to store one parameter and wouldn't have to iterate over a
    // vetor multiple times to compute likelihoods. The posterior would also
    // be Dirichlet, but it wouldn't have such a computationally convenient
    // form.

    /// Computes the posterior distribution from the data
    fn posterior(&self, x: &DataOrSuffStat<X, Fx>) -> Self;

    // Log marginal likelihood
    fn ln_m(&self, x: &DataOrSuffStat<X, Fx>) -> f64;

    // Log posterior predictive of y given x
    fn ln_pp(&self, y: &X, x: &DataOrSuffStat<X, Fx>) -> f64;

    /// Marginal likelihood of x
    fn m(&self, x: &DataOrSuffStat<X, Fx>) -> f64 {
        self.ln_m(x).exp()
    }

    /// Posterior Predictive distribution
    fn pp(&self, y: &X, x: &DataOrSuffStat<X, Fx>) -> f64 {
        self.ln_pp(&y, x).exp()
    }
}
