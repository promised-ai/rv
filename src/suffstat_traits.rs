/// The data for this distribution can be summarized by a statistic
pub trait HasSuffStat<X> {
    type Stat: SuffStat<X>;

    fn empty_suffstat() -> Self::Stat;

    /// Return the log likelihood for the data represented by the sufficient
    /// statistic.
    fn ln_f_stat(&self, stat: &Self::Stat) -> f64;
}

/// Is a [sufficient statistic](https://en.wikipedia.org/wiki/Sufficient_statistic) for a
/// distribution.
///
/// # Examples
///
/// Basic suffstat useage.
///
/// ```
/// use rv::data::BernoulliSuffStat;
/// use rv::suffstat_traits::SuffStat;
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
/// use rv::suffstat_traits::SuffStat;
/// use rv::traits::ConjugatePrior;
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
}
