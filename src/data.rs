use dist::Bernoulli;
/// Type aliases for common data types for priors
use traits::DataOrSuffStat;

pub type BernoulliData<'a, X> = DataOrSuffStat<'a, X, Bernoulli>;
