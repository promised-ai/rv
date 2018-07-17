use dist::{Bernoulli, Categorical, Gaussian};
use traits::DataOrSuffStat;

pub type BernoulliData<'a, X> = DataOrSuffStat<'a, X, Bernoulli>;
pub type CategoricalData<'a, X> = DataOrSuffStat<'a, X, Categorical>;
pub type GaussianData<'a, X> = DataOrSuffStat<'a, X, Gaussian>;
