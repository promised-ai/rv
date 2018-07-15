use dist::{Bernoulli, Categorical};
use traits::DataOrSuffStat;

pub type BernoulliData<'a, X> = DataOrSuffStat<'a, X, Bernoulli>;
pub type CategoricalData<'a, X> = DataOrSuffStat<'a, X, Categorical>;
