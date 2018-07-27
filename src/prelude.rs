//! Re-imports for convenience
#[doc(no_inline)]
pub use data::DataOrSuffStat;
#[doc(no_inline)]
pub use dist::*;
#[doc(no_inline)]
pub use traits::*;

pub type BernoulliData<'a, X> = DataOrSuffStat<'a, X, Bernoulli>;
pub type CategoricalData<'a, X> = DataOrSuffStat<'a, X, Categorical>;
pub type GaussianData<'a, X> = DataOrSuffStat<'a, X, Gaussian>;
