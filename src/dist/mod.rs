//! Probability distributions
//!
//! The distributions fall into three categories:
//!
//! 1. **Discrete** distributions assign probability to countable values.
//! 2. **Continuous** distributions assign probability to uncountable values
//!    on a continuum.
//! 3. **Prior** distributions assign probability to other probability
//!    distributions.
mod bernoulli;
mod beta;
mod beta_binom;
mod binomial;
mod categorical;
mod cauchy;
mod chi_squared;
mod crp;
mod dirichlet;
mod exponential;
mod gamma;
mod gaussian;
mod geometric;
mod gev;
mod invgamma;
mod laplace;
mod lognormal;
mod mixture;
mod mvg;
mod niw;
mod normal_gamma;
mod pareto;
mod poisson;
mod students_t;
mod uniform;
mod vonmises;
mod wishart;

pub use self::bernoulli::Bernoulli;
pub use self::beta::Beta;
pub use self::beta_binom::BetaBinomial;
pub use self::binomial::Binomial;
pub use self::categorical::Categorical;
pub use self::cauchy::Cauchy;
pub use self::chi_squared::ChiSquared;
pub use self::crp::Crp;
pub use self::dirichlet::{Dirichlet, SymmetricDirichlet};
pub use self::exponential::Exponential;
pub use self::gamma::Gamma;
pub use self::gaussian::Gaussian;
pub use self::geometric::Geometric;
pub use self::gev::Gev;
pub use self::invgamma::InvGamma;
pub use self::laplace::Laplace;
pub use self::lognormal::LogNormal;
pub use self::mixture::Mixture;
pub use self::mvg::MvGaussian;
pub use self::niw::NormalInvWishart;
pub use self::normal_gamma::NormalGamma;
pub use self::pareto::Pareto;
pub use self::poisson::Poisson;
pub use self::students_t::StudentsT;
pub use self::uniform::Uniform;
pub use self::vonmises::VonMises;
pub use self::wishart::InvWishart;
