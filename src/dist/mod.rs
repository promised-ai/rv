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
mod discrete_uniform;
mod exponential;
mod gamma;
mod gaussian;
mod geometric;
mod gev;
mod invgamma;
mod kumaraswamy;
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

pub use bernoulli::Bernoulli;
pub use beta::Beta;
pub use beta_binom::BetaBinomial;
pub use binomial::Binomial;
pub use categorical::Categorical;
pub use cauchy::Cauchy;
pub use chi_squared::ChiSquared;
pub use crp::Crp;
pub use dirichlet::{Dirichlet, SymmetricDirichlet};
pub use discrete_uniform::DiscreteUniform;
pub use exponential::Exponential;
pub use gamma::Gamma;
pub use gaussian::Gaussian;
pub use geometric::Geometric;
pub use gev::Gev;
pub use invgamma::InvGamma;
pub use kumaraswamy::Kumaraswamy;
pub use laplace::Laplace;
pub use lognormal::LogNormal;
pub use mixture::Mixture;
pub use mvg::MvGaussian;
pub use niw::NormalInvWishart;
pub use normal_gamma::NormalGamma;
pub use pareto::Pareto;
pub use poisson::Poisson;
pub use students_t::StudentsT;
pub use uniform::Uniform;
pub use vonmises::VonMises;
pub use wishart::InvWishart;
