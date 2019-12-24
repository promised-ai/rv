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
mod ks;
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

pub use bernoulli::{Bernoulli, BernoulliError};
pub use beta::{Beta, BetaError};
pub use beta_binom::{BetaBinomial, BetaBinomialError};
pub use binomial::{Binomial, BinomialError};
pub use categorical::{Categorical, CategoricalError};
pub use cauchy::{Cauchy, CauchyError};
pub use chi_squared::{ChiSquared, ChiSquaredError};
pub use crp::{Crp, CrpError};
pub use dirichlet::{Dirichlet, DirichletError, SymmetricDirichlet};
pub use discrete_uniform::{DiscreteUniform, DiscreteUniformError};
pub use exponential::{Exponential, ExponentialError};
pub use gamma::{Gamma, GammaError};
pub use gaussian::{Gaussian, GaussianError};
pub use geometric::{Geometric, GeometricError};
pub use gev::{Gev, GevError};
pub use invgamma::{InvGamma, InvGammaError};
pub use ks::KSTwoAsymptotic;
pub use kumaraswamy::{Kumaraswamy, KumaraswamyError};
pub use laplace::{Laplace, LaplaceError};
pub use lognormal::{LogNormal, LogNormalError};
pub use mixture::{Mixture, MixtureError};
pub use mvg::{MvGaussian, MvGaussianError};
pub use niw::{NormalInvWishart, NormalInvWishartError};
pub use normal_gamma::{NormalGamma, NormalGammaError};
pub use pareto::{Pareto, ParetoError};
pub use poisson::{Poisson, PoissonError};
pub use students_t::{StudentsT, StudentsTError};
pub use uniform::{Uniform, UniformError};
pub use vonmises::{VonMises, VonMisesError};
pub use wishart::{InvWishart, InvWishartError};
