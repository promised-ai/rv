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
mod betaprime;
mod binomial;
mod categorical;
mod cauchy;
mod chi_squared;
mod crp;
mod dirichlet;
mod discrete_uniform;
mod empirical;
mod exponential;
mod gamma;
mod gaussian;
mod geometric;
mod gev;
mod inv_chi_squared;
mod invgamma;
mod invgaussian;
mod ks;
mod kumaraswamy;
mod laplace;
mod lognormal;
mod mixture;
#[cfg(feature = "arraydist")]
mod mvg;
mod neg_binom;
#[cfg(feature = "arraydist")]
mod niw;
mod normal_gamma;
mod normal_inv_chi_squared;
mod normal_inv_gamma;
mod pareto;
mod poisson;
mod scaled_inv_chi_squared;
mod shifted;
mod skellam;
mod students_t;
mod uniform;
mod unit_powerlaw;
mod vonmises;
#[cfg(feature = "arraydist")]
mod wishart;

pub use bernoulli::{Bernoulli, BernoulliError};
pub use beta::{Beta, BetaError};
pub use beta_binom::{BetaBinomial, BetaBinomialError};
pub use betaprime::{BetaPrime, BetaPrimeError};
pub use binomial::{Binomial, BinomialError};
pub use categorical::{Categorical, CategoricalError};
pub use cauchy::{Cauchy, CauchyError};
pub use chi_squared::{ChiSquared, ChiSquaredError};
pub use crp::{Crp, CrpError};
pub use dirichlet::{Dirichlet, DirichletError, SymmetricDirichlet};
pub use discrete_uniform::{DiscreteUniform, DiscreteUniformError};
pub use empirical::Empirical;
pub use exponential::{Exponential, ExponentialError};
pub use gamma::{Gamma, GammaError};
pub use gaussian::{Gaussian, GaussianError};
pub use geometric::{Geometric, GeometricError};
pub use gev::{Gev, GevError};
pub use inv_chi_squared::{InvChiSquared, InvChiSquaredError};
pub use invgamma::{InvGamma, InvGammaError};
pub use invgaussian::{InvGaussian, InvGaussianError};
pub use ks::KsTwoAsymptotic;
pub use kumaraswamy::{Kumaraswamy, KumaraswamyError};
pub use laplace::{Laplace, LaplaceError};
pub use lognormal::{LogNormal, LogNormalError};
pub use mixture::{Mixture, MixtureError};
#[cfg(feature = "arraydist")]
pub use mvg::{MvGaussian, MvGaussianError};
pub use neg_binom::{NegBinomial, NegBinomialError};
#[cfg(feature = "arraydist")]
pub use niw::{NormalInvWishart, NormalInvWishartError};
pub use normal_gamma::{NormalGamma, NormalGammaError};
pub use normal_inv_chi_squared::{
    NormalInvChiSquared, NormalInvChiSquaredError,
};
pub use normal_inv_gamma::{NormalInvGamma, NormalInvGammaError};
pub use pareto::{Pareto, ParetoError};
pub use poisson::{Poisson, PoissonError};
pub use scaled_inv_chi_squared::{
    ScaledInvChiSquared, ScaledInvChiSquaredError,
};
pub use shifted::Shifted;
pub use skellam::{Skellam, SkellamError};
pub use students_t::{StudentsT, StudentsTError};
pub use uniform::{Uniform, UniformError};
pub use unit_powerlaw::{UnitPowerLaw, UnitPowerLawError};
pub use vonmises::{VonMises, VonMisesError};
#[cfg(feature = "arraydist")]
pub use wishart::{InvWishart, InvWishartError};
