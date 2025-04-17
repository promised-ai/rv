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
mod cdvm;
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
mod scaled;
mod scaled_inv_chi_squared;
mod shifted;
mod skellam;
mod students_t;
mod uniform;
mod unit_powerlaw;
mod vonmises;
#[cfg(feature = "arraydist")]
mod wishart;

pub use bernoulli::{Bernoulli, BernoulliError, BernoulliParameters};
pub use beta::{Beta, BetaError, BetaParameters};
pub use beta_binom::{BetaBinomial, BetaBinomialError, BetaBinomialParameters};
pub use betaprime::{BetaPrime, BetaPrimeError, BetaPrimeParameters};
pub use binomial::{Binomial, BinomialError, BinomialParameters};
pub use categorical::{Categorical, CategoricalError, CategoricalParameters};
pub use cauchy::{Cauchy, CauchyError, CauchyParameters};
pub use cdvm::{Cdvm, CdvmError, CdvmParameters};
pub use chi_squared::{ChiSquared, ChiSquaredError, ChiSquaredParameters};
pub use crp::{Crp, CrpError, CrpParameters};
pub use dirichlet::{
    Dirichlet, DirichletError, SymmetricDirichlet, SymmetricDirichletParameters,
};
pub use discrete_uniform::{
    DiscreteUniform, DiscreteUniformError, DiscreteUniformParameters,
};
pub use empirical::Empirical;
pub use exponential::{Exponential, ExponentialError, ExponentialParameters};
pub use gamma::{Gamma, GammaError, GammaParameters};
pub use gaussian::{Gaussian, GaussianError, GaussianParameters};
pub use geometric::{Geometric, GeometricError, GeometricParameters};
pub use gev::{Gev, GevError, GevParameters};
pub use inv_chi_squared::{
    InvChiSquared, InvChiSquaredError, InvChiSquaredParameters,
};
pub use invgamma::{InvGamma, InvGammaError, InvGammaParameters};
pub use invgaussian::{InvGaussian, InvGaussianError, InvGaussianParameters};
pub use ks::KsTwoAsymptotic;
pub use kumaraswamy::{Kumaraswamy, KumaraswamyError, KumaraswamyParameters};
pub use laplace::{Laplace, LaplaceError, LaplaceParameters};
pub use lognormal::{LogNormal, LogNormalError, LogNormalParameters};
pub use mixture::{Mixture, MixtureError, MixtureParameters};
#[cfg(feature = "arraydist")]
pub use mvg::{MvGaussian, MvGaussianError, MvGaussianParameters};
pub use neg_binom::{NegBinomial, NegBinomialError, NegBinomialParameters};
#[cfg(feature = "arraydist")]
pub use niw::{
    NormalInvWishart, NormalInvWishartError, NormalInvWishartParameters,
};
pub use normal_gamma::{NormalGamma, NormalGammaError, NormalGammaParameters};
pub use normal_inv_chi_squared::{
    NormalInvChiSquared, NormalInvChiSquaredError,
    NormalInvChiSquaredParameters,
};
pub use normal_inv_gamma::{
    NormalInvGamma, NormalInvGammaError, NormalInvGammaParameters,
};
pub use pareto::{Pareto, ParetoError, ParetoParameters};
pub use poisson::{Poisson, PoissonError, PoissonParameters};
pub use scaled::{Scaled, ScaledError, ScaledParameters};
pub use scaled_inv_chi_squared::{
    ScaledInvChiSquared, ScaledInvChiSquaredError,
    ScaledInvChiSquaredParameters,
};
pub use shifted::{Shifted, ShiftedError, ShiftedParameters};
pub use skellam::{Skellam, SkellamError, SkellamParameters};
pub use students_t::{StudentsT, StudentsTError, StudentsTParameters};
pub use uniform::{Uniform, UniformError, UniformParameters};
pub use unit_powerlaw::{
    UnitPowerLaw, UnitPowerLawError, UnitPowerLawParameters,
};
pub use vonmises::{VonMises, VonMisesError, VonMisesParameters};
#[cfg(feature = "arraydist")]
pub use wishart::{InvWishart, InvWishartError, InvWishartParameters};
