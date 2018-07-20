mod bernoulli;
mod beta;
mod categorical;
mod cauchy;
mod crp;
mod dirichlet;
mod exponential;
mod gamma;
mod gaussian;
mod invgamma;
mod laplace;
mod normal_gamma;
mod poisson;
mod suffstats;
mod uniform;

pub use self::bernoulli::Bernoulli;
pub use self::beta::Beta;
pub use self::categorical::Categorical;
pub use self::cauchy::Cauchy;
pub use self::crp::Crp;
pub use self::dirichlet::Dirichlet;
pub use self::exponential::Exponential;
pub use self::gamma::Gamma;
pub use self::gaussian::Gaussian;
pub use self::invgamma::InvGamma;
pub use self::laplace::Laplace;
pub use self::normal_gamma::NormalGamma;
pub use self::poisson::Poisson;
pub use self::suffstats::{
    BernoulliSuffStat, CategoricalSuffStat, GaussianSuffStat,
};
pub use self::uniform::Uniform;

extern crate num;
use self::num::traits::FromPrimitive;

pub trait CategoricalDatum:
    Sized + Into<usize> + Sync + Copy + FromPrimitive
{
}

impl<T> CategoricalDatum for T where
    T: Clone + Into<usize> + Sync + Copy + FromPrimitive
{}
