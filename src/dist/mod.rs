mod bernoulli;
mod beta;
mod categorical;
mod cauchy;
mod chi_squared;
mod crp;
mod dirichlet;
mod exponential;
mod gamma;
mod gaussian;
mod invgamma;
mod laplace;
mod normal_gamma;
mod poisson;
mod students_t;
mod uniform;

pub use self::bernoulli::Bernoulli;
pub use self::beta::Beta;
pub use self::categorical::Categorical;
pub use self::cauchy::Cauchy;
pub use self::chi_squared::ChiSquared;
pub use self::crp::Crp;
pub use self::dirichlet::Dirichlet;
pub use self::exponential::Exponential;
pub use self::gamma::Gamma;
pub use self::gaussian::Gaussian;
pub use self::invgamma::InvGamma;
pub use self::laplace::Laplace;
pub use self::normal_gamma::NormalGamma;
pub use self::poisson::Poisson;
pub use self::students_t::StudentsT;
pub use self::uniform::Uniform;
