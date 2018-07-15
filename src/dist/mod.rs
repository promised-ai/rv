pub mod bernoulli;
pub mod beta;
pub mod categorical;
pub mod dirichlet;
pub mod gamma;
pub mod gaussian;
pub mod invgamma;

pub use self::bernoulli::Bernoulli;
pub use self::beta::Beta;
pub use self::categorical::Categorical;
pub use self::dirichlet::Dirichlet;
pub use self::gamma::Gamma;
pub use self::gaussian::Gaussian;
pub use self::invgamma::InvGamma;
