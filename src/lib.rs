//! Provides random variables for probabilistic modeling.
//!
//! # Examples
//!
//! For more examples, make sure to check out the `examples` directory.
//!
//! ## Conjugate analysis of coin flips
//!
//! ```rust
//! extern crate rand;
//! extern crate rv;
//!
//! use rand::Rng;
//! use rv::prelude::*;
//!
//! fn main() {
//!     let mut rng = rand::thread_rng();
//!
//!     // A sequence of observations
//!     let flips = vec![true, false, true, true, true, false, true];
//!
//!     // Construct the Jeffreys prior of Beta(0.5, 0.5)
//!     let prior = Beta::jeffreys();
//!
//!     // Packages the data in a wrapper that marks it as having come from
//!     // Bernoulli trials.
//!     let obs: BernoulliData<bool> = DataOrSuffStat::Data(&flips);
//!
//!     // Generate the posterior distributoin P(θ|x); the distribution of
//!     // probable coin weights
//!     let posterior: Beta = prior.posterior(&obs);
//!     
//!     // What is the probability that the next flip would come up heads
//!     // (true) given the observed flips (posterior predictive)?
//!     let p_heads = prior.pp(&true, &obs);
//! }
//! ```
#[macro_use]
extern crate serde_derive;

pub mod consts;
pub mod data;
pub mod dist;
pub mod model;
pub mod partition;
pub mod prelude;
mod priors;
pub mod traits;
pub mod utils;
