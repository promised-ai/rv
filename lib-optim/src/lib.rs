//! A library for optimization algorithms in Rust

pub mod bfgs;
pub mod line_search;
pub mod root_finding;
//pub mod lbfgsb;

pub type Result<T> = std::result::Result<T, OptimizeError>;

#[derive(Debug, PartialEq, Hash)]
pub enum OptimizeError {
    /// During optimization, the evaluated point was to become numerically unstable
    NumericalDivergence,
    /// Maximum number of iterations reached in attempt to minimize.
    MaxIterationReached,
    /// A rounding error which can cause runaway was encountered.
    RoundingError,
}
