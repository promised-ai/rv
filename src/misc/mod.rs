//! Random utilities
pub mod bessel;
pub(crate) mod entropy;
mod func;
mod ks;
mod legendre;
#[cfg(feature = "arraydist")]
mod mardia;
mod quad;
mod seq;
mod x2;

pub use func::*;
pub use ks::*;
pub use legendre::*;
#[cfg(feature = "arraydist")]
pub use mardia::mardia;
pub use quad::*;
pub use quad::*;
pub use seq::*;
pub use x2::x2_test;
