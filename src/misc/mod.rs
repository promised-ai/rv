//! Random utilities
pub mod bessel;
mod func;
mod ks;
mod mardia;
mod seq;
mod x2;

pub use self::func::*;
pub use self::ks::ks_test;
pub use self::mardia::mardia;
pub use self::seq::*;
pub use self::x2::x2_test;
