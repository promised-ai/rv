//! CDVM distribution over x in (0, m-1)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::misc::func::LogSumExp;
use crate::misc::ln_pflip;
use crate::traits::*;
use rand::Rng;
use std::f64;
use std::fmt;
use std::ops::Neg;
use std::sync::OnceLock;

/// [CDVM distribution](https://arxiv.org/pdf/2009.05437),
/// A distribution over x in (0, m-1) where m is the number of categories.
///
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Cdvm {
    /// Number of categories
    modulus: usize,

    /// mean direction (μ)
    mu: f64,

    /// concentration parameter (κ)
    kappa: f64,

    /// Cached log-normalization constant
    #[cfg_attr(feature = "serde1", serde(skip))]
    log_norm_const: OnceLock<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CdvmParameters {
    pub modulus: usize,
    pub mu: f64,
    pub kappa: f64,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum CdvmError {
    /// The number of categories is less than 2
    InvalidCategories { modulus: usize },

    /// Kappa must be non-negative
    KappaNegative { kappa: f64 },
}

fn cdvm_kernel(modulus: usize, mu: f64, kappa: f64, x: usize) -> f64 {
    let twopi = std::f64::consts::TAU;
    kappa * (twopi * (x as f64 - mu) / modulus as f64).cos()
}

impl Cdvm {
    /// Create a new CDVM distribution
    ///
    /// # Arguments
    /// * `modulus` - Number of categories
    /// * `mu` - mean direction (must be in [0, modulus))
    /// * `kappa` - concentration (must be non-negative)
    pub fn new(modulus: usize, mu: f64, kappa: f64) -> Result<Self, CdvmError> {
        // Validate parameters
        if modulus < 2 {
            return Err(CdvmError::InvalidCategories { modulus });
        }
        if kappa < 0.0 {
            return Err(CdvmError::KappaNegative { kappa });
        }

        Ok(Cdvm {
            modulus,
            mu: mu % modulus as f64,
            kappa,
            log_norm_const: OnceLock::new(),
        })
    }

    /// Get the number of categories
    pub fn modulus(&self) -> usize {
        self.modulus
    }

    /// Get the von Mises mean direction
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Get the von Mises concentration parameter
    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Compute or fetch cached normalization constant
    fn get_log_norm_const(&self) -> f64 {
        *self.log_norm_const.get_or_init(|| {
            let m = self.modulus;
            let mu = self.mu;
            let kappa = self.kappa;
            // For CDVM, the normalization constant is just the von Mises normalizer
            // since the categorical probabilities already sum to 1
            (0..m)
                .map(|r| cdvm_kernel(m, mu, kappa, r))
                .logsumexp()
                .neg()
        })
    }
}

impl Parameterized for Cdvm {
    type Parameters = CdvmParameters;

    fn emit_params(&self) -> Self::Parameters {
        CdvmParameters {
            modulus: self.modulus,
            mu: self.mu,
            kappa: self.kappa,
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new(params.modulus, params.mu, params.kappa).unwrap()
    }
}

impl PartialEq for Cdvm {
    fn eq(&self, other: &Cdvm) -> bool {
        self.modulus == other.modulus
            && self.mu == other.mu
            && self.kappa == other.kappa
    }
}

impl From<&Cdvm> for String {
    fn from(cdvm: &Cdvm) -> String {
        format!(
            "CDVM(modulus: {}, μ: {}, κ: {})",
            cdvm.modulus, cdvm.mu, cdvm.kappa
        )
    }
}

impl Mean<f64> for Cdvm {
    fn mean(&self) -> Option<f64> {
        Some(self.mu)
    }
}

impl Mode<usize> for Cdvm {
    fn mode(&self) -> Option<usize> {
        Some(self.mu.round() as usize)
    }
}

impl_display!(Cdvm);

impl std::error::Error for CdvmError {}

impl fmt::Display for CdvmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCategories { modulus } => {
                write!(
                    f,
                    "number of categories ({}) must be at least 2",
                    modulus
                )
            }
            Self::KappaNegative { kappa } => {
                write!(f, "kappa ({}) must be non-negative", kappa)
            }
        }
    }
}

impl HasDensity<usize> for Cdvm {
    fn ln_f(&self, x: &usize) -> f64 {
        let m = self.modulus;
        let mu = self.mu;
        let kappa = self.kappa;
        cdvm_kernel(m, mu, kappa, *x) + self.get_log_norm_const()
    }
}

impl Support<usize> for Cdvm {
    fn supports(&self, x: &usize) -> bool {
        *x < self.modulus
    }
}

// TODO: We should be able to speed this up by using an early-exit approach and
// selecting points in the right order (close to mean first)
impl Sampleable<usize> for Cdvm {
    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        ln_pflip((0..self.modulus).map(|r| self.ln_f(&r)), true, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::f64::consts::PI;

    const TOL: f64 = 1E-12;
    const N_TRIES: usize = 5;
    const X2_PVAL: f64 = 0.001;

    #[test]
    fn new_should_validate_parameters() {
        // Valid parameters should work
        assert!(Cdvm::new(3, 1.5, 1.0).is_ok());

        // Invalid modulus should fail
        assert!(matches!(
            Cdvm::new(1, 0.5, 1.0),
            Err(CdvmError::InvalidCategories { modulus: 1 })
        ));

        // Invalid kappa should fail
        assert!(matches!(
            Cdvm::new(3, 1.5, -1.0),
            Err(CdvmError::KappaNegative { kappa: -1.0 })
        ));
    }

    #[test]
    fn supports_correct_range() {
        let cdvm = Cdvm::new(4, 1.5, 1.0).unwrap();

        assert!(cdvm.supports(&0));
        assert!(cdvm.supports(&1));
        assert!(cdvm.supports(&2));
        assert!(cdvm.supports(&3));
        assert!(!cdvm.supports(&4));
    }

    proptest! {
        #[test]
        fn ln_f_symmetry(
            m in 3..100usize,
            mu in 0.0..100f64,
            kappa in 0.1..50.0f64,
            x in 0..100usize
        ) {
            let mu = mu % (m as f64);
            let cdvm1 = Cdvm::new(m, mu, kappa).unwrap();
            let cdvm2 = Cdvm::new(m, (m as f64) - mu, kappa).unwrap();

            let x1 = x % m;
            let x2 = m - x1;

            let lnf1 = cdvm1.ln_f(&x1);
            let lnf2 = cdvm2.ln_f(&x2);
            prop_assert!((lnf1 - lnf2).abs() < TOL,
                "ln_f not symmetric for m={}, mu={}, kappa={}, x={}, lnf1={}, lnf2={}", m, mu, kappa, x, lnf1, lnf2);
        }
    }

    proptest! {
        #[test]
        fn density_is_normalized(
            m in 3..100usize,
            mu in 0.0..100f64,
            kappa in 0.1..50.0f64,
        ) {
            let cdvm = Cdvm::new(m, mu, kappa).unwrap();

            // For the density to be normalized, the logsum should be zero
            let logsum = (0..m).map(|x| cdvm.ln_f(&x)).logsumexp();
            prop_assert!((logsum).abs() < TOL,
                "density not normalized for m={}, mu={}, kappa={}, logsum={}", m, mu, kappa, logsum);
        }
    }

    proptest! {
        #[test]
        fn wrap_around_invariance(
            m in 3..100usize,
            mu in 0.0..100f64,
            kappa in 0.1..50.0f64,
            x in 0..100usize,
        ) {
            let mu = mu % (m as f64);
            let x = x % m;
            let cdvm = Cdvm::new(m, mu, kappa).unwrap();
            prop_assert!((cdvm.ln_f(&x) - cdvm.ln_f(&(x + m))).abs() < TOL,
                "ln_f not invariant to wrap-around for m={}, mu={}, kappa={}, x={}", m, mu, kappa, x);
        }
    }

    #[test]
    fn parameterized_trait() {
        let original = Cdvm::new(3, 1.5, 1.0).unwrap();
        let params = original.emit_params();
        let reconstructed = Cdvm::from_params(params);

        assert_eq!(original, reconstructed);
    }
}
