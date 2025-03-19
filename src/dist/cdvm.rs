//! CDVM distribution over x in (0, m-1)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::CdvmSuffStat;
use crate::impl_display;
use crate::misc::func::LogSumExp;
use crate::misc::ln_pflip;
use crate::traits::*;
use rand::Rng;
use std::f64;
use std::fmt;
use std::sync::OnceLock;

// TODO: This can be *much* more efficient if we replace the modulus with
// something like this. In particular, the suffstat would only need quick
// lookups and additions, with no trig functions
//
// #[derive(Debug, Clone,
//     PartialEq)] pub struct CdvmModulus { m: usize, twopi_over_m: f64, sines:
//     Vec<f64>, cosines: Vec<f64>, }

// impl CdvmModulus {
//     fn new(m: usize) -> Self {
//         let twopi_over_m = 2.0 * std::f64::consts::PI / m as f64;
//         let sines = (0..m).map(|x| (twopi_over_m * (x as f64)).sin()).collect();
//         let cosines = (0..m).map(|x| (twopi_over_m * (x as f64)).cos()).collect();
//         Self {
//             m,
//             twopi_over_m,
//             sines,
//             cosines,
//         }
//     }
// }

/// [CDVM distribution](https://arxiv.org/pdf/2009.05437),
/// A unimodal distribution over x in (0, m-1) where m is the number of categories.
///
/// Note that in while the paper uses μ ∈ [0, 2π), we use μ ∈ [0, m)
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

    /// Cached 2π/m
    #[cfg_attr(feature = "serde1", serde(skip))]
    twopi_over_m: OnceLock<f64>,
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

impl Cdvm {
    /// Create a new CDVM distribution
    ///
    /// # Arguments
    /// * `mu` - mean direction (must be in [0, modulus))
    /// * `kappa` - concentration (must be non-negative)
    /// * `modulus` - Number of categories
    pub fn new(mu: f64, kappa: f64, modulus: usize) -> Result<Self, CdvmError> {
        // Validate parameters
        if modulus < 2 {
            return Err(CdvmError::InvalidCategories { modulus });
        }
        if kappa < 0.0 {
            return Err(CdvmError::KappaNegative { kappa });
        }

        Ok(Cdvm::new_unchecked(mu, kappa, modulus))
    }

    /// Creates a new CDVM without checking whether the parameters are valid.
    #[inline]
    pub fn new_unchecked(mu: f64, kappa: f64, modulus: usize) -> Self {
        Cdvm {
            modulus,
            mu,
            kappa,
            log_norm_const: OnceLock::new(),
            twopi_over_m: OnceLock::new(),
        }
    }

    fn cdvm_kernel(&self, x: usize) -> f64 {
        self.kappa * ((self.twopi_over_m() * (x as f64 - self.mu)).cos())
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

    /// Get the cached 2π/m
    pub fn twopi_over_m(&self) -> f64 {
        *self
            .twopi_over_m
            .get_or_init(|| 2.0 * std::f64::consts::PI / self.modulus as f64)
    }

    /// Compute or fetch cached normalization constant
    fn log_norm_const(&self) -> f64 {
        *self.log_norm_const.get_or_init(|| {
            // For CDVM, the normalization constant is just the von Mises normalizer
            // since the categorical probabilities already sum to 1
            (0..self.modulus).map(|x| self.cdvm_kernel(x)).logsumexp()
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
        Self::new(params.mu, params.kappa, params.modulus).unwrap()
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
        self.cdvm_kernel(*x) - self.log_norm_const()
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

impl HasSuffStat<usize> for Cdvm {
    type Stat = CdvmSuffStat;

    fn empty_suffstat(&self) -> Self::Stat {
        CdvmSuffStat::new(self.modulus)
    }

    fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
        let twopimu_over_m = self.mu * self.twopi_over_m();
        // TODO: Should we cache twopimu_over_m.cos() and twopimu_over_m.sin()?

        self.kappa
            * (stat.sum_cos() * twopimu_over_m.cos()
                + stat.sum_sin() * twopimu_over_m.sin())
            - stat.n() as f64 * self.log_norm_const()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn new_should_validate_parameters() {
        // Valid parameters should work
        assert!(Cdvm::new(1.5, 1.0, 3).is_ok());

        // Invalid modulus should fail
        assert!(matches!(
            Cdvm::new(1.5, 1.0, 1),
            Err(CdvmError::InvalidCategories { modulus: 1 })
        ));

        // Invalid kappa should fail
        assert!(matches!(
            Cdvm::new(1.5, -1.0, 3),
            Err(CdvmError::KappaNegative { kappa: -1.0 })
        ));
    }

    #[test]
    fn supports_correct_range() {
        let cdvm = Cdvm::new(1.5, 1.0, 4).unwrap();

        assert!(cdvm.supports(&0));
        assert!(cdvm.supports(&1));
        assert!(cdvm.supports(&2));
        assert!(cdvm.supports(&3));
        assert!(!cdvm.supports(&4));
    }

    proptest! {
        #[test]
        fn ln_f_symmetry(
            m in 3..100_usize,
            mu in 0.0..100_f64,
            kappa in 0.1..50.0_f64,
            x in 0..100_usize
        ) {
            let mu = mu % (m as f64);
            let cdvm1 = Cdvm::new(mu, kappa, m).unwrap();
            let cdvm2 = Cdvm::new((m as f64) - mu, kappa, m).unwrap();

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
            m in 3..100_usize,
            mu in 0.0..100_f64,
            kappa in 0.1..50.0_f64,
        ) {
            let cdvm = Cdvm::new(mu, kappa, m).unwrap();

            // For the density to be normalized, the logsum should be zero
            let logsum = (0..m).map(|x| cdvm.ln_f(&x)).logsumexp();
            prop_assert!((logsum).abs() < TOL,
                "density not normalized for m={}, mu={}, kappa={}, logsum={}", m, mu, kappa, logsum);
        }
    }

    proptest! {
        #[test]
        fn wrap_around_invariance(
            m in 3..100_usize,
            mu in 0.0..100_f64,
            kappa in 0.1..50.0_f64,
            x in 0..100_usize,
        ) {
            let mu = mu % (m as f64);
            let x = x % m;
            let cdvm = Cdvm::new(mu, kappa, m).unwrap();
            prop_assert!((cdvm.ln_f(&x) - cdvm.ln_f(&(x + m))).abs() < TOL,
                "ln_f not invariant to wrap-around for m={}, mu={}, kappa={}, x={}", m, mu, kappa, x);
        }
    }

    #[test]
    fn parameterized_trait() {
        let original = Cdvm::new(1.5, 1.0, 3).unwrap();
        let params = original.emit_params();
        let reconstructed = Cdvm::from_params(params);

        assert_eq!(original, reconstructed);
    }

    proptest! {
        #[test]
        fn ln_f_matches_ln_f_stat(
            m in 3..100_usize,
            mu in 0.0..100_f64,
            kappa in 0.1..50.0_f64,
            xs in prop::collection::vec(0..100_usize, 1..20),
        ) {
            let mu = mu % (m as f64);
            let xs: Vec<usize> = xs.into_iter().map(|x| x % m).collect();
            let cdvm = Cdvm::new(mu, kappa, m).unwrap();

            // Calculate ln_f for each x and sum them
            let ln_f_sum: f64 = xs.iter().map(|x| cdvm.ln_f(x)).sum();

            // Create sufficient statistics from the data
            let stat = CdvmSuffStat::from_data(m, &xs);

            // Get ln_f_stat
            let ln_f_stat = cdvm.ln_f_stat(&stat);

            // They should be equal
            assert!((ln_f_sum - ln_f_stat).abs() < TOL,
                "ln_f_sum ({}) != ln_f_stat ({}) for m={}, mu={}, kappa={}, xs={:?}",
                ln_f_sum, ln_f_stat, m, mu, kappa, xs);
        }
    }
}
