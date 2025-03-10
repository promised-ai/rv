//! CDVM distribution over x in (0, m-1)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::misc::func::LogSumExp;  
use crate::traits::*;
use rand::Rng;
use std::f64;
use std::ops::Neg;       
use std::fmt;
use std::sync::OnceLock;
use crate::misc::ln_pflip;


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

    /// Mu must be in [0, modulus)
    MuOutOfRange { mu: f64 },
    
    /// Kappa must be non-negative
    KappaNegative { kappa: f64 },
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
        if mu < 0.0 || mu >= modulus as f64 {
            return Err(CdvmError::MuOutOfRange { mu });
        }
        if kappa < 0.0 {
            return Err(CdvmError::KappaNegative { kappa });
        }
        
        Ok(Cdvm {
            modulus,
            mu,
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
            let pi = std::f64::consts::PI;
            let m = self.modulus;
            let mu = self.mu;
            let kappa = self.kappa;
            // For CDVM, the normalization constant is just the von Mises normalizer
            // since the categorical probabilities already sum to 1
            (0..m).map(|r| {
                kappa * (((2 * r) as f64).mul_add(pi, -mu) / m as f64).cos()
            }).logsumexp().neg()
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

impl_display!(Cdvm);

impl std::error::Error for CdvmError {}

impl fmt::Display for CdvmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCategories { modulus } => {
                write!(f, "number of categories ({}) must be at least 2", modulus)
            }
            Self::MuOutOfRange { mu } => {
                write!(f, "mu ({}) must be in [0, modulus)", mu)
            }
            Self::KappaNegative { kappa } => {
                write!(f, "kappa ({}) must be non-negative", kappa)
            }
            
        }
    }
}

impl HasDensity<usize> for Cdvm {
    fn ln_f(&self, x: &usize) -> f64 {
        let pi = std::f64::consts::PI;
        let m = self.modulus;
        let mu = self.mu;
        let kappa = self.kappa;
        kappa.mul_add((((2 * x) as f64).mul_add(pi, -mu) / m as f64).cos(), self.get_log_norm_const())
    }
}

impl Support<usize> for Cdvm {
    fn supports(&self, x: &usize) -> bool {
        *x < self.modulus
    }
}

impl Sampleable<usize> for Cdvm {

    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        ln_pflip((0..self.modulus).map(|r| self.ln_f(&r)), true, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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

        // Invalid mu should fail
        assert!(matches!(
            Cdvm::new(3, 3.0, 1.0),
            Err(CdvmError::MuOutOfRange { mu: 3.0 })
        ));
        assert!(matches!(
            Cdvm::new(3, -1.0, 1.0),
            Err(CdvmError::MuOutOfRange { mu: -1.0 })
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

    #[proptest]
    #[test]
    fn ln_f_is_symmetric() {
        let cdvm = Cdvm::new(4, 2.0, 1.0).unwrap();
        let mu = cdvm.mu();
        let m = cdvm.modulus();

        // For points equidistant from mu, ln_f should be equal
        for i in 0..m/2 {
            let x1 = ((mu as usize + i) % m) as usize;
            let x2 = ((mu as usize + m - i) % m) as usize;
            assert::close(cdvm.ln_f(&x1), cdvm.ln_f(&x2), TOL);
        }
    }

    // #[test]
    // fn draw_test() {
    //     let mut rng = StdRng::seed_from_u64(42);
    //     let cdvm = Cdvm::new(4, 1.5, 2.0).unwrap();
        
    //     // Sample a large number of draws and check frequencies
    //     let n_samples = 1000;
    //     let mut counts = vec![0; cdvm.modulus()];
        
    //     for _ in 0..n_samples {
    //         let x = cdvm.draw(&mut rng);
    //         counts[x] += 1;
    //     }

    //     // All samples should be within support
    //     assert!(counts.iter().enumerate().all(|(i, _)| cdvm.supports(&i)));

    //     // Frequencies should roughly follow the probability distribution
    //     // We use chi-square test to verify this
    //     let expected_probs: Vec<f64> = (0..cdvm.modulus())
    //         .map(|i| (cdvm.ln_f(&i)).exp())
    //         .collect();

    //     let sum: f64 = expected_probs.iter().sum();
    //     let normalized_probs: Vec<f64> = expected_probs.iter()
    //         .map(|&p| p / sum)
    //         .collect();

    //     // Convert counts to f64 for chi-square test
    //     let observed: Vec<u32> = counts.iter()
    //         .map(|&x| x as u32)
    //         .collect();

    //     let passes = (0..N_TRIES).fold(0, |acc, _| {
    //         let (_, p) = x2_test(&observed, &normalized_probs);
    //         if p > X2_PVAL {
    //             acc + 1
    //         } else {
    //             acc
    //         }
    //     });

    //     assert!(passes > 0);
    // }

    #[test]
    fn parameterized_trait() {
        let original = Cdvm::new(3, 1.5, 1.0).unwrap();
        let params = original.emit_params();
        let reconstructed = Cdvm::from_params(params);
        
        assert_eq!(original, reconstructed);
    }
}

