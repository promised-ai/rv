//! CDVM distribution over x in (0, m-1)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::traits::*;
use rand::Rng;
use std::f64;
use std::fmt;
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
    InvalidCategories { m: usize },

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
            return Err(CdvmError::InvalidCategories { m });
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
                (kappa * (((2 * r) as f64 * pi - mu) / m as f64).cos())
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
            Self::InvalidCategories { m } => {
                write!(f, "number of categories ({}) must be at least 2", m)
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
        kappa * (((2 * x) as f64 * pi - mu) / m as f64).cos() + self.get_log_norm_const()
    }
}

impl Support<usize> for Cdvm {
    fn supports(&self, x: &usize) -> bool {
        0 <= *x && *x < self.modulus
    }
}

impl Sampleable<(usize, f64)> for Cdvm {
    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        // Sample categorical component
        let mut cumsum = 0.0;
        let u: f64 = rng.gen();
        
        let k = self.alpha.iter()
            .enumerate()
            .find(|(_, &p)| {
                cumsum += p;
                u <= cumsum
            })
            .map(|(i, _)| i)
            .unwrap_or(self.m - 1);

        // Sample von Mises component
        let theta = rand_distr::VonMises::new(self.mu, self.kappa)
            .unwrap()
            .sample(rng);

        (k, theta)
    }

    fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<(usize, f64)> {
        (0..n).map(|_| self.draw(rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_new_valid() {
        let cdvm = Cdvm::new(
            3,
            vec![0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            1.0,
            0.0,
        );
        assert!(cdvm.is_ok());
    }

    #[test]
    fn test_new_invalid_m() {
        let cdvm = Cdvm::new(
            1,
            vec![1.0],
            1.0,
            0.0,
        );
        assert!(matches!(cdvm, Err(CdvmError::InvalidCategories { m: 1 })));
    }

    #[test]
    fn test_new_invalid_alpha_length() {
        let cdvm = Cdvm::new(
            3,
            vec![1.0, 1.0],
            1.0,
            0.0,
        );
        assert!(matches!(cdvm, Err(CdvmError::AlphaLengthMismatch { .. })));
    }

    #[test]
    fn test_new_invalid_alpha() {
        let cdvm = Cdvm::new(
            3,
            vec![1.0, -1.0, 1.0],
            1.0,
            0.0,
        );
        assert!(matches!(cdvm, Err(CdvmError::InvalidProbabilities { .. })));
    }

    #[test]
    fn test_new_invalid_kappa() {
        let cdvm = Cdvm::new(
            3,
            vec![1.0, 1.0, 1.0],
            -1.0,
            0.0,
        );
        assert!(matches!(cdvm, Err(CdvmError::KappaNegative { .. })));
    }

    #[test]
    fn test_new_invalid_mu() {
        let cdvm = Cdvm::new(
            3,
            vec![1.0, 1.0, 1.0],
            1.0,
            2.0 * PI,
        );
        assert!(matches!(cdvm, Err(CdvmError::MuOutOfRange { .. })));
    }

    #[test]
    fn test_support() {
        let cdvm = Cdvm::new(
            3,
            vec![0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            1.0,
            0.0,
        ).unwrap();

        // Valid point
        assert!(cdvm.supports(&(1, 0.0)));

        // Invalid category
        assert!(!cdvm.supports(&(3, 0.0)));

        // Invalid angle
        assert!(!cdvm.supports(&(1, 4.0)));
    }

    #[test]
    fn test_sampling() {
        let mut rng = rand::thread_rng();
        let cdvm = Cdvm::new(
            3,
            vec![0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            1.0,
            0.0,
        ).unwrap();

        // Test single draw
        let (k, theta) = cdvm.draw(&mut rng);
        assert!(k < 3);
        assert!(theta >= -PI && theta <= PI);

        // Test multiple draws
        let samples = cdvm.sample(100, &mut rng);
        assert_eq!(samples.len(), 100);
        for (k, theta) in samples {
            assert!(k < 3);
            assert!(theta >= -PI && theta <= PI);
        }
    }
}
