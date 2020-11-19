use nalgebra::{DMatrix, DVector};

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Model of noise to use in Gaussian Process
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum NoiseModel {
    /// The same noise is applied to all values
    Uniform(f64),
    /// Different noise values are applied to each y-value
    PerPoint(DVector<f64>),
}

impl Default for NoiseModel {
    fn default() -> Self {
        NoiseModel::Uniform(1E-10)
    }
}

impl NoiseModel {
    /// Enact the given noise model onto the given covariance matrix
    pub fn add_noise_to_kernel(
        &self,
        cov: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, String> {
        match self {
            NoiseModel::Uniform(noise) => {
                let diag = DVector::from_element(cov.nrows(), noise.powi(2));
                Ok(cov + &DMatrix::from_diagonal(&diag))
            }
            NoiseModel::PerPoint(sigma) => {
                if cov.nrows() == sigma.nrows() {
                    Ok(cov + &DMatrix::from_diagonal(&sigma))
                } else {
                    Err(format!("Per point noise must be the same size a y_train (expected: {}, got: {})", cov.nrows(), sigma.nrows()))
                }
            }
        }
    }
}
