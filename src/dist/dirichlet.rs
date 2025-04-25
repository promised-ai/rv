//! Dirichlet and Symmetric Dirichlet distributions over simplexes
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::misc::ln_gammafn;
use crate::misc::vec_to_string;
use crate::traits::{ContinuousDistr, HasDensity, Parameterized, Sampleable, Support};
use rand::Rng;
use rand_distr::Gamma as RGamma;
use std::fmt;
use std::sync::OnceLock;

mod categorical_prior;

/// Symmetric [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
/// where all alphas are the same.
///
/// `SymmetricDirichlet { alpha, k }` is mathematical equivalent to
/// `Dirichlet { alphas: vec![alpha; k] }`. This version has some extra
/// optimizations to seep up computing the PDF and drawing random vectors.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct SymmetricDirichlet {
    alpha: f64,
    k: usize,
    /// Cached `ln_gamma(alpha)`
    #[cfg_attr(feature = "serde1", serde(skip))]
    ln_gamma_alpha: OnceLock<f64>,
}

pub struct SymmetricDirichletParameters {
    pub alpha: f64,
    pub k: usize,
}

impl Parameterized for SymmetricDirichlet {
    type Parameters = SymmetricDirichletParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            alpha: self.alpha(),
            k: self.k(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.alpha, params.k)
    }
}

impl PartialEq for SymmetricDirichlet {
    fn eq(&self, other: &Self) -> bool {
        self.alpha == other.alpha && self.k == other.k
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum SymmetricDirichletError {
    /// k parameter is zero
    KIsZero,
    /// alpha parameter(s) is less than or equal to zero
    AlphaTooLow { alpha: f64 },
    /// alpha parameter(s) is infinite or NaN
    AlphaNotFinite { alpha: f64 },
}

impl SymmetricDirichlet {
    /// Create a new symmetric Dirichlet distribution
    ///
    /// # Arguments
    /// - alpha: The Dirichlet weight.
    /// - k : The number of weights. `alpha` will be replicated `k` times.
    #[inline]
    pub fn new(alpha: f64, k: usize) -> Result<Self, SymmetricDirichletError> {
        if k == 0 {
            Err(SymmetricDirichletError::KIsZero)
        } else if alpha <= 0.0 {
            Err(SymmetricDirichletError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(SymmetricDirichletError::AlphaNotFinite { alpha })
        } else {
            Ok(Self {
                alpha,
                k,
                ln_gamma_alpha: OnceLock::new(),
            })
        }
    }

    /// Create a new `SymmetricDirichlet` without checking whether the parameters
    /// are valid.
    #[inline]
    #[must_use] pub fn new_unchecked(alpha: f64, k: usize) -> Self {
        Self {
            alpha,
            k,
            ln_gamma_alpha: OnceLock::new(),
        }
    }

    /// The Jeffrey's Dirichlet prior for Categorical distributions
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::SymmetricDirichlet;
    /// let symdir = SymmetricDirichlet::jeffreys(4).unwrap();
    /// assert_eq!(symdir, SymmetricDirichlet::new(0.5, 4).unwrap());
    /// ```
    #[inline]
    pub fn jeffreys(k: usize) -> Result<Self, SymmetricDirichletError> {
        if k == 0 {
            Err(SymmetricDirichletError::KIsZero)
        } else {
            Ok(Self {
                alpha: 0.5,
                k,
                ln_gamma_alpha: OnceLock::new(),
            })
        }
    }

    /// Get the alpha uniform weight parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::SymmetricDirichlet;
    /// let symdir = SymmetricDirichlet::new(1.2, 5).unwrap();
    /// assert_eq!(symdir.alpha(), 1.2);
    /// ```
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set the value of alpha
    ///
    /// # Example
    /// ```rust
    /// # use rv::dist::SymmetricDirichlet;
    /// let mut symdir = SymmetricDirichlet::new(1.1, 5).unwrap();
    /// assert_eq!(symdir.alpha(), 1.1);
    ///
    /// symdir.set_alpha(2.3).unwrap();
    /// assert_eq!(symdir.alpha(), 2.3);
    /// ```
    ///
    /// Will error for invalid parameters
    ///
    /// ```rust
    /// # use rv::dist::SymmetricDirichlet;
    /// # let mut symdir = SymmetricDirichlet::new(1.1, 5).unwrap();
    /// assert!(symdir.set_alpha(0.5).is_ok());
    /// assert!(symdir.set_alpha(0.0).is_err());
    /// assert!(symdir.set_alpha(-1.0).is_err());
    /// assert!(symdir.set_alpha(f64::INFINITY).is_err());
    /// assert!(symdir.set_alpha(f64::NEG_INFINITY).is_err());
    /// assert!(symdir.set_alpha(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_alpha(
        &mut self,
        alpha: f64,
    ) -> Result<(), SymmetricDirichletError> {
        if alpha <= 0.0 {
            Err(SymmetricDirichletError::AlphaTooLow { alpha })
        } else if !alpha.is_finite() {
            Err(SymmetricDirichletError::AlphaNotFinite { alpha })
        } else {
            self.set_alpha_unchecked(alpha);
            self.ln_gamma_alpha = OnceLock::new();
            Ok(())
        }
    }

    /// Set the value of alpha without input validation
    #[inline]
    pub fn set_alpha_unchecked(&mut self, alpha: f64) {
        self.alpha = alpha;
        self.ln_gamma_alpha = OnceLock::new();
    }

    /// Get the number of weights, k
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::SymmetricDirichlet;
    /// let symdir = SymmetricDirichlet::new(1.2, 5).unwrap();
    /// assert_eq!(symdir.k(), 5);
    /// ```
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    fn ln_gamma_alpha(&self) -> f64 {
        *self.ln_gamma_alpha.get_or_init(|| ln_gammafn(self.alpha))
    }
}

impl From<&SymmetricDirichlet> for String {
    fn from(symdir: &SymmetricDirichlet) -> String {
        format!("SymmetricDirichlet({}; α: {})", symdir.k, symdir.alpha)
    }
}

impl_display!(SymmetricDirichlet);

impl Sampleable<Vec<f64>> for SymmetricDirichlet {
    fn draw<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        let g = RGamma::new(self.alpha, 1.0).unwrap();
        let mut xs: Vec<f64> = (0..self.k).map(|_| rng.sample(g)).collect();
        let z: f64 = xs.iter().sum();
        xs.iter_mut().for_each(|x| *x /= z);
        xs
    }
}

impl HasDensity<Vec<f64>> for SymmetricDirichlet {
    fn ln_f(&self, x: &Vec<f64>) -> f64 {
        let kf = self.k as f64;
        let sum_ln_gamma = self.ln_gamma_alpha() * kf;
        let ln_gamma_sum = ln_gammafn(self.alpha * kf);

        let am1 = self.alpha - 1.0;
        let term = x.iter().fold(0.0, |acc, &xi| am1.mul_add(xi.ln(), acc));

        term - (sum_ln_gamma - ln_gamma_sum)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum DirichletError {
    /// k parameter is zero
    KIsZero,
    /// alpha vector is empty
    AlphasEmpty,
    /// alphas parameter has one or more entries less than or equal to zero
    AlphaTooLow { ix: usize, alpha: f64 },
    /// alphas parameter has one or infinite or NaN entries
    AlphaNotFinite { ix: usize, alpha: f64 },
}

/// [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
/// over points on the k-simplex.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Dirichlet {
    /// A `Vec` of real numbers in (0, ∞)
    pub(crate) alphas: Vec<f64>,
}

pub struct DirichletParameters {
    pub alphas: Vec<f64>,
}

impl Parameterized for Dirichlet {
    type Parameters = DirichletParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            alphas: self.alphas().clone(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.alphas)
    }
}

impl From<SymmetricDirichlet> for Dirichlet {
    fn from(symdir: SymmetricDirichlet) -> Self {
        Dirichlet::new_unchecked(vec![symdir.alpha; symdir.k])
    }
}

impl From<&SymmetricDirichlet> for Dirichlet {
    fn from(symdir: &SymmetricDirichlet) -> Self {
        Dirichlet::new_unchecked(vec![symdir.alpha; symdir.k])
    }
}

impl Dirichlet {
    /// Creates a `Dirichlet` with a given `alphas` vector
    pub fn new(alphas: Vec<f64>) -> Result<Self, DirichletError> {
        if alphas.is_empty() {
            return Err(DirichletError::AlphasEmpty);
        }

        alphas.iter().enumerate().try_for_each(|(ix, &alpha)| {
            if alpha <= 0.0 {
                Err(DirichletError::AlphaTooLow { ix, alpha })
            } else if !alpha.is_finite() {
                Err(DirichletError::AlphaNotFinite { ix, alpha })
            } else {
                Ok(())
            }
        })?;

        Ok(Dirichlet { alphas })
    }

    /// Creates a new Dirichlet without checking whether the parameters are
    /// valid.
    #[inline]
    #[must_use] pub fn new_unchecked(alphas: Vec<f64>) -> Self {
        Dirichlet { alphas }
    }

    /// Creates a `Dirichlet` where all alphas are identical.
    ///
    /// # Notes
    ///
    /// `SymmetricDirichlet` if faster and more compact, and is the preferred
    /// way to represent a Dirichlet symmetric weights.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rv::dist::{Dirichlet, SymmetricDirichlet};
    /// # use rv::traits::*;
    /// let dir = Dirichlet::symmetric(1.0, 4).unwrap();
    /// assert_eq!(*dir.alphas(), vec![1.0, 1.0, 1.0, 1.0]);
    ///
    /// // Equivalent to SymmetricDirichlet
    /// let symdir = SymmetricDirichlet::new(1.0, 4).unwrap();
    /// let x: Vec<f64> = vec![0.1, 0.4, 0.3, 0.2];
    /// assert::close(dir.ln_f(&x), symdir.ln_f(&x), 1E-12);
    /// ```
    #[inline]
    pub fn symmetric(alpha: f64, k: usize) -> Result<Self, DirichletError> {
        if k == 0 {
            Err(DirichletError::KIsZero)
        } else if alpha <= 0.0 {
            Err(DirichletError::AlphaTooLow { ix: 0, alpha })
        } else if !alpha.is_finite() {
            Err(DirichletError::AlphaNotFinite { ix: 0, alpha })
        } else {
            Ok(Dirichlet {
                alphas: vec![alpha; k],
            })
        }
    }

    /// Creates a `Dirichlet` with all alphas = 0.5 (Jeffreys prior)
    ///
    /// # Notes
    ///
    /// `SymmetricDirichlet` if faster and more compact, and is the preferred
    /// way to represent a Dirichlet symmetric weights.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rv::dist::Dirichlet;
    /// # use rv::dist::SymmetricDirichlet;
    /// # use rv::traits::*;
    /// let dir = Dirichlet::jeffreys(3).unwrap();
    /// assert_eq!(*dir.alphas(), vec![0.5, 0.5, 0.5]);
    ///
    /// // Equivalent to SymmetricDirichlet::jeffreys
    /// let symdir = SymmetricDirichlet::jeffreys(3).unwrap();
    /// let x: Vec<f64> = vec![0.1, 0.4, 0.5];
    /// assert::close(dir.ln_f(&x), symdir.ln_f(&x), 1E-12);
    /// ```
    #[inline]
    pub fn jeffreys(k: usize) -> Result<Self, DirichletError> {
        if k == 0 {
            Err(DirichletError::KIsZero)
        } else {
            Ok(Dirichlet::new_unchecked(vec![0.5; k]))
        }
    }

    /// The length of `alphas` / the number of categories
    #[inline]
    #[must_use] pub fn k(&self) -> usize {
        self.alphas.len()
    }

    /// Get a reference to the weights vector, `alphas`
    #[inline]
    #[must_use] pub fn alphas(&self) -> &Vec<f64> {
        &self.alphas
    }
}

impl From<&Dirichlet> for String {
    fn from(dir: &Dirichlet) -> String {
        format!("Dir(α: {})", vec_to_string(&dir.alphas, 5))
    }
}

impl_display!(Dirichlet);

impl ContinuousDistr<Vec<f64>> for SymmetricDirichlet {}

impl Support<Vec<f64>> for SymmetricDirichlet {
    fn supports(&self, x: &Vec<f64>) -> bool {
        if x.len() == self.k {
            let sum = x.iter().fold(0.0, |acc, &xi| acc + xi);
            x.iter().all(|&xi| xi > 0.0) && (1.0 - sum).abs() < 1E-12
        } else {
            false
        }
    }
}

impl Sampleable<Vec<f64>> for Dirichlet {
    fn draw<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        let gammas: Vec<RGamma<f64>> = self
            .alphas
            .iter()
            .map(|&alpha| RGamma::new(alpha, 1.0).unwrap())
            .collect();
        let mut xs: Vec<f64> = gammas.iter().map(|g| rng.sample(g)).collect();
        let z: f64 = xs.iter().sum();
        xs.iter_mut().for_each(|x| *x /= z);
        xs
    }
}

impl HasDensity<Vec<f64>> for Dirichlet {
    fn ln_f(&self, x: &Vec<f64>) -> f64 {
        // XXX: could cache all ln_gamma(alpha)
        let sum_ln_gamma: f64 = self
            .alphas
            .iter()
            .fold(0.0, |acc, &alpha| acc + ln_gammafn(alpha));

        let ln_gamma_sum: f64 = ln_gammafn(self.alphas.iter().sum::<f64>());

        let term = x
            .iter()
            .zip(self.alphas.iter())
            .fold(0.0, |acc, (&xi, &alpha)| {
                (alpha - 1.0).mul_add(xi.ln(), acc)
            });

        term - (sum_ln_gamma - ln_gamma_sum)
    }
}

impl ContinuousDistr<Vec<f64>> for Dirichlet {}

impl Support<Vec<f64>> for Dirichlet {
    fn supports(&self, x: &Vec<f64>) -> bool {
        if x.len() == self.alphas.len() {
            let sum = x.iter().fold(0.0, |acc, &xi| acc + xi);
            x.iter().all(|&xi| xi > 0.0) && (1.0 - sum).abs() < 1E-12
        } else {
            false
        }
    }
}

impl std::error::Error for SymmetricDirichletError {}
impl std::error::Error for DirichletError {}

impl fmt::Display for SymmetricDirichletError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlphaTooLow { alpha } => {
                write!(f, "alpha ({alpha}) must be greater than zero")
            }
            Self::AlphaNotFinite { alpha } => {
                write!(f, "alpha ({alpha}) was non-finite")
            }
            Self::KIsZero => write!(f, "k must be greater than zero"),
        }
    }
}

impl fmt::Display for DirichletError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KIsZero => write!(f, "k must be greater than zero"),
            Self::AlphasEmpty => write!(f, "alphas vector was empty"),
            Self::AlphaTooLow { ix, alpha } => {
                write!(f, "Invalid alpha at index {ix}: {alpha} <= 0.0")
            }
            Self::AlphaNotFinite { ix, alpha } => {
                write!(f, "Non-finite alpha at index {ix}: {alpha}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_basic_impls, verify_cache_resets};

    const TOL: f64 = 1E-12;

    mod dir {
        use super::*;

        test_basic_impls!(Vec<f64>, Dirichlet, Dirichlet::jeffreys(4).unwrap());

        #[test]
        fn properly_sized_points_on_simplex_should_be_in_support() {
            let dir = Dirichlet::symmetric(1.0, 4).unwrap();
            assert!(dir.supports(&vec![0.25, 0.25, 0.25, 0.25]));
            assert!(dir.supports(&vec![0.1, 0.2, 0.3, 0.4]));
        }

        #[test]
        fn improperly_sized_points_should_not_be_in_support() {
            let dir = Dirichlet::symmetric(1.0, 3).unwrap();
            assert!(!dir.supports(&vec![0.25, 0.25, 0.25, 0.25]));
            assert!(!dir.supports(&vec![0.1, 0.2, 0.7, 0.4]));
        }

        #[test]
        fn properly_sized_points_off_simplex_should_not_be_in_support() {
            let dir = Dirichlet::symmetric(1.0, 4).unwrap();
            assert!(!dir.supports(&vec![0.25, 0.25, 0.26, 0.25]));
            assert!(!dir.supports(&vec![0.1, 0.3, 0.3, 0.4]));
        }

        #[test]
        fn draws_should_be_in_support() {
            let mut rng = rand::thread_rng();
            // Small alphas gives us more variability in the simplex, and more
            // variability gives us a better test.
            let dir = Dirichlet::jeffreys(10).unwrap();
            for _ in 0..100 {
                let x = dir.draw(&mut rng);
                assert!(dir.supports(&x));
            }
        }

        #[test]
        fn sample_should_return_the_proper_number_of_draws() {
            let mut rng = rand::thread_rng();
            let dir = Dirichlet::jeffreys(3).unwrap();
            let xs: Vec<Vec<f64>> = dir.sample(88, &mut rng);
            assert_eq!(xs.len(), 88);
        }

        #[test]
        fn log_pdf_symmetric() {
            let dir = Dirichlet::symmetric(1.0, 3).unwrap();
            assert::close(
                dir.ln_pdf(&vec![0.2, 0.3, 0.5]),
                std::f64::consts::LN_2,
                TOL,
            );
        }

        #[test]
        fn log_pdf_jeffreys() {
            let dir = Dirichlet::jeffreys(3).unwrap();
            assert::close(
                dir.ln_pdf(&vec![0.2, 0.3, 0.5]),
                -0.084_598_117_749_354_22,
                TOL,
            );
        }

        #[test]
        fn log_pdf() {
            let dir = Dirichlet::new(vec![1.0, 2.0, 3.0]).unwrap();
            assert::close(
                dir.ln_pdf(&vec![0.2, 0.3, 0.5]),
                1.504_077_396_776_273_7,
                TOL,
            );
        }
    }

    mod symdir {
        use std::f64::consts::PI;

        use super::*;

        test_basic_impls!(
            Vec<f64>,
            SymmetricDirichlet,
            SymmetricDirichlet::jeffreys(4).unwrap()
        );

        #[test]
        fn sample_should_return_the_proper_number_of_draws() {
            let mut rng = rand::thread_rng();
            let symdir = SymmetricDirichlet::jeffreys(3).unwrap();
            let xs: Vec<Vec<f64>> = symdir.sample(88, &mut rng);
            assert_eq!(xs.len(), 88);
        }

        #[test]
        fn log_pdf_jeffreys() {
            let symdir = SymmetricDirichlet::jeffreys(3).unwrap();
            assert::close(
                symdir.ln_pdf(&vec![0.2, 0.3, 0.5]),
                -0.084_598_117_749_354_22,
                TOL,
            );
        }

        #[test]
        fn properly_sized_points_off_simplex_should_not_be_in_support() {
            let symdir = SymmetricDirichlet::new(1.0, 4).unwrap();
            assert!(!symdir.supports(&vec![0.25, 0.25, 0.26, 0.25]));
            assert!(!symdir.supports(&vec![0.1, 0.3, 0.3, 0.4]));
        }

        #[test]
        fn draws_should_be_in_support() {
            let mut rng = rand::thread_rng();
            // Small alphas gives us more variability in the simplex, and more
            // variability gives us a better test.
            let symdir = SymmetricDirichlet::jeffreys(10).unwrap();
            for _ in 0..100 {
                let x: Vec<f64> = symdir.draw(&mut rng);
                assert!(symdir.supports(&x));
            }
        }

        verify_cache_resets!(
            [unchecked],
            ln_f_is_same_after_reset_unchecked_alpha_identically,
            set_alpha_unchecked,
            SymmetricDirichlet::new(1.2, 2).unwrap(),
            vec![0.1_f64, 0.9_f64],
            1.2,
            PI
        );

        verify_cache_resets!(
            [checked],
            ln_f_is_same_after_reset_checked_alpha_identically,
            set_alpha,
            SymmetricDirichlet::new(1.2, 2).unwrap(),
            vec![0.1_f64, 0.9_f64],
            1.2,
            PI
        );
    }
}
