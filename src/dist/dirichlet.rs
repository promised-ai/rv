//! Dirichlet and Symmetric Dirichlet distributions over simplexes
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::impl_display;
use crate::misc::vec_to_string;
use crate::result;
use crate::traits::*;
use rand::distributions::Gamma as RGamma;
use rand::Rng;
use special::Gamma as _;

/// Symmetric [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
/// where all alphas are the same.
///
/// `SymmetricDirichlet { alpha, k }` is mathematicall equivalent to
/// `Dirichlet { alphas: vec![alpha; k] }`. This version has some extra
/// optimizations to seep up computing the PDF and drawing random vectors.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct SymmetricDirichlet {
    alpha: f64,
    k: usize,
}

impl SymmetricDirichlet {
    /// Create a new symmetric Dirichlet distributon
    ///
    /// # Arguments
    /// - alpha: The Dirichlet weight.
    /// - k : The number of weights. `alpha` will be replicated `k` times.
    pub fn new(alpha: f64, k: usize) -> result::Result<Self> {
        let k_ok = k > 0;
        let alpha_ok = alpha > 0.0 && alpha.is_finite();

        if !k_ok {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let err =
                result::Error::new(err_kind, "k must be greater than zero");
            Err(err)
        } else if !alpha_ok {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "Alpha must be finite and greater than zero";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        } else {
            Ok(SymmetricDirichlet { alpha, k })
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
    pub fn jeffreys(k: usize) -> result::Result<Self> {
        if k == 0 {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let err =
                result::Error::new(err_kind, "k must be greater than zero");
            Err(err)
        } else {
            Ok(SymmetricDirichlet { alpha: 0.5, k })
        }
    }

    /// Get the alpha unfiorm weight parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::SymmetricDirichlet;
    /// let symdir = SymmetricDirichlet::new(1.2, 5).unwrap();
    /// assert_eq!(symdir.alpha(), 1.2);
    /// ```
    pub fn alpha(&self) -> f64 {
        self.alpha
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
    pub fn k(&self) -> usize {
        self.k
    }
}

impl From<&SymmetricDirichlet> for String {
    fn from(symdir: &SymmetricDirichlet) -> String {
        format!("SymmetricDirichlet({}; α: {})", symdir.k, symdir.alpha)
    }
}

impl_display!(SymmetricDirichlet);

impl Rv<Vec<f64>> for SymmetricDirichlet {
    fn draw<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        let g = RGamma::new(self.alpha, 1.0);
        let xs: Vec<f64> = (0..self.k).map(|_| rng.sample(g)).collect();
        let z = xs.iter().fold(0.0, |acc, x| acc + x);
        xs.iter().map(|x| x / z).collect()
    }

    fn ln_f(&self, x: &Vec<f64>) -> f64 {
        let kf = self.k as f64;
        let sum_ln_gamma = self.alpha.ln_gamma().0 * kf;
        let ln_gamma_sum = (self.alpha * kf).ln_gamma().0;

        let am1 = self.alpha - 1.0;
        let term = x.iter().fold(0.0, |acc, &xi| acc + am1 * xi.ln());

        term - (sum_ln_gamma - ln_gamma_sum)
    }
}

/// [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
/// over points on the k-simplex.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Dirichlet {
    /// A `Vec` of real numbers in (0, ∞)
    alphas: Vec<f64>,
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
    pub fn new(alphas: Vec<f64>) -> result::Result<Self> {
        if alphas.is_empty() {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let err =
                result::Error::new(err_kind, "k must be greater than zero");
            Err(err)
        } else if alphas.iter().any(|&a| !(a > 0.0 && a.is_finite())) {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "All alphas must be finite and greater than zero";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        } else {
            Ok(Dirichlet { alphas })
        }
    }

    fn new_unchecked(alphas: Vec<f64>) -> Self {
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
    /// # use rv::traits::Rv;
    /// let dir = Dirichlet::symmetric(1.0, 4).unwrap();
    /// assert_eq!(*dir.alphas(), vec![1.0, 1.0, 1.0, 1.0]);
    ///
    /// // Equivalent to SymmetricDirichlet
    /// let symdir = SymmetricDirichlet::new(1.0, 4).unwrap();
    /// let x: Vec<f64> = vec![0.1, 0.4, 0.3, 0.2];
    /// assert::close(dir.ln_f(&x), symdir.ln_f(&x), 1E-12);
    /// ```
    pub fn symmetric(alpha: f64, k: usize) -> result::Result<Self> {
        if k == 0 {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let err =
                result::Error::new(err_kind, "k must be greater than zero");
            Err(err)
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
    /// # use rv::traits::Rv;
    /// let dir = Dirichlet::jeffreys(3).unwrap();
    /// assert_eq!(*dir.alphas(), vec![0.5, 0.5, 0.5]);
    ///
    /// // Equivalent to SymmetricDirichlet::jeffreys
    /// let symdir = SymmetricDirichlet::jeffreys(3).unwrap();
    /// let x: Vec<f64> = vec![0.1, 0.4, 0.5];
    /// assert::close(dir.ln_f(&x), symdir.ln_f(&x), 1E-12);
    /// ```
    pub fn jeffreys(k: usize) -> result::Result<Self> {
        Dirichlet::symmetric(0.5, k)
    }

    /// The length of `alphas` / the number of categories
    pub fn k(&self) -> usize {
        self.alphas.len()
    }

    /// Get a reference to the weights vector, `alphas`
    pub fn alphas(&self) -> &Vec<f64> {
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
        if x.len() != self.k {
            false
        } else {
            let sum = x.iter().fold(0.0, |acc, &xi| acc + xi);
            x.iter().all(|&xi| xi > 0.0) && (1.0 - sum).abs() < 1E-12
        }
    }
}

impl Rv<Vec<f64>> for Dirichlet {
    fn draw<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        // TODO: offload to Gamma distribution
        let gammas: Vec<RGamma> = self
            .alphas
            .iter()
            .map(|&alpha| RGamma::new(alpha, 1.0))
            .collect();
        let xs: Vec<f64> = gammas.iter().map(|g| rng.sample(g)).collect();
        let z = xs.iter().fold(0.0, |acc, x| acc + x);
        xs.iter().map(|x| x / z).collect()
    }

    fn ln_f(&self, x: &Vec<f64>) -> f64 {
        let sum_ln_gamma: f64 = self
            .alphas
            .iter()
            .fold(0.0, |acc, &alpha| acc + alpha.ln_gamma().0);

        let ln_gamma_sum: f64 = self
            .alphas
            .iter()
            .fold(0.0, |acc, &alpha| acc + alpha)
            .ln_gamma()
            .0;

        let term = x
            .iter()
            .zip(self.alphas.iter())
            .fold(0.0, |acc, (&xi, &alpha)| acc + (alpha - 1.0) * xi.ln());

        term - (sum_ln_gamma - ln_gamma_sum)
    }
}

impl ContinuousDistr<Vec<f64>> for Dirichlet {}

impl Support<Vec<f64>> for Dirichlet {
    fn supports(&self, x: &Vec<f64>) -> bool {
        if x.len() != self.alphas.len() {
            false
        } else {
            let sum = x.iter().fold(0.0, |acc, &xi| acc + xi);
            x.iter().all(|&xi| xi > 0.0) && (1.0 - sum).abs() < 1E-12
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-12;

    mod dir {
        use super::*;

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
            // variability gives us a beter test.
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
        fn log_pdf_symemtric() {
            let dir = Dirichlet::symmetric(1.0, 3).unwrap();
            assert::close(
                dir.ln_pdf(&vec![0.2, 0.3, 0.5]),
                0.69314718055994529,
                TOL,
            );
        }

        #[test]
        fn log_pdf_jeffreys() {
            let dir = Dirichlet::jeffreys(3).unwrap();
            assert::close(
                dir.ln_pdf(&vec![0.2, 0.3, 0.5]),
                -0.084598117749354218,
                TOL,
            );
        }

        #[test]
        fn log_pdf() {
            let dir = Dirichlet::new(vec![1.0, 2.0, 3.0]).unwrap();
            assert::close(
                dir.ln_pdf(&vec![0.2, 0.3, 0.5]),
                1.5040773967762737,
                TOL,
            );
        }
    }

    mod symdir {
        use super::*;

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
                -0.084598117749354218,
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
            // variability gives us a beter test.
            let symdir = SymmetricDirichlet::jeffreys(10).unwrap();
            for _ in 0..100 {
                let x: Vec<f64> = symdir.draw(&mut rng);
                assert!(symdir.supports(&x));
            }
        }

    }
}
