//! Dirichlet and Symmetric Dirichlet distributions over simplexes
extern crate rand;
extern crate special;

use self::rand::distributions::Gamma as RGamma;
use self::rand::Rng;
use self::special::Gamma as SGamma;
use std::io;

use traits::*;

/// Symmetric Dirichlet distribution where all alphas are the same.
///
/// `SymmetricDirichlet { alpha, k }` is mathematicall equivalent to
/// `Dirichlet { alphas: vec![alpha; k] }`. This version has some extra
/// optimizations to seep up computing the PDF and drawing random vectors.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct SymmetricDirichlet {
    pub alpha: f64,
    pub k: usize,
}

impl SymmetricDirichlet {
    pub fn new(alpha: f64, k: usize) -> io::Result<Self> {
        let k_ok = k > 0;
        let alpha_ok = alpha > 0.0 && alpha.is_finite();

        if !k_ok {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "k must be greater than zero");
            Err(err)
        } else if !alpha_ok {
            let err_kind = io::ErrorKind::InvalidInput;
            let msg = "Alpha must be finite and greater than zero";
            let err = io::Error::new(err_kind, msg);
            Err(err)
        } else {
            Ok(SymmetricDirichlet { alpha, k })
        }
    }

    pub fn jeffreys(k: usize) -> io::Result<Self> {
        SymmetricDirichlet::new(0.5, k)
    }
}

impl Rv<Vec<f64>> for SymmetricDirichlet {
    fn draw<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        let g = RGamma::new(self.alpha, 1.0);
        let xs: Vec<f64> = (0..self.k).map(|_| rng.sample(g)).collect();
        let z = xs.iter().fold(0.0, |acc, x| acc + x);
        xs.iter().map(|x| x / z).collect()
    }

    #[inline]
    fn ln_normalizer() -> f64 {
        0.0
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

/// Dirichlet distribution over points on the k-simplex
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Dirichlet {
    /// A `Vec` of real numbers in (0, ∞)
    pub alphas: Vec<f64>,
}

impl Dirichlet {
    /// Creates a `Dirichlet` with a given `alphas` vector
    pub fn new(alphas: Vec<f64>) -> io::Result<Self> {
        if alphas.iter().any(|&a| !(a > 0.0 && a.is_finite())) {
            let err_kind = io::ErrorKind::InvalidInput;
            let msg = "All alphas must be finite and greater than zero";
            let err = io::Error::new(err_kind, msg);
            Err(err)
        } else {
            Ok(Dirichlet { alphas })
        }
    }

    /// Creates a `Dirichlet` where all alphas are identical
    ///
    /// # Eaxmples
    ///
    /// ```
    /// # extern crate rv;
    /// # use rv::dist::Dirichlet;
    /// #
    /// let dir = Dirichlet::symmetric(1.0, 4);
    /// assert_eq!(dir.alphas, vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn symmetric(alpha: f64, k: usize) -> Self {
        Dirichlet::new(vec![alpha; k]).unwrap()
    }

    /// Creates a `Dirichlet` with all alphas = 0.5 (Feffreys prior)
    ///
    /// # Eaxmples
    ///
    /// ```
    /// # extern crate rv;
    /// # use rv::dist::Dirichlet;
    /// #
    /// let dir = Dirichlet::jeffreys(3);
    /// assert_eq!(dir.alphas, vec![0.5, 0.5, 0.5]);
    /// ```
    pub fn jeffreys(k: usize) -> Self {
        Dirichlet::new(vec![0.5; k]).unwrap()
    }

    /// The length of `alphas` / the number of categories
    pub fn k(&self) -> usize {
        self.alphas.len()
    }
}

impl ContinuousDistr<Vec<f64>> for SymmetricDirichlet {}

impl Support<Vec<f64>> for SymmetricDirichlet {
    fn contains(&self, x: &Vec<f64>) -> bool {
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

    #[inline]
    fn ln_normalizer() -> f64 {
        0.0
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
    fn contains(&self, x: &Vec<f64>) -> bool {
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
    extern crate assert;
    use super::*;

    const TOL: f64 = 1E-12;

    mod dir {
        use super::*;

        #[test]
        fn properly_sized_points_on_simplex_should_be_in_support() {
            let dir = Dirichlet::symmetric(1.0, 4);
            assert!(dir.contains(&vec![0.25, 0.25, 0.25, 0.25]));
            assert!(dir.contains(&vec![0.1, 0.2, 0.3, 0.4]));
        }

        #[test]
        fn improperly_sized_points_should_not_be_in_support() {
            let dir = Dirichlet::symmetric(1.0, 3);
            assert!(!dir.contains(&vec![0.25, 0.25, 0.25, 0.25]));
            assert!(!dir.contains(&vec![0.1, 0.2, 0.7, 0.4]));
        }

        #[test]
        fn properly_sized_points_off_simplex_should_not_be_in_support() {
            let dir = Dirichlet::symmetric(1.0, 4);
            assert!(!dir.contains(&vec![0.25, 0.25, 0.26, 0.25]));
            assert!(!dir.contains(&vec![0.1, 0.3, 0.3, 0.4]));
        }

        #[test]
        fn draws_should_be_in_support() {
            let mut rng = rand::thread_rng();
            // Small alphas gives us more variability in the simplex, and more
            // variability gives us a beter test.
            let dir = Dirichlet::jeffreys(10);
            for _ in 0..100 {
                let x = dir.draw(&mut rng);
                assert!(dir.contains(&x));
            }
        }

        #[test]
        fn sample_should_return_the_proper_number_of_draws() {
            let mut rng = rand::thread_rng();
            let dir = Dirichlet::jeffreys(3);
            let xs: Vec<Vec<f64>> = dir.sample(88, &mut rng);
            assert_eq!(xs.len(), 88);
        }

        #[test]
        fn log_pdf_symemtric() {
            let dir = Dirichlet::symmetric(1.0, 3);
            assert::close(
                dir.ln_pdf(&vec![0.2, 0.3, 0.5]),
                0.69314718055994529,
                TOL,
            );
        }

        #[test]
        fn log_pdf_jeffreys() {
            let dir = Dirichlet::jeffreys(3);
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
            assert!(!symdir.contains(&vec![0.25, 0.25, 0.26, 0.25]));
            assert!(!symdir.contains(&vec![0.1, 0.3, 0.3, 0.4]));
        }

        #[test]
        fn draws_should_be_in_support() {
            let mut rng = rand::thread_rng();
            // Small alphas gives us more variability in the simplex, and more
            // variability gives us a beter test.
            let symdir = SymmetricDirichlet::jeffreys(10).unwrap();
            for _ in 0..100 {
                let x: Vec<f64> = symdir.draw(&mut rng);
                assert!(symdir.contains(&x));
            }
        }

    }
}
