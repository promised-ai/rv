//! Dirichlet and Symmetric Dirichlet distributions over simplexes
extern crate rand;
extern crate special;

use self::rand::distributions::Gamma as RGamma;
use self::rand::Rng;
use self::special::Gamma as SGamma;

use traits::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Dirichlet {
    /// A `Vec` of real numbers in (0, ∞)
    pub alphas: Vec<f64>,
}

impl Dirichlet {
    /// Creates a `Dirichlet` with a given `alphas` vector
    pub fn new(alphas: Vec<f64>) -> Self {
        Dirichlet { alphas }
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
        Dirichlet::new(vec![alpha; k])
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
        Dirichlet::new(vec![0.5; k])
    }

    /// The length of `alphas` / the number of categories
    pub fn k(&self) -> usize {
        self.alphas.len()
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

    fn ln_normalizer(&self) -> f64 {
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
        let dir = Dirichlet::new(vec![1.0, 2.0, 3.0]);
        assert::close(
            dir.ln_pdf(&vec![0.2, 0.3, 0.5]),
            1.5040773967762737,
            TOL,
        );
    }
}
