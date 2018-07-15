extern crate rand;
extern crate special;

use self::rand::distributions::Gamma as RGamma;
use self::rand::Rng;
use self::special::Gamma as SGamma;

use traits::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Dirichlet {
    pub alphas: Vec<f64>,
}

impl Dirichlet {
    pub fn new(alphas: Vec<f64>) -> Self {
        Dirichlet { alphas: alphas }
    }

    pub fn symmetric(alpha: f64, k: usize) -> Self {
        Dirichlet::new(vec![alpha; k])
    }

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

        let ln_gamma_sum: f64 =
            self.alphas.iter().fold(0.0, |acc, &alpha| acc + alpha);

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
            x.iter().all(|&xi| xi > 0.0)
                && x.iter().fold(0.0, |acc, &xi| acc + xi).abs() < 1E-12
        }
    }
}
