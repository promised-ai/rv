extern crate rand;
extern crate special;

use self::rand::Rng;
use self::special::Gamma as SGamma;

use data::{CategoricalDatum, CategoricalSuffStat, DataOrSuffStat};
use dist::{Categorical, Dirichlet, SymmetricDirichlet};
use prelude::CategoricalData;
use traits::*;

impl Rv<Categorical> for SymmetricDirichlet {
    fn ln_f(&self, x: &Categorical) -> f64 {
        self.ln_f(&x.weights())
    }

    fn ln_normalizer(&self) -> f64 {
        0.0
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Categorical {
        let weights: Vec<f64> = self.draw(&mut rng);
        Categorical::new(&weights).expect("Invalid draw")
    }
}

impl<X: CategoricalDatum> ConjugatePrior<X, Categorical>
    for SymmetricDirichlet
{
    type Posterior = Dirichlet;
    fn posterior(&self, x: &CategoricalData<X>) -> Self::Posterior {
        let stat = extract_stat(self.k, x);

        let alphas: Vec<f64> =
            stat.counts.iter().map(|&ct| self.alpha + ct).collect();

        Dirichlet::new(alphas).unwrap()
    }

    fn ln_m(&self, x: &CategoricalData<X>) -> f64 {
        let sum_alpha = self.alpha * self.k as f64;
        let stat = extract_stat(self.k, x);

        // terms
        let a = sum_alpha.ln_gamma().0;
        let b = (sum_alpha + stat.n as f64).ln_gamma().0;
        let c = stat
            .counts
            .iter()
            .fold(0.0, |acc, &ct| acc + (self.alpha + ct).ln_gamma().0);
        let d = self.alpha.ln_gamma().0 * self.k as f64;

        a - b + c - d
    }

    fn ln_pp(&self, y: &X, x: &CategoricalData<X>) -> f64 {
        let post = self.posterior(x);
        let norm = post.alphas.iter().fold(0.0, |acc, &a| acc + a);
        let ix: usize = (*y).into();
        post.alphas[ix].ln() - norm.ln()
    }
}

impl Rv<Categorical> for Dirichlet {
    fn ln_f(&self, x: &Categorical) -> f64 {
        self.ln_f(&x.weights())
    }

    fn ln_normalizer(&self) -> f64 {
        0.0
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Categorical {
        let weights: Vec<f64> = self.draw(&mut rng);
        Categorical::new(&weights).expect("Invalid draw")
    }
}

fn extract_stat<X: CategoricalDatum>(
    k: usize,
    x: &CategoricalData<X>,
) -> CategoricalSuffStat {
    match x {
        DataOrSuffStat::SuffStat(ref s) => (*s).clone(),
        DataOrSuffStat::Data(xs) => {
            let mut stat = CategoricalSuffStat::new(k);
            xs.iter().for_each(|y| stat.observe(y));
            stat
        }
    }
}

impl<X: CategoricalDatum> ConjugatePrior<X, Categorical> for Dirichlet {
    type Posterior = Self;
    fn posterior(&self, x: &CategoricalData<X>) -> Self::Posterior {
        let stat = extract_stat(self.k(), x);

        let alphas: Vec<f64> = self
            .alphas
            .iter()
            .zip(stat.counts.iter())
            .map(|(&a, &ct)| a + ct)
            .collect();

        Dirichlet::new(alphas).unwrap()
    }

    fn ln_m(&self, x: &CategoricalData<X>) -> f64 {
        let sum_alpha = self.alphas.iter().fold(0.0, |acc, &a| acc + a);
        let stat = extract_stat(self.k(), x);

        // terms
        let a = sum_alpha.ln_gamma().0;
        let b = (sum_alpha + stat.n as f64).ln_gamma().0;
        let c = self
            .alphas
            .iter()
            .zip(stat.counts.iter())
            .fold(0.0, |acc, (&a, &ct)| acc + (a + ct).ln_gamma().0);
        let d = self.alphas.iter().fold(0.0, |acc, &a| acc + a.ln_gamma().0);

        a - b + c - d
    }

    fn ln_pp(&self, y: &X, x: &CategoricalData<X>) -> f64 {
        let post = self.posterior(x);
        let norm = post.alphas.iter().fold(0.0, |acc, &a| acc + a);
        let ix: usize = (*y).into();
        post.alphas[ix].ln() - norm.ln()
    }
}
