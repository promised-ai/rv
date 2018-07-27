extern crate rand;
extern crate special;

use self::rand::Rng;
use self::special::Beta as SBeta;

use data::{BernoulliSuffStat, DataOrSuffStat};
use dist::{Bernoulli, Beta};
use traits::*;

impl Rv<Bernoulli> for Beta {
    fn ln_f(&self, x: &Bernoulli) -> f64 {
        self.ln_f(&x.p)
    }

    fn ln_normalizer(&self) -> f64 {
        0.0
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Bernoulli {
        let p: f64 = self.draw(&mut rng);
        Bernoulli::new(p).expect("Failed to draw valid weight")
    }
}

impl Support<Bernoulli> for Beta {
    fn contains(&self, x: &Bernoulli) -> bool {
        0.0 < x.p && x.p < 1.0
    }
}

impl ContinuousDistr<Bernoulli> for Beta {}

impl ConjugatePrior<bool, Bernoulli> for Beta {
    type Posterior = Self;
    fn posterior(&self, x: &DataOrSuffStat<bool, Bernoulli>) -> Self {
        let (n, k) = match x {
            DataOrSuffStat::Data(ref xs) => {
                let mut stat = BernoulliSuffStat::new();
                xs.iter().for_each(|x| stat.observe(x));
                (stat.n, stat.k)
            }
            DataOrSuffStat::SuffStat(ref stat) => (stat.n, stat.k),
        };

        let a = self.alpha + k as f64;
        let b = self.beta + (n - k) as f64;

        Beta::new(a, b).expect("Invalid posterior parameters")
    }

    fn ln_m(&self, x: &DataOrSuffStat<bool, Bernoulli>) -> f64 {
        let post = self.posterior(x);
        post.alpha.ln_beta(post.beta) - self.alpha.ln_beta(self.beta)
    }

    fn ln_pp(&self, y: &bool, x: &DataOrSuffStat<bool, Bernoulli>) -> f64 {
        //  P(y=1 | xs) happens to be the posterior mean
        let post = self.posterior(x);
        let p: f64 = post.mean().expect("Mean undefined");
        if *y {
            p.ln()
        } else {
            (1.0 - p).ln()
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate assert;
    use super::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn posterior_from_data() {
        let data = vec![false, true, false, true, true];
        let xs = DataOrSuffStat::Data::<bool, Bernoulli>(&data);

        let posterior = Beta::new(1.0, 1.0).unwrap().posterior(&xs);

        assert::close(posterior.alpha, 4.0, TOL);
        assert::close(posterior.beta, 3.0, TOL);
    }
}
