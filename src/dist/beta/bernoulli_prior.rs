use rand::Rng;
use special::Beta as SBeta;

use crate::data::{BernoulliSuffStat, Booleable, DataOrSuffStat};
use crate::dist::{Bernoulli, Beta};
use crate::suffstat_traits::*;
use crate::traits::*;

impl Rv<Bernoulli> for Beta {
    fn ln_f(&self, x: &Bernoulli) -> f64 {
        self.ln_f(&x.p())
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Bernoulli {
        let p: f64 = self.draw(&mut rng);
        Bernoulli::new(p).expect("Failed to draw valid weight")
    }
}

impl Support<Bernoulli> for Beta {
    fn supports(&self, x: &Bernoulli) -> bool {
        0.0 < x.p() && x.p() < 1.0
    }
}

impl ContinuousDistr<Bernoulli> for Beta {}

impl<X: Booleable> ConjugatePrior<X, Bernoulli> for Beta {
    type Posterior = Self;
    type LnMCache = f64;
    type LnPpCache = (f64, f64);

    #[allow(clippy::many_single_char_names)]
    fn posterior(&self, x: &DataOrSuffStat<X, Bernoulli>) -> Self {
        let (n, k) = match x {
            DataOrSuffStat::Data(xs) => {
                let mut stat = BernoulliSuffStat::new();
                xs.iter().for_each(|x| stat.observe(x));
                (stat.n(), stat.k())
            }
            DataOrSuffStat::SuffStat(stat) => (stat.n(), stat.k()),
            DataOrSuffStat::None => (0, 0),
        };

        let a = self.alpha() + k as f64;
        let b = self.beta() + (n - k) as f64;

        Beta::new(a, b).expect("Invalid posterior parameters")
    }

    #[inline]
    fn ln_m_cache(&self) -> Self::LnMCache {
        self.alpha().ln_beta(self.beta())
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::LnMCache,
        x: &DataOrSuffStat<X, Bernoulli>,
    ) -> f64 {
        let post = self.posterior(x);
        post.alpha().ln_beta(post.beta()) - cache
    }

    #[inline]
    fn ln_pp_cache(&self, x: &DataOrSuffStat<X, Bernoulli>) -> Self::LnPpCache {
        //  P(y=1 | xs) happens to be the posterior mean
        let post = self.posterior(x);
        let p: f64 = post.mean().expect("Mean undefined");
        (p.ln(), (1.0 - p).ln())
    }

    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &X) -> f64 {
        //  P(y=1 | xs) happens to be the posterior mean
        if y.into_bool() {
            cache.0
        } else {
            cache.1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn posterior_from_data_bool() {
        let data = vec![false, true, false, true, true];
        let xs = DataOrSuffStat::Data::<bool, Bernoulli>(&data);

        let posterior = Beta::new(1.0, 1.0).unwrap().posterior(&xs);

        assert::close(posterior.alpha(), 4.0, TOL);
        assert::close(posterior.beta(), 3.0, TOL);
    }

    #[test]
    fn posterior_from_data_u16() {
        let data: Vec<u16> = vec![0, 1, 0, 1, 1];
        let xs = DataOrSuffStat::Data::<u16, Bernoulli>(&data);

        let posterior = Beta::new(1.0, 1.0).unwrap().posterior(&xs);

        assert::close(posterior.alpha(), 4.0, TOL);
        assert::close(posterior.beta(), 3.0, TOL);
    }
}
