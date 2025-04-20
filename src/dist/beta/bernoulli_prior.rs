use rand::Rng;
use special::Beta as SBeta;
use std::marker::PhantomData;

use crate::data::{BernoulliSuffStat, Booleable};
use crate::dist::{Bernoulli, Beta};
use crate::traits::*;

impl HasDensity<Bernoulli> for Beta {
    fn ln_f(&self, x: &Bernoulli) -> f64 {
        self.ln_f(&x.p())
    }
}

impl Sampleable<Bernoulli> for Beta {
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

// Helper struct to avoid the Booleable type parameter issue
struct BetaConjugate<X: Booleable> {
    inner: Beta,
    _phantom: PhantomData<X>,
}

impl<X: Booleable> BetaConjugate<X> {
    fn new(inner: Beta) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
    
    fn posterior(&self, stat: &BernoulliSuffStat) -> Beta {
        let (n, k) = (<BernoulliSuffStat as SuffStat<X>>::n(stat), stat.k());

        let a = self.inner.alpha() + k as f64;
        let b = self.inner.beta() + (n - k) as f64;

        Beta::new(a, b).expect("Invalid posterior parameters")
    }
}

impl<X> ConjugatePrior<X, Bernoulli> for Beta
where
    X: Booleable,
{
    type Posterior = Self;
    type MCache = f64;
    type PpCache = (f64, f64);

    fn empty_stat(&self) -> BernoulliSuffStat {
        BernoulliSuffStat::new()
    }

    #[allow(clippy::many_single_char_names)]
    fn posterior(&self, stat: &BernoulliSuffStat) -> Self {
        let helper = BetaConjugate::<X>::new(self.clone());
        helper.posterior(stat)
    }

    #[inline]
    fn ln_m_cache(&self) -> Self::MCache {
        self.alpha().ln_beta(self.beta())
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::MCache,
        stat: &BernoulliSuffStat,
    ) -> f64 {
        let helper = BetaConjugate::<X>::new(self.clone());
        let post = helper.posterior(stat);
        post.alpha().ln_beta(post.beta()) - cache
    }

    #[inline]
    fn ln_pp_cache(&self, stat: &BernoulliSuffStat) -> Self::PpCache {
        //  P(y=1 | xs) happens to be the posterior mean
        let helper = BetaConjugate::<X>::new(self.clone());
        let post = helper.posterior(stat);
        let p: f64 = post.mean().expect("Mean undefined");
        (p.ln(), (1.0 - p).ln())
    }

    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &X) -> f64 {
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

    // TODO: Update test_conjugate_prior macro for the new ConjugatePrior trait
    // test_conjugate_prior!(bool, Bernoulli, Beta, Beta::new(0.5, 1.2).unwrap());

    #[test]
    fn posterior_from_data_bool() {
        let data = vec![false, true, false, true, true];
        
        // Create sufficient statistic from the data
        let mut stat = BernoulliSuffStat::new();
        stat.observe_many(&data);

        // Use turbofish syntax to specify the type parameter
        let prior: Beta = Beta::new(1.0, 1.0).unwrap();
        let posterior = <Beta as ConjugatePrior<bool, Bernoulli>>::posterior(&prior, &stat);

        assert::close(posterior.alpha(), 4.0, TOL);
        assert::close(posterior.beta(), 3.0, TOL);
    }

    #[test]
    fn posterior_from_data_u16() {
        let data: Vec<u16> = vec![0, 1, 0, 1, 1];
        
        // Create sufficient statistic from the data
        let mut stat = BernoulliSuffStat::new();
        stat.observe_many(&data);

        // Use turbofish syntax to specify the type parameter
        let prior: Beta = Beta::new(1.0, 1.0).unwrap();
        let posterior = <Beta as ConjugatePrior<u16, Bernoulli>>::posterior(&prior, &stat);

        assert::close(posterior.alpha(), 4.0, TOL);
        assert::close(posterior.beta(), 3.0, TOL);
    }
}
