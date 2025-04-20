use rand::Rng;
use special::Beta as SBeta;
use std::marker::PhantomData;

use crate::data::{Booleable, BernoulliSuffStat};
use crate::dist::{Bernoulli, Beta, UnitPowerLaw};
use crate::traits::*;

impl HasDensity<Bernoulli> for UnitPowerLaw {
    fn ln_f(&self, x: &Bernoulli) -> f64 {
        self.ln_f(&x.p())
    }
}

impl Sampleable<Bernoulli> for UnitPowerLaw {
    fn draw<R: Rng>(&self, mut rng: &mut R) -> Bernoulli {
        let p: f64 = self.draw(&mut rng);
        Bernoulli::new(p).expect("Failed to draw valid weight")
    }
}

impl Support<Bernoulli> for UnitPowerLaw {
    fn supports(&self, x: &Bernoulli) -> bool {
        0.0 < x.p() && x.p() < 1.0
    }
}

impl ContinuousDistr<Bernoulli> for UnitPowerLaw {}

// Helper struct to avoid the Booleable type parameter issue
struct UnitPowerLawConjugate<X: Booleable> {
    inner: UnitPowerLaw,
    _phantom: PhantomData<X>,
}

impl<X: Booleable> UnitPowerLawConjugate<X> {
    fn new(inner: UnitPowerLaw) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
    
    fn posterior(&self, stat: &BernoulliSuffStat) -> Beta {
        let (n, k) = (<BernoulliSuffStat as SuffStat<X>>::n(stat), stat.k());

        let a = self.inner.alpha() + k as f64;
        let b = (1 + (n - k)) as f64;

        Beta::new(a, b).expect("Invalid posterior parameters")
    }
}

impl<X> ConjugatePrior<X, Bernoulli> for UnitPowerLaw 
where 
    X: Booleable,
{
    type Posterior = Beta;
    type MCache = f64;
    type PpCache = (f64, f64);

    fn empty_stat(&self) -> BernoulliSuffStat {
        BernoulliSuffStat::new()
    }

    #[allow(clippy::many_single_char_names)]
    fn posterior(&self, stat: &BernoulliSuffStat) -> Beta {
        let helper = UnitPowerLawConjugate::<X>::new(self.clone());
        helper.posterior(stat)
    }

    #[inline]
    fn ln_m_cache(&self) -> Self::MCache {
        -self.alpha_ln()
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::MCache,
        stat: &BernoulliSuffStat,
    ) -> f64 {
        let helper = UnitPowerLawConjugate::<X>::new(self.clone());
        let post: Beta = helper.posterior(stat);
        post.alpha().ln_beta(post.beta()) - cache
    }

    #[inline]
    fn ln_pp_cache(&self, stat: &BernoulliSuffStat) -> Self::PpCache {
        //  P(y=1 | xs) happens to be the posterior mean
        let helper = UnitPowerLawConjugate::<X>::new(self.clone());
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
    use crate::data::BernoulliSuffStat;

    const TOL: f64 = 1E-12;

    #[test]
    fn posterior_from_data_bool() {
        let data = vec![false, true, false, true, true];
        
        // Create sufficient statistic from the data
        let mut stat = BernoulliSuffStat::new();
        stat.observe_many(&data);

        // Use turbofish syntax to specify the type parameter
        let prior: UnitPowerLaw = UnitPowerLaw::new(1.0).unwrap();
        let posterior = <UnitPowerLaw as ConjugatePrior<bool, Bernoulli>>::posterior(&prior, &stat);

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
        let prior: UnitPowerLaw = UnitPowerLaw::new(1.0).unwrap();
        let posterior = <UnitPowerLaw as ConjugatePrior<u16, Bernoulli>>::posterior(&prior, &stat);

        assert::close(posterior.alpha(), 4.0, TOL);
        assert::close(posterior.beta(), 3.0, TOL);
    }
}
