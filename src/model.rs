extern crate rand;

use self::rand::Rng;
use std::marker::PhantomData;
use traits::*;

/// A wrapper for a complete conjugate model
pub struct ConjugateModel<'pr, X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: 'pr + ConjugatePrior<X, Fx>,
{
    prior: &'pr Pr,
    suffstat: Fx::Stat,
    _phantom: PhantomData<X>,
}

impl<'pr, X, Fx, Pr> ConjugateModel<'pr, X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: 'pr + ConjugatePrior<X, Fx>,
{
    pub fn new(fx: &Fx, pr: &'pr Pr) -> Self {
        ConjugateModel {
            prior: pr,
            suffstat: fx.empty_suffstat(),
            _phantom: PhantomData,
        }
    }

    pub fn ln_m(&self) -> f64 {
        self.prior.ln_m(&self.obs())
    }

    pub fn ln_pp(&self, y: &X) -> f64 {
        self.prior.ln_pp(&y, &self.obs())
    }

    pub fn posterior(&self) -> Pr {
        self.prior.posterior(&self.obs())
    }

    fn obs(&self) -> DataOrSuffStat<X, Fx> {
        DataOrSuffStat::SuffStat(&self.suffstat)
    }
}

impl<'pr, X, Fx, Pr> SuffStat<X> for ConjugateModel<'pr, X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: 'pr + ConjugatePrior<X, Fx>,
{
    fn observe(&mut self, x: &X) {
        self.suffstat.observe(&x);
    }

    fn forget(&mut self, x: &X) {
        self.suffstat.forget(&x);
    }
}

impl<'pr, X, Fx, Pr> Rv<X> for ConjugateModel<'pr, X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: 'pr + ConjugatePrior<X, Fx>,
{
    fn ln_f(&self, x: &X) -> f64 {
        self.prior.ln_pp(&x, &self.obs())
    }

    #[inline]
    fn ln_normalizer(&self) -> f64 {
        0.0
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> X {
        let post = self.posterior();
        let fx: Fx = post.draw(&mut rng);
        fx.draw(&mut rng)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<X> {
        let post = self.posterior();
        (0..n)
            .map(|_| {
                let fx: Fx = post.draw(&mut rng);
                fx.draw(&mut rng)
            })
            .collect()
    }
}
