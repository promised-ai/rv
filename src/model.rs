use crate::data::DataOrSuffStat;
use crate::traits::*;
use rand::Rng;
use std::marker::PhantomData;
use std::sync::Arc;

/// A wrapper for a complete conjugate model
///
/// # Paramters
///
/// `X`: The type of the data/observations to be modeled
/// `Fx`: The type of the likelihood, *f(x|θ)*
/// `Pr`: The type of the prior on the parameters of `Fx`, π(θ)
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ConjugateModel<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
{
    /// Pointer to an `Rv` implementing `ConjugatePrior` for `Fx`
    prior: Arc<Pr>,
    /// A `SuffStat` for `Fx`
    suffstat: Fx::Stat,
    _phantom: PhantomData<X>,
}

impl<X, Fx, Pr> ConjugateModel<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
{
    /// Create a new conjugate model
    ///
    /// # Arguments
    ///
    /// *fx*:
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use rv::prelude::*;
    /// use rv::ConjugateModel;
    ///
    /// let pr = Arc::new(Beta::jeffreys());
    /// let fx = Bernoulli::uniform();
    /// let model = ConjugateModel::<bool, Bernoulli, Beta>::new(&fx, pr);
    /// ```
    pub fn new(fx: &Fx, pr: Arc<Pr>) -> Self {
        ConjugateModel {
            prior: pr,
            suffstat: fx.empty_suffstat(),
            _phantom: PhantomData,
        }
    }

    /// Log marginal likelihood, *f(obs)*
    pub fn ln_m(&self) -> f64 {
        self.prior.ln_m(&self.obs())
    }

    /// Log posterior predictive, *f(y|obs)*
    pub fn ln_pp(&self, y: &X) -> f64 {
        self.prior.ln_pp(&y, &self.obs())
    }

    /// Return the posterior distribution
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use rv::prelude::*;
    /// use rv::ConjugateModel;
    ///
    /// let flips: Vec<bool> = vec![true, false, true, false, false, false];
    ///
    /// let pr = Arc::new(Beta::new(1.0, 1.0).unwrap());
    /// let fx = Bernoulli::uniform();
    /// let mut model = ConjugateModel::<bool, Bernoulli, Beta>::new(&fx, pr);
    ///
    /// model.observe_many(&flips);
    ///
    /// let post = model.posterior();
    ///
    /// assert_eq!(post, Beta::new(3.0, 5.0).unwrap());
    /// ```
    pub fn posterior(&self) -> Pr::Posterior {
        self.prior.posterior(&self.obs())
    }

    /// Return the observations
    fn obs(&self) -> DataOrSuffStat<X, Fx> {
        DataOrSuffStat::SuffStat(&self.suffstat)
    }
}

impl<X, Fx, Pr> SuffStat<X> for ConjugateModel<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
{
    fn n(&self) -> usize {
        self.suffstat.n()
    }

    fn observe(&mut self, x: &X) {
        self.suffstat.observe(&x);
    }

    fn forget(&mut self, x: &X) {
        self.suffstat.forget(&x);
    }
}

impl<X, Fx, Pr> Rv<X> for ConjugateModel<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
{
    fn ln_f(&self, x: &X) -> f64 {
        self.prior.ln_pp(&x, &self.obs())
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
