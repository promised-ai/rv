use crate::traits::{
    ConjugatePrior, DataOrSuffStat, HasDensity, HasSuffStat, Rv, Sampleable,
    SuffStat,
};
use rand::Rng;
use std::marker::PhantomData;
use std::sync::Arc;

/// A wrapper for a complete conjugate model
///
/// # Parameters
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
        self.prior.ln_pp(y, &self.obs())
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
    fn obs(&self) -> DataOrSuffStat<'_, X, Fx> {
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
        self.suffstat.observe(x);
    }

    fn forget(&mut self, x: &X) {
        self.suffstat.forget(x);
    }

    fn merge(&mut self, other: Self) {
        self.suffstat.merge(other.suffstat);
    }
}

impl<X, Fx, Pr> HasDensity<X> for ConjugateModel<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
{
    fn ln_f(&self, x: &X) -> f64 {
        self.prior.ln_pp(x, &self.obs())
    }
}

impl<X, Fx, Pr> Sampleable<X> for ConjugateModel<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
{
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

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        dist::{Bernoulli, Beta, ChiSquared},
        traits::Cdf,
    };

    #[test]
    fn basic() {
        let mut model = ConjugateModel::new(
            &Bernoulli::uniform(),
            Arc::new(Beta::jeffreys()),
        );

        model.observe_many(&[true, false]);
        assert_eq!(model.n(), 2);
        assert::close(model.ln_m(), -8.0_f64.ln(), 1e-6);

        model.forget(&true);
        assert_eq!(model.n(), 1);
        assert::close(model.ln_m(), 0.5_f64.ln(), 1e-6);

        let mut other_model = ConjugateModel::new(
            &Bernoulli::uniform(),
            Arc::new(Beta::jeffreys()),
        );

        other_model.observe_many(&[true, true]);
        model.merge(other_model);

        assert_eq!(model.n(), 3);
        assert_eq!(model.suffstat.k(), 2);
    }

    #[test]
    fn density() {
        let mut model = ConjugateModel::new(
            &Bernoulli::uniform(),
            Arc::new(Beta::jeffreys()),
        );

        model.observe_many(&[true, false]);

        assert::close(model.ln_f(&true), (1.5_f64 / (1.5 + 1.5)).ln(), 1e-6);
        assert::close(model.ln_pp(&true), (1.5_f64 / (1.5 + 1.5)).ln(), 1e-6);
    }

    #[test]
    fn sample() {
        let mut rng = SmallRng::seed_from_u64(0x1234);

        let mut model = ConjugateModel::new(
            &Bernoulli::uniform(),
            Arc::new(Beta::jeffreys()),
        );

        model.observe_many(&[true, false]);

        let sample = model.sample(1000, &mut rng);

        let alpha = 1.5;
        let beta = 1.5;
        let p_expected = alpha / (alpha + beta);

        let p_observed = sample
            .iter()
            .map(|x| if *x { 1.0 } else { 0.0 })
            .sum::<f64>()
            / (sample.len() as f64);

        let x2 = (p_observed - p_expected).powi(2) / p_expected
            + (p_observed - p_expected).powi(2) / (1.0 - p_expected);

        assert!(ChiSquared::new_unchecked(1.0).cdf(&x2) < 0.05);
    }
}
