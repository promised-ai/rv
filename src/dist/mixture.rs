extern crate rand;

use self::rand::Rng;
use std::marker::PhantomData;
use std::io;
use misc::{logsumexp, pflip};
use traits::*;


pub struct Mixture<X, Fx>
    where Fx: Rv<X>
{
    pub weights: Vec<f64>,
    pub components: Vec<Fx>,
    phantom: PhantomData<X>
}

impl<X, Fx> Mixture<X, Fx>
    where Fx: Rv<X>
{
    pub fn new(weights: Vec<f64>, components: Vec<Fx>) -> io::Result<Self> {
        let weights_sum = weights.iter().fold(0.0, |acc, &w| acc + w);
        if (weights_sum - 1.0).abs() > 1E-12 {
            Ok(Mixture { weights, components, phantom: PhantomData })
        } else {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "weights must sum to 1");
            Err(err)
        }
    }
}

impl<X, Fx> Rv<X> for Mixture<X, Fx>
    where Fx: Rv<X>
{
    fn ln_f(&self, x: &X) -> f64 {
        let lfs: Vec<f64> = self
            .weights
            .iter()
            .zip(self.components.iter())
            .map(|(&w, cpnt)| w.ln() + cpnt.ln_f(&x))
            .collect();

        logsumexp(&lfs)
    }

    fn f(&self, x: &X) -> f64 {
        self.weights
            .iter()
            .zip(self.components.iter())
            .fold(0.0, |acc, (&w, cpnt)| acc + w * cpnt.f(&x))
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> X {
        let k: usize = pflip(&self.weights, 1, &mut rng)[0];
        self.components[k].draw(&mut rng)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<X> {
        pflip(&self.weights, n, &mut rng)
            .iter()
            .map(|&k| self.components[k].draw(&mut rng))
            .collect()
    }
}

impl<X, Fx> Support<X> for Mixture<X, Fx>
    where Fx: Rv<X> + Support<X>
{
    fn supports(&self, x: &X) -> bool {
        self.components.iter().any(|cpnt| cpnt.supports(&x))
    }
}

impl<X, Fx> ContinuousDistr<X> for Mixture<X, Fx>
    where Fx: Rv<X> + ContinuousDistr<X> { }

impl<X, Fx> DiscreteDistr<X> for Mixture<X, Fx>
    where Fx: Rv<X> + DiscreteDistr<X> { }
