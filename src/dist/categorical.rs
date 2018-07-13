extern crate num;
extern crate rand;

use self::num::traits::FromPrimitive;
use self::rand::Rng;
use std::marker::PhantomData;
use traits::*;
use utils::{argmax, ln_pflip, logsumexp};

pub trait CategoricalDatum:
    Sized + Into<usize> + Sync + Copy + FromPrimitive
{
}

impl<T> CategoricalDatum for T where
    T: Clone + Into<usize> + Sync + Copy + FromPrimitive
{}

pub struct Categorical<T: CategoricalDatum> {
    // Use log weights instead to optimize for computation of ln_f
    ln_weights: Vec<f64>,
    _phantom: PhantomData<T>,
}

impl<T: CategoricalDatum> Categorical<T> {
    pub fn new(weights: &Vec<f64>) -> Self {
        let ln_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        let ln_norm = logsumexp(&ln_weights);
        Categorical {
            ln_weights: ln_weights.iter().map(|lnw| lnw - ln_norm).collect(),
            _phantom: PhantomData,
        }
    }

    pub fn uniform(k: usize) -> Self {
        let lnp = (1.0 / k as f64).ln();

        Categorical {
            ln_weights: vec![lnp; k],
            _phantom: PhantomData,
        }
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv for Categorical<$kind> {
            type DatumType = $kind;

            fn ln_f(&self, x: &$kind) -> f64 {
                let ix: usize = (*x).into();
                self.ln_weights[ix]
            }

            fn ln_normalizer(&self) -> f64 {
                0.0
            }

            fn draw<R: Rng>(&self, mut rng: &mut R) -> $kind {
                let ix = ln_pflip(&self.ln_weights, 1, &mut rng)[0];
                FromPrimitive::from_usize(ix).unwrap()
            }

            fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<$kind> {
                ln_pflip(&self.ln_weights, n, &mut rng)
                    .iter()
                    .map(|&ix| FromPrimitive::from_usize(ix).unwrap())
                    .collect()
            }
        }

        impl Support for Categorical<$kind> {
            fn contains(&self, x: &$kind) -> bool {
                (*x as usize) < self.ln_weights.len()
            }
        }

        impl Cdf for Categorical<$kind> {
            fn cdf(&self, x: &$kind) -> f64 {
                let ix = *x as usize;
                self.ln_weights[0..ix]
                    .iter()
                    .fold(0.0, |acc, ln_weight| acc + ln_weight.exp())
            }
        }

        impl DiscreteDistr for Categorical<$kind> {}

        impl Mean for Categorical<$kind> {
            type MeanType = f64;

            fn mean(&self) -> Option<f64> {
                let m = self
                    .ln_weights
                    .iter()
                    .enumerate()
                    .fold(0.0, |acc, (ix, ln_weight)| {
                        acc + (ix as f64) * ln_weight.exp()
                    });
                Some(m)
            }
        }

        impl Mode for Categorical<$kind> {
            fn mode(&self) -> Option<$kind> {
                // FIXME: Return None if more than one max value
                Some(argmax(&self.ln_weights) as $kind)
            }
        }

        impl Entropy for Categorical<$kind> {
            fn entropy(&self) -> f64 {
                self.ln_weights.iter().fold(0.0, |acc, ln_weight| {
                    acc - ln_weight.exp() * ln_weight
                })
            }
        }
    };
}

impl_traits!(u8);
impl_traits!(u16);
impl_traits!(usize);

#[cfg(test)]
mod tests {
    use super::*;
    extern crate assert;

    const TOL: f64 = 1E-12;
}
