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

/// Distribution over unordered values in [0, k)
pub struct Categorical<T: CategoricalDatum> {
    // Use log weights instead to optimize for computation of ln_f
    ln_weights: Vec<f64>,
    _phantom: PhantomData<T>,
}

impl<T: CategoricalDatum> Categorical<T> {
    /// Construct a new Categorical distribution from weights
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate assert;
    /// extern crate rv;
    ///
    /// use rv::traits::*;
    /// use rv::dist::Categorical;
    ///
    /// let weights: Vec<f64> = vec![4.0, 2.0, 3.0, 1.0];
    /// let cat = Categorical::<u8>::new(&weights);
    ///
    /// assert!(cat.contains(&0));
    /// assert!(cat.contains(&3));
    /// assert!(!cat.contains(&4));
    ///
    /// assert::close(cat.pmf(&0), 0.4, 10E-12);
    /// ```
    pub fn new(weights: &Vec<f64>) -> Self {
        let ln_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        let ln_norm = logsumexp(&ln_weights);
        let normed_weights =
            ln_weights.iter().map(|lnw| lnw - ln_norm).collect();
        Categorical::from_ln_weights(normed_weights)
    }

    /// Build a Categorical distribution from normalized log weights
    pub fn from_ln_weights(ln_weights: Vec<f64>) -> Self {
        assert!(logsumexp(&ln_weights).abs() < 10E-12);
        Categorical {
            ln_weights: ln_weights,
            _phantom: PhantomData,
        }
    }

    /// Creates a Categorical distribution over [0, k) with uniform weights
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

        impl DiscreteDistr for Categorical<$kind> {}

        impl Mode for Categorical<$kind> {
            fn mode(&self) -> Option<$kind> {
                // FIXME: Return None if more than one max value
                let max_ixs = argmax(&self.ln_weights);
                if max_ixs.len() > 1 {
                    None
                } else {
                    Some(max_ixs[0] as $kind)
                }
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

    #[test]
    fn ln_weights_should_logsumexp_to_1() {
        // weights the def do not sum to 1
        let weights: Vec<f64> = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let cat = Categorical::<u8>::new(&weights);
        assert::close(logsumexp(&cat.ln_weights), 0.0, TOL);
    }

    #[test]
    fn ln_weights_unifor_should_logsumexp_to_1() {
        let cat = Categorical::<u8>::uniform(5);
        let ln_weight = (1_f64 / 5.0).ln();

        cat.ln_weights
            .iter()
            .for_each(|&ln_w| assert::close(ln_w, ln_weight, TOL));
        assert::close(logsumexp(&cat.ln_weights), 0.0, TOL);
    }

    #[test]
    fn ln_f_should_be_ln_weight() {
        let cat = Categorical::<u8>::new(&vec![2.0, 1.0, 2.0, 4.0, 3.0]);
        assert::close(cat.ln_f(&0), -1.791759469228055, TOL);
        assert::close(cat.ln_f(&1), -2.4849066497880004, TOL);
        assert::close(cat.ln_f(&2), -1.791759469228055, TOL);
        assert::close(cat.ln_f(&3), -1.0986122886681098, TOL);
        assert::close(cat.ln_f(&4), -1.3862943611198906, TOL);
    }

    #[test]
    fn ln_pmf_should_be_ln_weight() {
        let cat = Categorical::<u8>::new(&vec![2.0, 1.0, 2.0, 4.0, 3.0]);
        assert::close(cat.ln_pmf(&0), -1.791759469228055, TOL);
        assert::close(cat.ln_pmf(&1), -2.4849066497880004, TOL);
        assert::close(cat.ln_pmf(&2), -1.791759469228055, TOL);
        assert::close(cat.ln_pmf(&3), -1.0986122886681098, TOL);
        assert::close(cat.ln_pmf(&4), -1.3862943611198906, TOL);
    }

    #[test]
    fn draw_should_return_numbers_in_0_to_k() {
        let mut rng = rand::thread_rng();
        let k = 5;
        let cat = Categorical::<u8>::uniform(k);
        let mut counts = vec![0; k];
        for _ in 0..1000 {
            let k = cat.draw(&mut rng);
            counts[k as usize] += 1;
            assert!(k < 5);
        }
        assert!(counts.iter().all(|&ct| ct > 0));
    }

    #[test]
    fn sample_should_return_the_correct_number_of_draws() {
        let mut rng = rand::thread_rng();
        let cat = Categorical::<u8>::uniform(5);
        let xs = cat.sample(103, &mut rng);
        assert_eq!(xs.len(), 103);
    }

    #[test]
    fn should_contain_zero_to_one_minus_k() {
        let k = 3;
        let cat = Categorical::<u8>::uniform(k);

        assert!(cat.contains(&0));
        assert!(cat.contains(&1));
        assert!(cat.contains(&2));
        assert!(!cat.contains(&3));
    }

    #[test]
    fn uniform_mode_does_not_exist() {
        let cat = Categorical::<u8>::uniform(4);
        assert!(cat.mode().is_none());
    }

    #[test]
    fn mode() {
        let cat = Categorical::<u8>::new(&vec![1.0, 2.0, 3.0, 1.0]);
        assert_eq!(cat.mode().unwrap(), 2);
    }
}