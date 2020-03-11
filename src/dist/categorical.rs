//! Categorical distribution of x<sub>k</sub> in {0, 1, ..., k-1}
#[cfg(feature = "serde1")]
use serde_derive::{Deserialize, Serialize};

use crate::data::{CategoricalDatum, CategoricalSuffStat};
use crate::impl_display;
use crate::misc::{argmax, ln_pflip, logsumexp, vec_to_string};
use crate::traits::*;
use rand::Rng;
use std::fmt;

/// [Categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution)
/// over unordered values in [0, k).
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Categorical {
    // Use log weights instead to optimize for computation of ln_f
    ln_weights: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum CategoricalError {
    /// One or more of the weights is infinite or NaN
    NonFiniteWeight { ix: usize, ln: bool, weight: f64 },
    /// One or more of the weights is less than zero
    NegativeWeight { ix: usize, weight: f64 },
    /// The weights do not sum to 1
    WeightsDoNotSumToOne { ln: bool, sum: f64 },
    /// Weights has not entries
    EmptyWights,
}

impl Categorical {
    /// Construct a new Categorical distribution from weights
    ///
    /// # Arguments
    /// - weights: A vector describing the proportional likelihood of each
    ///   outcome. The weights must all be positive, but do not need to sum to
    ///   1 because they will be normalized in the constructor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rv::traits::*;
    /// # use rv::dist::Categorical;
    /// let weights: Vec<f64> = vec![4.0, 2.0, 3.0, 1.0];
    /// let cat = Categorical::new(&weights).unwrap();
    ///
    /// assert!(cat.supports(&0_u8));
    /// assert!(cat.supports(&3_u8));
    /// assert!(!cat.supports(&4_u8));
    ///
    /// assert::close(cat.pmf(&0_u8), 0.4, 1E-12);
    /// ```
    pub fn new(weights: &[f64]) -> Result<Self, CategoricalError> {
        if weights.is_empty() {
            return Err(CategoricalError::EmptyWights);
        }

        weights.iter().enumerate().try_for_each(|(ix, &weight)| {
            if weight < 0.0 {
                Err(CategoricalError::NegativeWeight { ix, weight })
            } else if !weight.is_finite() {
                Err(CategoricalError::NonFiniteWeight {
                    ix,
                    ln: false,
                    weight,
                })
            } else {
                Ok(())
            }
        })?;

        let ln_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        let ln_norm = logsumexp(&ln_weights);
        let normed_weights =
            ln_weights.iter().map(|lnw| lnw - ln_norm).collect();
        Ok(Categorical::new_unchecked(normed_weights))
    }

    /// Build a Categorical distribution from normalized log weights
    ///
    /// # Arguments
    /// - ln_weights: A vector describing the proportional likelihood of each
    ///   outcome in log space. sum(exp(ln_weights)) must be equal to 1.
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::traits::*;
    /// # use rv::dist::Categorical;
    /// let ln_weights: Vec<f64> = vec![
    ///     -2.3025850929940455,
    ///     -1.6094379124341003,
    ///     -1.2039728043259361,
    ///     -0.916290731874155
    /// ];
    ///
    /// let cat = Categorical::from_ln_weights(ln_weights).unwrap();
    ///
    /// assert::close(cat.pmf(&0_u8), 0.1, 1E-12);
    /// assert::close(cat.pmf(&1_u8), 0.2, 1E-12);
    /// assert::close(cat.pmf(&2_u8), 0.3, 1E-12);
    /// assert::close(cat.pmf(&3_u8), 0.4, 1E-12);
    /// ```
    pub fn from_ln_weights(
        ln_weights: Vec<f64>,
    ) -> Result<Self, CategoricalError> {
        if ln_weights.is_empty() {
            return Err(CategoricalError::EmptyWights);
        }

        ln_weights
            .iter()
            .enumerate()
            .try_for_each(|(ix, &weight)| {
                if weight.is_finite() {
                    Ok(())
                } else {
                    Err(CategoricalError::NonFiniteWeight {
                        ix,
                        ln: false,
                        weight,
                    })
                }
            })?;

        let sum = logsumexp(&ln_weights).abs();
        if sum < 10E-12 {
            Ok(Categorical { ln_weights })
        } else {
            Err(CategoricalError::WeightsDoNotSumToOne { ln: true, sum })
        }
    }

    /// Creates a new Categorical without checking whether the ln weights are
    /// valid.
    #[inline]
    fn new_unchecked(ln_weights: Vec<f64>) -> Self {
        Categorical { ln_weights }
    }

    /// Creates a Categorical distribution over [0, k) with uniform weights
    #[inline]
    pub fn uniform(k: usize) -> Self {
        let lnp = (1.0 / k as f64).ln();
        Categorical::new_unchecked(vec![lnp; k])
    }

    /// Return the weights (`exp(ln_weights)`)
    #[inline]
    pub fn weights(&self) -> Vec<f64> {
        self.ln_weights.iter().map(|&w| w.exp()).collect()
    }

    /// Get the number of possible outcomes
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Categorical;
    /// let cat = Categorical::uniform(4);
    /// assert_eq!(cat.k(), 4);
    /// ```
    #[inline]
    pub fn k(&self) -> usize {
        self.ln_weights.len()
    }

    /// Get a reference to the weights
    #[inline]
    pub fn ln_weights(&self) -> &Vec<f64> {
        &self.ln_weights
    }
}

impl From<&Categorical> for String {
    fn from(cat: &Categorical) -> String {
        let weights = vec_to_string(&cat.weights(), 5);
        format!("Categorical({};, {})", cat.k(), weights)
    }
}

impl_display!(Categorical);

impl<X: CategoricalDatum> Rv<X> for Categorical {
    fn ln_f(&self, x: &X) -> f64 {
        let ix: usize = x.into_usize();
        self.ln_weights[ix]
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> X {
        let ix = ln_pflip(&self.ln_weights, 1, true, &mut rng)[0];
        CategoricalDatum::from_usize(ix)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<X> {
        ln_pflip(&self.ln_weights, n, true, &mut rng)
            .iter()
            .map(|&ix| CategoricalDatum::from_usize(ix))
            .collect()
    }
}

impl<X: CategoricalDatum> Support<X> for Categorical {
    fn supports(&self, x: &X) -> bool {
        let ix: usize = x.into_usize();
        ix < self.ln_weights.len()
    }
}

impl<X: CategoricalDatum> DiscreteDistr<X> for Categorical {}

impl<X: CategoricalDatum> Cdf<X> for Categorical {
    fn cdf(&self, x: &X) -> f64 {
        let xu: usize = x.into_usize();
        self.ln_weights
            .iter()
            .take(xu + 1)
            .fold(0.0, |acc, &w| w.exp() + acc)
    }
}

impl<X: CategoricalDatum> Mode<X> for Categorical {
    fn mode(&self) -> Option<X> {
        // FIXME: Return None if more than one max value
        let max_ixs = argmax(&self.ln_weights);
        if max_ixs.len() > 1 {
            None
        } else {
            Some(CategoricalDatum::from_usize(max_ixs[0]))
        }
    }
}

impl Entropy for Categorical {
    fn entropy(&self) -> f64 {
        self.ln_weights
            .iter()
            .fold(0.0, |acc, ln_weight| acc - ln_weight.exp() * ln_weight)
    }
}

impl<X: CategoricalDatum> HasSuffStat<X> for Categorical {
    type Stat = CategoricalSuffStat;
    fn empty_suffstat(&self) -> Self::Stat {
        CategoricalSuffStat::new(self.k())
    }
}

impl KlDivergence for Categorical {
    fn kl(&self, other: &Self) -> f64 {
        self.ln_weights
            .iter()
            .zip(other.ln_weights.iter())
            .fold(0.0, |acc, (&ws, &wo)| acc + ws.exp() * (ws - wo))
    }
}

impl fmt::Display for CategoricalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteWeight { ix, ln, weight } if *ln => {
                write!(f, "non-finite ln weight at index {}: {}", ix, weight)
            }
            Self::NonFiniteWeight { ix, weight, .. } => {
                write!(f, "non-finite weight at index {}: {}", ix, weight)
            }
            Self::NegativeWeight { ix, weight } => {
                write!(f, "negative weight at index {}: {}", ix, weight)
            }
            Self::WeightsDoNotSumToOne { ln, sum } if *ln => {
                write!(f, "ln weights sum to {}, should sum to zero", sum)
            }
            Self::WeightsDoNotSumToOne { sum, .. } => {
                write!(f, "weights sum to {}, should sum to one", sum)
            }
            Self::EmptyWights => write!(f, "empty weights vector"),
        }
    }
}

impl std::error::Error for CategoricalError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::x2_test;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;
    const N_TRIES: usize = 5;
    const X2_PVAL: f64 = 0.2;

    test_basic_impls!(Categorical::uniform(3));

    #[test]
    fn ln_weights_should_logsumexp_to_1() {
        // weights the def do not sum to 1
        let weights: Vec<f64> = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let cat = Categorical::new(&weights).unwrap();
        assert::close(logsumexp(&cat.ln_weights), 0.0, TOL);
    }

    #[test]
    fn ln_weights_unifor_should_logsumexp_to_1() {
        let cat = Categorical::uniform(5);
        let ln_weight = (1_f64 / 5.0).ln();

        cat.ln_weights
            .iter()
            .for_each(|&ln_w| assert::close(ln_w, ln_weight, TOL));
        assert::close(logsumexp(&cat.ln_weights), 0.0, TOL);
    }

    #[test]
    fn ln_f_should_be_ln_weight() {
        let cat = Categorical::new(&vec![2.0, 1.0, 2.0, 4.0, 3.0]).unwrap();
        assert::close(cat.ln_f(&0_u8), -1.791759469228055, TOL);
        assert::close(cat.ln_f(&1_u8), -2.4849066497880004, TOL);
        assert::close(cat.ln_f(&2_u8), -1.791759469228055, TOL);
        assert::close(cat.ln_f(&3_u8), -1.0986122886681098, TOL);
        assert::close(cat.ln_f(&4_u8), -1.3862943611198906, TOL);
    }

    #[test]
    fn ln_pmf_should_be_ln_weight() {
        let cat = Categorical::new(&vec![2.0, 1.0, 2.0, 4.0, 3.0]).unwrap();
        assert::close(cat.ln_pmf(&0_u16), -1.791759469228055, TOL);
        assert::close(cat.ln_pmf(&1_u16), -2.4849066497880004, TOL);
        assert::close(cat.ln_pmf(&2_u16), -1.791759469228055, TOL);
        assert::close(cat.ln_pmf(&3_u16), -1.0986122886681098, TOL);
        assert::close(cat.ln_pmf(&4_u16), -1.3862943611198906, TOL);
    }

    #[test]
    fn draw_should_return_numbers_in_0_to_k() {
        let mut rng = rand::thread_rng();
        let k = 5;
        let cat = Categorical::uniform(k);
        let mut counts = vec![0; k];
        for _ in 0..1000 {
            let ix: usize = cat.draw(&mut rng);
            counts[ix] += 1;
            assert!(ix < 5);
        }
        assert!(counts.iter().all(|&ct| ct > 0));
    }

    #[test]
    fn sample_should_return_the_correct_number_of_draws() {
        let mut rng = rand::thread_rng();
        let cat = Categorical::uniform(5);
        let xs: Vec<u8> = cat.sample(103, &mut rng);
        assert_eq!(xs.len(), 103);
    }

    #[test]
    fn should_contain_zero_to_one_minus_k() {
        let k = 3;
        let cat = Categorical::uniform(k);

        assert!(cat.supports(&0_usize));
        assert!(cat.supports(&1_usize));
        assert!(cat.supports(&2_usize));
        assert!(!cat.supports(&3_usize));
    }

    #[test]
    fn uniform_mode_does_not_exist() {
        let mode: Option<u8> = Categorical::uniform(4).mode();
        assert!(mode.is_none());
    }

    #[test]
    fn mode() {
        let cat = Categorical::new(&vec![1.0, 2.0, 3.0, 1.0]).unwrap();
        let mode: usize = cat.mode().unwrap();
        assert_eq!(mode, 2);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let cat = Categorical::new(&vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ps: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; 4];
            let xs: Vec<usize> = cat.sample(1000, &mut rng);
            xs.iter().for_each(|&x| f_obs[x] += 1);
            let (_, p) = x2_test(&f_obs, &ps);
            if p > X2_PVAL {
                acc + 1
            } else {
                acc
            }
        });
        assert!(passes > 0);
    }

    #[test]
    fn kl() {
        let cat1 = Categorical::new(&vec![
            0.2280317, 0.1506706, 0.33620052, 0.13911904, 0.14597815,
        ])
        .unwrap();
        let cat2 = Categorical::new(&vec![
            0.30050657, 0.04237857, 0.20973238, 0.32858568, 0.1187968,
        ])
        .unwrap();

        // Allow extra error for the normalization
        assert::close(cat1.kl(&cat2), 0.1973394327976612, 1E-7);
        assert::close(cat2.kl(&cat1), 0.18814408198625582, 1E-7);
    }

    #[test]
    fn cdf() {
        let cat = Categorical::new(&vec![1.0, 2.0, 4.0, 3.0]).unwrap();
        assert::close(cat.cdf(&0_u8), 0.1, TOL);
        assert::close(cat.cdf(&1_u8), 0.3, TOL);
        assert::close(cat.cdf(&2_u8), 0.7, TOL);
        assert::close(cat.cdf(&3_u8), 1.0, TOL);
    }
}
