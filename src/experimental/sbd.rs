use peroxide::fuga::Algorithm;
use rand::seq::SliceRandom;
use rand::Rng;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use super::StickSequence;
use crate::traits::*;

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Sbd<B: Rv<f64> + Clone> {
    pub sticks: StickSequence<B>,
}

impl<B: Rv<f64> + Clone> Sbd<B> {
    pub fn new(breaker: B) -> Sbd<B> {
        let sticks = StickSequence::new(breaker, None);
        Self { sticks }
    }

    pub fn invccdf(&self, u: f64) -> usize {
        self.sticks.extendmap_ccdf(
            |ccdf| ccdf.last().unwrap() < &u,
            |ccdf| ccdf.iter().position(|q| *q < u).unwrap() - 1,
        )
    }

    fn multi_invccdf_sorted(&self, ps: &[f64]) -> Vec<usize> {
        let n = ps.len();
        self.sticks.extendmap_ccdf(
            // Note that ccdf is decreasing, but xs is increasing
            |ccdf| ccdf.last().unwrap() < ps.first().unwrap(),
            |ccdf| {
                let mut result: Vec<usize> = Vec::with_capacity(n);

                // We'll start at the end of the sorted uniforms (the largest value)
                let mut i: usize = n - 1;
                for q in ccdf.iter().skip(1).enumerate() {
                    while ps[i] > *q.1 {
                        result.push(q.0);
                        if i == 0 {
                            break;
                        } else {
                            i -= 1;
                        }
                    }
                }
                result
            },
        )
    }
}

impl<B: Rv<f64> + Clone> Support<usize> for Sbd<B> {
    fn supports(&self, x: &usize) -> bool {
        x.ge(&0)
    }
}

impl<B: Rv<f64> + Clone> Cdf<usize> for Sbd<B> {
    fn sf(&self, x: &usize) -> f64 {
        self.sticks.ccdf(x + 1)
    }

    fn cdf(&self, x: &usize) -> f64 {
        1.0 - self.sf(x)
    }
}

impl<B: Rv<f64> + Clone> InverseCdf<usize> for Sbd<B> {
    fn invcdf(&self, p: f64) -> usize {
        self.invccdf(1.0 - p)
    }
}

impl<B: Rv<f64> + Clone> DiscreteDistr<usize> for Sbd<B> {}

impl<B: Rv<f64> + Clone> Mode<usize> for Sbd<B> {
    fn mode(&self) -> Option<usize> {
        let w0 = self.sticks.weight(0);
        // Once the unallocated mass is less than that of first stick, the
        // allocated mass is guaranteed to contain the mode.
        let n = self.sticks.extendmap_ccdf(
            |ccdf| ccdf.last().unwrap() < &w0,
            |ccdf| {
                let weights: Vec<f64> =
                    ccdf.windows(2).map(|qs| qs[0] - qs[1]).collect();
                weights.arg_max()
            },
        );
        Some(n)
    }
}

// Normalizing a cumulative sum of Exp(1) random variables yields sorted uniforms
fn sorted_uniforms<R: Rng>(n: usize, rng: &mut R) -> Vec<f64> {
    let mut xs: Vec<_> = (0..n)
        .map(|_| -rng.gen::<f64>().ln())
        .scan(0.0, |state, x| {
            *state += x;
            Some(*state)
        })
        .collect();
    let max = *xs.last().unwrap() - rng.gen::<f64>().ln();
    (0..n).for_each(|i| xs[i] /= max);
    xs
}

impl<B: Rv<f64> + Clone> HasDensity<usize> for Sbd<B> {
    fn f(&self, n: &usize) -> f64 {
        let sticks = &self.sticks;
        sticks.weight(*n)
    }

    fn ln_f(&self, n: &usize) -> f64 {
        self.f(n).ln()
    }
}

impl<B: Rv<f64> + Clone> Sampleable<usize> for Sbd<B> {
    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        let u: f64 = rng.gen();
        self.invccdf(u)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<usize> {
        let ps = sorted_uniforms(n, &mut rng);
        let mut result = self.multi_invccdf_sorted(&ps);

        // At this point `result` is sorted, so we need to shuffle it.
        // Note that shuffling is O(n) but sorting is O(n log n)
        result.shuffle(&mut rng);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert::close;
    use rand::thread_rng;
    use crate::prelude::UnitPowerLaw;

    #[test]
    fn test_sorted_uniforms() {
        let mut rng = thread_rng();
        let n = 10000;
        let xs = sorted_uniforms(n, &mut rng);
        assert!(xs.len() == n);

        // Result is sorted and in the unit interval
        assert!(&0.0 < xs.first().unwrap());
        assert!(xs.last().unwrap() < &1.0);
        assert!(xs.windows(2).all(|w| w[0] <= w[1]));

        // Mean is 1/2
        let mean = xs.iter().sum::<f64>() / n as f64;
        close(mean, 0.5, 1e-2);

        // Variance is 1/12
        let var = xs.iter().map(|x| (x - 0.5).powi(2)).sum::<f64>() / n as f64;
        close(var, 1.0 / 12.0, 1e-2);
    }

    #[test]
    fn test_multi_invccdf_sorted() {
        let sbd = Sbd::new(UnitPowerLaw::new(10.0).unwrap());
        let ps = sorted_uniforms(5, &mut thread_rng());
        assert_eq!(
            sbd.multi_invccdf_sorted(&ps),
            ps.iter().rev().map(|p| sbd.invccdf(*p)).collect::<Vec<_>>()
        )
    }
}

// impl ConjugatePrior<usize, Sbd> for UnitPowerLaw {
//     type Posterior = UnitPowerLaw;
//     type LnMCache = StickSequence;
//     type LnPpCache = StickSequence;

//     // fn posterior(&self, x: &DataOrSuffStat<usize, Sbd>) -> Self::Posterior {
//     //     let mut sticks = self.sticks.clone();
//     //     x.suff_stat().counts.iter().for_each(|&n| {
//     //         sticks.observe(n);
//     //     });
//     //     Self { sticks }
//     // }

//     // fn ln_m_cache(&self) -> Self::LnMCache {
//     //     self.sticks.clone()
//     // }

//     // fn ln_m_with_cache(&self, cache: &Self::LnMCache, x: &DataOrSuffStat<usize, Sbd>) -> f64 {
//     //     let mut result = 0.0;
//     //     for (n, &count) in x.data_or_suff_stat().counts.iter().enumerate() {
//     //         result += cache.ccdf(n + 1) * count as f64;
//     //     }
//     //     result
//     // }

//     // fn ln_pp_cache(&self, x: &DataOrSuffStat<usize, Sbd>) -> Self::LnPpCache {
//     //     self.sticks.clone()
//     // }

//     // fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &usize) -> f64 {
//     //     cache.ccdf(y + 1)
//     // }

//     // /// The log marginal likelihood
//     // fn ln_m(&self, x: &DataOrSuffStat<usize, Sbd>) -> f64 {
//     //     let cache = self.ln_m_cache();
//     //     self.ln_m_with_cache(&cache, x)
//     // }

//     // /// Log posterior predictive of y given x
//     // fn ln_pp(&self, y: &usize, x: &DataOrSuffStat<usize, Sbd>) -> f64 {
//     //     let cache = self.ln_pp_cache(x);
//     //     self.ln_pp_with_cache(&cache, y)
//     // }

//     // /// Marginal likelihood of x
//     // fn m(&self, x: &DataOrSuffStat<usize, Sbd>) -> f64 {
//     //     self.ln_m(x).exp()
//     // }

//     // /// Posterior Predictive distribution
//     // fn pp(&self, y: &usize, x: &DataOrSuffStat<usize, Sbd>) -> f64 {
//     //     self.ln_pp(y, x).exp()
//     // }
// }
