use crate::data::{CategoricalSuffStat, DataOrSuffStat};
use crate::dist::{Beta, Dirichlet};
use crate::prelude::{Categorical, SymmetricDirichlet};
use crate::traits::{ConjugatePrior, Rv, SuffStat};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::sbd::Sbd;
use super::sbd_stat::SbdSuffStat;

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct Sb {
    alpha: f64,
    k: usize,
    seed: Option<u64>,
}

impl Sb {
    pub fn new(alpha: f64, k: usize, seed: Option<u64>) -> Self {
        Self { alpha, k, seed }
    }

    pub fn set_alpha_unchecked(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug)]
pub struct SbPosterior {
    alpha: f64,
    lookup: HashMap<usize, usize>,
    dir: Dirichlet,
}

impl Rv<Sbd> for Sb {
    fn ln_f(&self, x: &Sbd) -> f64 {
        // let k = x.k() + 1;
        // let symdir = SymmetricDirichlet::new_unchecked(self.alpha, k);
        // x.inner
        //     .read()
        //     .map(|obj| symdir.ln_f(&obj.ln_weights))
        //     .unwrap()
        //
        let k = x.k();
        if k == 0 {
            return 1.0;
        }

        let beta = Beta::new_unchecked(1.0, self.alpha);
        x.inner
            .read()
            .map(|inner| {
                inner.ln_weights.iter().take(k).map(|&lnw| lnw.exp()).fold(
                    (1.0, 0.0),
                    |(rm_mass, ln_f), w| {
                        let ln_f_b = beta.ln_f(&(w / rm_mass));
                        // dbg!(&ln_f_b, &w, &rm_mass);
                        (rm_mass - w, ln_f + ln_f_b)
                    },
                )
            })
            .unwrap()
            .1
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Sbd {
        let mut remaining_mass = 1.0;
        let mut weights: Vec<f64> = Vec::new();
        let beta = Beta::new_unchecked(1.0, self.alpha);
        (0..self.k).for_each(|_| {
            let p: f64 = beta.draw(rng);
            let w = remaining_mass * p;
            remaining_mass -= w;
            weights.push(w);
        });
        weights.push(remaining_mass);

        Sbd::from_canonical_weights(&weights, self.alpha, self.seed).unwrap()
    }
}

impl Rv<Sbd> for SbPosterior {
    fn ln_f(&self, _x: &Sbd) -> f64 {
        unimplemented!()
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Sbd {
        let weights: Vec<f64> = self.dir.draw(rng);
        Sbd::from_weights_and_lookup(
            &weights,
            self.lookup.clone(),
            self.alpha,
            Some(rng.gen()),
        )
        .unwrap()
    }
}

fn sbpost_from_stat(alpha: f64, stat: &SbdSuffStat) -> SbPosterior {
    let lookup: HashMap<usize, usize> = stat
        .counts()
        .keys()
        .enumerate()
        .map(|(ix, x)| (*x, ix))
        .collect();

    let dir = {
        let alphas: Vec<f64> = stat
            .counts()
            .iter()
            .map(|(_, &ct)| ct as f64 + alpha)
            .chain(std::iter::once(alpha))
            .collect();

        Dirichlet::new(alphas).unwrap()
    };

    assert_eq!(lookup.len() + 1, dir.alphas().len());

    SbPosterior { alpha, lookup, dir }
}

fn sbm_from_stat(alpha: f64, stat: &SbdSuffStat) -> f64 {
    if stat.n() == 0 {
        panic!("SBD Stat is empty {:?}", stat);
    }

    let stat = CategoricalSuffStat::from_parts_unchecked(
        stat.n(),
        stat.counts()
            .values()
            .map(|&ct| ct as f64)
            .chain(std::iter::once(alpha))
            .collect(),
    );

    let symdir = SymmetricDirichlet::new(alpha, stat.counts().len()).unwrap();
    symdir.ln_m(&DataOrSuffStat::SuffStat::<usize, Categorical>(&stat))
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct SbCache {
    ln_weights: HashMap<usize, f64>,
    ln_f_new: f64,
}

impl ConjugatePrior<usize, Sbd> for Sb {
    type Posterior = SbPosterior;
    type LnMCache = ();
    type LnPpCache = SbCache;

    fn ln_m_cache(&self) -> Self::LnMCache {}

    fn ln_pp_cache(&self, x: &DataOrSuffStat<usize, Sbd>) -> Self::LnPpCache {
        let post = self.posterior(x);
        // we'll need the alpha for computing 1 / (1 + alpha), which is the
        // expected liklelihood if a new class
        let alpha = post.dir.alphas().last().unwrap();
        // Need to norm the alphas to probabilities
        let ln_norm = post.dir.alphas().iter().sum::<f64>().ln();
        let ln_weights = post
            .lookup
            .iter()
            .map(|(&x, &ix)| (x, post.dir.alphas[ix].ln() - ln_norm))
            .collect();

        // ln (1/(1 + alpha))
        let ln_f_new = (1.0 + alpha).recip().ln() - ln_norm;

        SbCache {
            ln_weights,
            ln_f_new,
        }
    }

    fn posterior(&self, x: &DataOrSuffStat<usize, Sbd>) -> Self::Posterior {
        match x {
            DataOrSuffStat::Data(xs) => {
                let mut stat = SbdSuffStat::new();
                stat.observe_many(xs);
                sbpost_from_stat(self.alpha, &stat)
            }
            DataOrSuffStat::SuffStat(stat) => {
                sbpost_from_stat(self.alpha, stat)
            }
            DataOrSuffStat::None => panic!("Need data for posterior"),
        }
    }

    fn ln_m_with_cache(
        &self,
        _cache: &Self::LnMCache,
        x: &DataOrSuffStat<usize, Sbd>,
    ) -> f64 {
        match x {
            DataOrSuffStat::Data(xs) => {
                let mut stat = SbdSuffStat::new();
                stat.observe_many(xs);
                sbm_from_stat(self.alpha, &stat)
            }
            DataOrSuffStat::SuffStat(stat) => sbm_from_stat(self.alpha, stat),
            DataOrSuffStat::None => panic!("Need data for posterior"),
        }
    }

    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &usize) -> f64 {
        // FIXME: I feel like this isn't quite right
        cache.ln_weights.get(y).copied().unwrap_or(cache.ln_f_new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rv_sbd_smoke() {
        let alpha = 0.5;
        let prior = Sb::new(alpha, 1, None);
        let sbd = Sbd::new(alpha, None).unwrap();
        // populate sbd
        (0..1).for_each(|x| {
            sbd.ln_f(&x);
        });

        let ln_f = prior.ln_f(&sbd);
        eprintln!("{ln_f}");
    }

    #[test]
    fn sbd_posterior_smoke() {
        let alpha = 0.5;
        let prior = Sb::new(alpha, 1, None);
        let mut stat = SbdSuffStat::new();
        // populate sbd
        (0..10_usize)
            .cycle()
            .take(100)
            .for_each(|x| stat.observe(&x));

        let mut rng = rand::thread_rng();
        let post = prior.posterior(&DataOrSuffStat::SuffStat(&stat));
        let sbd = post.draw(&mut rng);

        eprintln!("POST: {:?}", post);
        eprintln!("SBD: {:?}", sbd);
    }

    #[test]
    fn breaking_ex1() {
        // This example used to result in NaN weight
        let ln_weights = [
            (-0.627_512_722_454_201_5_f64).exp(),
            (-0.763_394_023_914_272_2_f64).exp(),
        ];
        let alpha = 1.0;
        let prior = Sb::new(alpha, 1, None);
        let sbd =
            Sbd::from_canonical_weights(&ln_weights, alpha, None).unwrap();
        let ln_f = prior.ln_f(&sbd);
        eprintln!("{:?}", ln_weights);
        eprintln!("{ln_f}");
    }

    #[test]
    fn posterior_sanity() {
        let data: Vec<usize> = vec![0, 3, 3, 3, 3, 3, 4, 4, 4, 6];

        let mut stat = SbdSuffStat::new();
        stat.observe_many(&data);

        let prior = Sb::new(0.1, 1, None);

        let mut rng = rand::thread_rng();
        let post = prior.posterior(&DataOrSuffStat::SuffStat(&stat));

        let mut sums = vec![0.0; 4];

        for _ in 0..100 {
            let sbd = post.draw(&mut rng);
            let f0 = sbd.f(&0_usize);
            let f3 = sbd.f(&3_usize);
            let f4 = sbd.f(&4_usize);
            let f6 = sbd.f(&6_usize);

            sums[0] += f0;
            sums[1] += f3;
            sums[2] += f4;
            sums[3] += f6;
        }
        assert!(sums[1] > sums[2]);
        assert!(sums[2] > sums[0]);
        assert!(sums[2] > sums[3]);
    }

    #[test]
    fn posterior_predictive() {
        let data: Vec<usize> = vec![0, 3, 3, 3, 3, 3, 4, 4, 4, 6];

        let mut st = SbdSuffStat::new();
        st.observe_many(&data);

        let stat = DataOrSuffStat::SuffStat(&st);

        let prior = Sb::new(0.1, 1, None);

        let ln_pp_0 =
            <Sb as ConjugatePrior<usize, Sbd>>::ln_pp(&prior, &0, &stat);
        let ln_pp_1 =
            <Sb as ConjugatePrior<usize, Sbd>>::ln_pp(&prior, &1, &stat);
        let ln_pp_2 =
            <Sb as ConjugatePrior<usize, Sbd>>::ln_pp(&prior, &2, &stat);
        let ln_pp_3 =
            <Sb as ConjugatePrior<usize, Sbd>>::ln_pp(&prior, &3, &stat);
        let ln_pp_4 =
            <Sb as ConjugatePrior<usize, Sbd>>::ln_pp(&prior, &4, &stat);
        let ln_pp_5 =
            <Sb as ConjugatePrior<usize, Sbd>>::ln_pp(&prior, &5, &stat);
        let ln_pp_6 =
            <Sb as ConjugatePrior<usize, Sbd>>::ln_pp(&prior, &6, &stat);
        let ln_pp_7 =
            <Sb as ConjugatePrior<usize, Sbd>>::ln_pp(&prior, &7, &stat);

        assert!(ln_pp_0 < ln_pp_4);
        assert::close(ln_pp_0, ln_pp_6, 1e-10);
        assert!(ln_pp_4 < ln_pp_3);
        assert!(ln_pp_1 < ln_pp_0);
        assert::close(ln_pp_1, ln_pp_2, 1e-10);
        assert::close(ln_pp_1, ln_pp_5, 1e-10);
        assert::close(ln_pp_1, ln_pp_7, 1e-10);

        let sum = ln_pp_0.exp() + ln_pp_3.exp() + ln_pp_4.exp() + ln_pp_6.exp();
        eprintln!("{sum}");
    }
}
