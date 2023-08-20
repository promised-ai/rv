use crate::data::DataOrSuffStat;
use crate::dist::{Beta, Dirichlet};
use crate::traits::{ConjugatePrior, Rv, SuffStat};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::sbd::Sbd;
use super::sbd_stat::SbdSuffStat;

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug)]
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
        let beta = Beta::new_unchecked(1.0, self.alpha);
        x.inner
            .read()
            .map(|inner| {
                inner.ln_weights.iter().map(|&lnw| lnw.exp()).fold(
                    (1.0, 0.0),
                    |(rm_mass, ln_f), w| {
                        (rm_mass - w, ln_f + beta.ln_f(&(w / rm_mass)))
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
            .collect();

        Dirichlet::new(alphas).unwrap()
    };

    SbPosterior { alpha, lookup, dir }
}

impl ConjugatePrior<usize, Sbd> for Sb {
    type LnMCache = ();
    type LnPpCache = ();
    type Posterior = SbPosterior;

    fn ln_m_cache(&self) -> Self::LnMCache {}

    fn ln_pp_cache(&self, _x: &DataOrSuffStat<usize, Sbd>) -> Self::LnPpCache {}

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
        _x: &DataOrSuffStat<usize, Sbd>,
    ) -> f64 {
        unimplemented!()
    }

    fn ln_pp_with_cache(&self, _cache: &Self::LnPpCache, _y: &usize) -> f64 {
        unimplemented!()
    }
}
