use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro128Plus;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use super::StickSequence;
use crate::dist::Beta;
use crate::misc::argmax;
use crate::misc::ln_pflip;
use crate::prelude::Mode;
use crate::suffstat_traits::HasSuffStat;
use crate::traits::Rv;

#[derive(Clone, Debug)]
pub enum SbdError {
    InvalidAlpha(f64),
    InvalidNumberOfWeights { n_weights: usize, n_entries: usize },
    WeightsDoNotSumToOne { sum: f64 },
}

impl std::error::Error for SbdError {}

impl std::fmt::Display for SbdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAlpha(alpha) => {
                write!(
                    f,
                    "alpha ({}) must be finite and greater than zero",
                    alpha
                )
            }
            Self::InvalidNumberOfWeights {
                n_weights,
                n_entries,
            } => {
                write!(
                    f,
                    "There should be one more weight than lookup entries. \
                    Given {n_weights}, but there are {n_entries} lookup entries",
                )
            }
            Self::WeightsDoNotSumToOne { sum } => {
                write!(f, "Weights do not sum to 1 ({sum})")
            }
        }
    }
}

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Sbd {
    pub sticks: StickSequence,
}

impl Sbd {
    // Is this how we want to set this up?
    pub fn new(sticks: StickSequence) -> Self {
        Self { sticks }
    }
}

// impl HasSuffStat<usize> for Sbd {
//     type Stat = SbdSuffStat;

//     fn empty_suffstat(&self) -> Self::Stat {
//         SbdSuffStat::new()
//     }

//     fn ln_f_stat(&self, _stat: &Self::Stat) -> f64 {
//         unimplemented!()
//     }
// }

impl Rv<usize> for Sbd {
    fn f(&self, n: &usize) -> f64 {
        let sticks = &self.sticks;
        sticks.weight(*n)
    }

    fn ln_f(&self, n: &usize) -> f64 {
        self.f(n).ln()
    }

    /// Alternate option:
    // fn ln_f(&self, x: &usize) -> f64 {
    //     self.with_inner(|inner| {
    //         if inner.num_cats() > *x {
    //             inner.ln_weights[*x]
    //         } else {
    //             self.with_inner_mut(|inner| {
    //                 inner.extend_until(&self.beta, move |inner| {
    //                     inner.num_cats() > *x
    //                 })[*x]
    //             })
    //         }
    //     })
    // }

    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        let u: f64 = rng.gen();
        self.sticks.extendmap_ccdf(
            |ccdf| ccdf.last().unwrap() < &u,
            |ccdf| ccdf.iter().position(|q| *q < u).unwrap() - 1,
        )
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn canonical_order_ln_f() {
        let sbd = Sbd::new(1.0, None).unwrap();
        let mut rm_mass = sbd.p_unobserved();
        for x in 0..10 {
            let ln_f_1 = sbd.ln_f(&x);
            let k = sbd.num_cats();
            assert!(rm_mass > sbd.p_unobserved());
            rm_mass = sbd.p_unobserved();

            let ln_f_2 = sbd.ln_f(&x);

            assert_eq!(ln_f_1, ln_f_2);
            assert_eq!(k, sbd.num_cats());
        }
    }

    #[test]
    fn static_ln_f_from_new() {
        let sbd = Sbd::new(1.0, None).unwrap();

        assert_eq!(sbd.num_cats(), 0);

        let lnf0 = sbd.ln_f(&0_usize);
        assert::close(lnf0, sbd.ln_f(&0_usize), 1e-12);

        assert_eq!(sbd.num_cats(), 1);

        let _lnf1 = sbd.ln_f(&1_usize); // causes new category to form
        assert::close(lnf0, sbd.ln_f(&0_usize), 1e-12);
        assert_eq!(sbd.num_cats(), 2);
    }

    #[test]
    fn draw_many_smoke() {
        let mut counter: HashMap<usize, usize> = HashMap::new();
        let mut rng = rand::thread_rng();
        let seed: u64 = rng.gen();
        eprintln!("draw_many_smoke seed: {seed}");
        let mut rng = rand_xoshiro::Xoroshiro128Plus::seed_from_u64(seed);
        let sbd = Sbd::new(1.0, None).unwrap();
        for _ in 0..1_000 {
            let x: usize = sbd.draw(&mut rng);
            counter.entry(x).and_modify(|ct| *ct += 1).or_insert(1);
        }
        // eprintln!("{:?}", counter);
    }

    #[test]
    fn repeatedly_compute_oob_lnf() {
        let sbd = Sbd::new(0.5, None).unwrap();
        assert_eq!(sbd.num_cats(), 0);

        sbd.ln_f(&0);
        assert_eq!(sbd.num_cats(), 1);

        sbd.ln_f(&1);
        assert_eq!(sbd.num_cats(), 2);

        sbd.ln_f(&1);
        sbd.ln_f(&1);
        assert_eq!(sbd.num_cats(), 2);

        sbd.ln_f(&0);
        assert_eq!(sbd.num_cats(), 2);
    }
}
