use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro128Plus;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

// use super::sb_stat::StickBreakingSuffStat;
use crate::dist::UnitPowerLaw;
use crate::experimental::stick_breaking_stat::StickBreakingSuffStat;
use crate::experimental::stick_sequence::StickSequence;
use crate::process_traits::Process;
use crate::suffstat_traits::*;
use crate::traits::Rv;

#[derive(Clone, Debug)]
pub enum StickBreakingError {
    InvalidAlpha(f64),
}

impl std::error::Error for StickBreakingError {}

impl std::fmt::Display for StickBreakingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAlpha(alpha) => {
                write!(
                    f,
                    "alpha ({}) must be finite and greater than zero",
                    alpha
                )
            }
        }
    }
}

// NOTE: We currently derive PartialEq, but this (we think) compares the
// internal state of the RNGs, which is probably not what we want.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct _Inner {
    remaining_mass: f64,
    // the bin weights. the last entry is ln(remaining_mass)
    pub ln_weights: Vec<f64>,
    rng: rand_xoshiro::Xoshiro128Plus,
}

impl _Inner {
    // pub fn next_category(&self) -> usize {
    //     self.ln_weights.len() - 1
    // }

    pub fn num_cats(&self) -> usize {
        self.ln_weights.len() - 1
    }

    fn extend(&mut self, powlaw: &UnitPowerLaw) -> f64 {
        let b: f64 = {
            let p: f64 = powlaw.draw(&mut self.rng);
            1.0 - p
        };
        let rm_mass = self.remaining_mass;
        let w = rm_mass * b;
        let rm_mass = rm_mass - w;

        let ln_w = w.ln();

        self.remaining_mass = rm_mass;
        if let Some(last) = self.ln_weights.last_mut() {
            *last = ln_w;
        } else {
            panic!("empty ln_weights");
        }
        self.ln_weights.push(rm_mass.ln());

        ln_w
    }

    fn extend_until<F>(&mut self, powlaw: &UnitPowerLaw, p: F) -> &Vec<f64>
    where
        F: Fn(&_Inner) -> bool,
    {
        while !p(self) {
            self.extend(powlaw);
        }
        &self.ln_weights
    }
}

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
// #[cfg_attr(
//     feature = "serde1",
//     serde(
//         rename_all = "snake_case",
//         from = "StickBreakingFmt",
//         into = "StickBreakingFmt"
//     )
// )]
#[derive(Clone, Debug, PartialEq)]
pub struct StickBreaking {
    pub alpha: f64,
}

impl StickBreaking {
    pub fn new(alpha: f64) -> Result<Self, StickBreakingError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(StickBreakingError::InvalidAlpha(alpha));
        }
        Ok(Self { alpha })
    }

    // pub fn from_ln_weights(
    //     ln_weights: Vec<f64>,
    //     alpha: f64,
    //     seed: Option<u64>,
    // ) -> Result<Self, StickBreakingError> {
    //     if alpha <= 0.0 || !alpha.is_finite() {
    //         Err(StickBreakingError::InvalidAlpha(alpha))
    //     } else {
    //         let powlaw = UnitPowerLaw::new_unchecked(alpha);

    //         let inner = _Inner {
    //             remaining_mass: ln_weights.last().unwrap().exp(),
    //             ln_weights,
    //             rng: seed.map_or_else(
    //                 Xoshiro128Plus::from_entropy,
    //                 Xoshiro128Plus::seed_from_u64,
    //             ),
    //         };

    //         Ok(Self {
    //             powlaw,
    //             inner: Arc::new(RwLock::new(inner)),
    //         })
    //     }
    // }

    // pub fn from_weights(
    //     weights: &[f64],
    //     alpha: f64,
    //     seed: Option<u64>,
    // ) -> Result<Self, StickBreakingError> {
    //     let sum = weights.iter().sum::<f64>();
    //     if (sum - 1.0).abs() > 1e-13 {
    //         return Err(StickBreakingError::WeightsDoNotSumToOne { sum });
    //     }
    //     let ln_weights = weights.iter().map(|w| w.ln()).collect();
    //     Self::from_ln_weights(ln_weights, alpha, seed)
    // }

    // pub fn from_canonical_weights(
    //     weights: &[f64],
    //     alpha: f64,
    //     seed: Option<u64>,
    // ) -> Result<Self, StickBreakingError> {
    //     if alpha <= 0.0 || !alpha.is_finite() {
    //         Err(StickBreakingError::InvalidAlpha(alpha))
    //     } else {
    //         let k = weights.len() - 1;
    //         let inner = _Inner {
    //             remaining_mass: weights[k],
    //             ln_weights: weights.iter().map(|&w| w.ln()).collect(),
    //             rng: seed.map_or_else(
    //                 Xoshiro128Plus::from_entropy,
    //                 Xoshiro128Plus::seed_from_u64,
    //             ),
    //         };

    //         assert_eq!(inner.ln_weights.len(), k + 1);

    //         Ok(Self {
    //             powlaw: UnitPowerLaw::new_unchecked(alpha),
    //             inner: Arc::new(RwLock::new(inner)),
    //         })
    //     }
    // }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl Process<StickSequence, &[f64]> for StickBreaking {
    fn ln_f(&self, x: &&[f64]) -> f64 {
        let stat = StickBreakingSuffStat::from(x);
        self.ln_f_stat(&stat)
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> StickSequence {
        let seed: u64 = rng.gen();

        StickSequence::new(self.alpha(), Some(seed)).unwrap()
    }
}

// #[cfg(test)]
// mod test {
//     use rand::SeedableRng;
//     use std::collections::HashMap;

//     use super::*;

//     #[test]
//     fn canonical_order_ln_f() {
//         let sb = StickBreaking::new(1.0, None).unwrap();
//         let mut rm_mass = sb.p_unobserved();
//         for x in 0..10 {
//             let ln_f_1 = sb.ln_f(&x);
//             let k = sb.num_cats();
//             assert!(rm_mass > sb.p_unobserved());
//             rm_mass = sb.p_unobserved();

//             let ln_f_2 = sb.ln_f(&x);

//             assert_eq!(ln_f_1, ln_f_2);
//             assert_eq!(k, sb.num_cats());
//         }
//     }

//     #[test]
//     fn static_ln_f_from_new() {
//         let sb = StickBreaking::new(1.0, None).unwrap();

//         assert_eq!(sb.num_cats(), 0);

//         let lnf0 = sb.ln_f(&0_usize);
//         assert::close(lnf0, sb.ln_f(&0_usize), 1e-12);

//         assert_eq!(sb.num_cats(), 1);

//         let _lnf1 = sb.ln_f(&1_usize); // causes new category to form
//         assert::close(lnf0, sb.ln_f(&0_usize), 1e-12);
//         assert_eq!(sb.num_cats(), 2);
//     }

//     #[test]
//     fn draw_many_smoke() {
//         let mut counter: HashMap<usize, usize> = HashMap::new();
//         let mut rng = rand::thread_rng();
//         let seed: u64 = rng.gen();
//         eprintln!("draw_many_smoke seed: {seed}");
//         let mut rng = rand_xoshiro::Xoroshiro128Plus::seed_from_u64(seed);
//         let sb = StickBreaking::new(1.0, None).unwrap();
//         for _ in 0..1_000 {
//             let x: usize = sb.draw(&mut rng);
//             counter.entry(x).and_modify(|ct| *ct += 1).or_insert(1);
//         }
//         // eprintln!("{:?}", counter);
//     }

//     #[test]
//     fn repeatedly_compute_oob_lnf() {
//         let sb = StickBreaking::new(0.5, None).unwrap();
//         assert_eq!(sb.num_cats(), 0);

//         sb.ln_f(&0);
//         assert_eq!(sb.num_cats(), 1);

//         sb.ln_f(&1);
//         assert_eq!(sb.num_cats(), 2);

//         sb.ln_f(&1);
//         sb.ln_f(&1);
//         assert_eq!(sb.num_cats(), 2);

//         sb.ln_f(&0);
//         assert_eq!(sb.num_cats(), 2);
//     }
// }
