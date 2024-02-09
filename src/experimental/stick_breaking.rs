use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro128Plus;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

// use super::sb_stat::StickBreakingSuffStat;
use crate::dist::UnitPowerLaw;
use crate::experimental::stick_breaking_stat::StickBreakingSuffStat;
use crate::misc::argmax;
use crate::misc::ln_pflip;
use crate::prelude::Mode;
use crate::suffstat_traits::*;
use crate::traits::Rv;

#[derive(Clone, Debug)]
pub enum StickBreakingError {
    InvalidAlpha(f64),
    WeightsDoNotSumToOne { sum: f64 },
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
            Self::WeightsDoNotSumToOne { sum } => {
                write!(f, "Weights do not sum to 1 ({sum})")
            }
        }
    }
}

// We'd like to be able to serialize and deserialize StickBreaking, but serde can't handle
// `Arc` or `RwLock`. So we use `StickBreakingFmt` as an intermediate type.
#[cfg(feature = "serde1")]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
struct StickBreakingFmt {
    powlaw: UnitPowerLaw,
    inner: _Inner,
}

#[cfg(feature = "serde1")]
impl From<StickBreakingFmt> for StickBreaking {
    fn from(fmt: StickBreakingFmt) -> Self {
        Self {
            powlaw: fmt.powlaw,
            inner: Arc::new(RwLock::new(fmt.inner)),
        }
    }
}

#[cfg(feature = "serde1")]
impl From<StickBreaking> for StickBreakingFmt {
    fn from(sb: StickBreaking) -> Self {
        Self {
            powlaw: sb.powlaw,
            inner: sb.inner.read().map(|inner| inner.clone()).unwrap(),
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
    pub fn next_category(&self) -> usize {
        self.ln_weights.len() - 1
    }

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
#[cfg_attr(
    feature = "serde1",
    serde(
        rename_all = "snake_case",
        from = "StickBreakingFmt",
        into = "StickBreakingFmt"
    )
)]
#[derive(Clone, Debug)]
pub struct StickBreaking {
    pub powlaw: UnitPowerLaw,
    pub inner: Arc<RwLock<_Inner>>,
}

impl PartialEq<StickBreaking> for StickBreaking {
    fn eq(&self, other: &StickBreaking) -> bool {
        self.powlaw == other.powlaw
            && self.with_inner(|inner| {
                other.with_inner(|other_inner| *inner == *other_inner)
            })
    }
}

impl StickBreaking {
    pub fn new(
        alpha: f64,
        seed: Option<u64>,
    ) -> Result<Self, StickBreakingError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(StickBreakingError::InvalidAlpha(alpha));
        }
        let powlaw = UnitPowerLaw::new_unchecked(alpha);
        let inner = _Inner {
            remaining_mass: 1.0,
            ln_weights: vec![0.0], // ln(1)
            rng: seed.map_or_else(
                Xoshiro128Plus::from_entropy,
                Xoshiro128Plus::seed_from_u64,
            ),
        };
        Ok(Self {
            powlaw,
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    pub fn with_inner<F, Ans>(&self, f: F) -> Ans
    where
        F: FnOnce(&_Inner) -> Ans,
    {
        self.inner.read().map(|inner| f(&inner)).unwrap()
    }

    pub fn with_inner_mut<F, Ans>(&self, f: F) -> Ans
    where
        F: FnOnce(&mut _Inner) -> Ans,
    {
        self.inner.write().map(|mut inner| f(&mut inner)).unwrap()
    }

    pub fn from_ln_weights(
        ln_weights: Vec<f64>,
        alpha: f64,
        seed: Option<u64>,
    ) -> Result<Self, StickBreakingError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(StickBreakingError::InvalidAlpha(alpha));
        }
        let powlaw = UnitPowerLaw::new_unchecked(alpha);

        let inner = _Inner {
            remaining_mass: ln_weights.last().unwrap().exp(),
            ln_weights,
            rng: seed.map_or_else(
                Xoshiro128Plus::from_entropy,
                Xoshiro128Plus::seed_from_u64,
            ),
        };

        Ok(Self {
            powlaw,
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    pub fn from_weights(
        weights: &[f64],
        alpha: f64,
        seed: Option<u64>,
    ) -> Result<Self, StickBreakingError> {
        let sum = weights.iter().sum::<f64>();
        if (sum - 1.0).abs() > 1e-13 {
            return Err(StickBreakingError::WeightsDoNotSumToOne { sum });
        }
        let ln_weights = weights.iter().map(|w| w.ln()).collect();
        Self::from_ln_weights(ln_weights, alpha, seed)
    }

    pub fn from_canonical_weights(
        weights: &[f64],
        alpha: f64,
        seed: Option<u64>,
    ) -> Result<Self, StickBreakingError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            Err(StickBreakingError::InvalidAlpha(alpha))
        } else {
            let k = weights.len() - 1;
            let inner = _Inner {
                remaining_mass: weights[k],
                ln_weights: weights.iter().map(|&w| w.ln()).collect(),
                rng: seed.map_or_else(
                    Xoshiro128Plus::from_entropy,
                    Xoshiro128Plus::seed_from_u64,
                ),
            };

            assert_eq!(inner.ln_weights.len(), k + 1);

            Ok(Self {
                powlaw: UnitPowerLaw::new_unchecked(alpha),
                inner: Arc::new(RwLock::new(inner)),
            })
        }
    }

    pub fn num_cats(&self) -> usize {
        self.with_inner(|inner| inner.num_cats())
    }

    pub fn p_unobserved(&self) -> f64 {
        self.with_inner(|inner| inner.remaining_mass)
    }

    pub fn alpha(&self) -> f64 {
        self.powlaw.alpha()
    }
}

impl HasSuffStat<&[f64]> for StickBreaking {
    type Stat = StickBreakingSuffStat;

    fn empty_suffstat(&self) -> Self::Stat {
        StickBreakingSuffStat::new()
    }

    fn ln_f_stat(&self, _stat: &Self::Stat) -> f64 {
        unimplemented!()
    }
}

impl Rv<usize> for StickBreaking {
    fn ln_f(&self, x: &usize) -> f64 {
        if self.with_inner(|inner| inner.num_cats() > *x) {
            self.with_inner(|inner| inner.ln_weights[*x])
        } else {
            self.with_inner_mut(|inner| {
                inner.extend_until(&self.powlaw, move |inner| {
                    inner.num_cats() > *x
                })[*x]
            })
        }
    }

    /// Alternate option:
    // fn ln_f(&self, x: &usize) -> f64 {
    //     self.with_inner(|inner| {
    //         if inner.num_cats() > *x {
    //             inner.ln_weights[*x]
    //         } else {
    //             self.with_inner_mut(|inner| {
    //                 inner.extend_until(&self.powlaw, move |inner| {
    //                     inner.num_cats() > *x
    //                 })[*x]
    //             })
    //         }
    //     })
    // }

    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        let u: f64 = rng.gen();

        let powlaw = self.powlaw.clone();
        self.with_inner_mut(|inner| {
            let remaining_mass = inner.remaining_mass;
            let k = inner.num_cats();

            if u < 1.0 - remaining_mass {
                // TODO: Since we know the remaining mass, we can easily
                // normalize without needing logsumexp
                ln_pflip(&inner.ln_weights[..k], 1, false, rng)[0]
            } else {
                let ln_ws = inner.extend_until(&powlaw, |inner| {
                    inner.remaining_mass <= 1.0 - u
                });
                let ix = ln_pflip(ln_ws, 1, false, rng)[0];
                ix + k
            }
        })
    }
}

impl Mode<usize> for StickBreaking {
    fn mode(&self) -> Option<usize> {
        let i_max = self.with_inner_mut(|inner| {
            // TODO: Make this more efficient
            inner.extend_until(&self.powlaw, |inner| {
                argmax(&inner.ln_weights)[0] < inner.ln_weights.len() - 1
            });
            argmax(&inner.ln_weights)[0]
        });

        Some(i_max)
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn canonical_order_ln_f() {
        let sb = StickBreaking::new(1.0, None).unwrap();
        let mut rm_mass = sb.p_unobserved();
        for x in 0..10 {
            let ln_f_1 = sb.ln_f(&x);
            let k = sb.num_cats();
            assert!(rm_mass > sb.p_unobserved());
            rm_mass = sb.p_unobserved();

            let ln_f_2 = sb.ln_f(&x);

            assert_eq!(ln_f_1, ln_f_2);
            assert_eq!(k, sb.num_cats());
        }
    }

    #[test]
    fn static_ln_f_from_new() {
        let sb = StickBreaking::new(1.0, None).unwrap();

        assert_eq!(sb.num_cats(), 0);

        let lnf0 = sb.ln_f(&0_usize);
        assert::close(lnf0, sb.ln_f(&0_usize), 1e-12);

        assert_eq!(sb.num_cats(), 1);

        let _lnf1 = sb.ln_f(&1_usize); // causes new category to form
        assert::close(lnf0, sb.ln_f(&0_usize), 1e-12);
        assert_eq!(sb.num_cats(), 2);
    }

    #[test]
    fn draw_many_smoke() {
        let mut counter: HashMap<usize, usize> = HashMap::new();
        let mut rng = rand::thread_rng();
        let seed: u64 = rng.gen();
        eprintln!("draw_many_smoke seed: {seed}");
        let mut rng = rand_xoshiro::Xoroshiro128Plus::seed_from_u64(seed);
        let sb = StickBreaking::new(1.0, None).unwrap();
        for _ in 0..1_000 {
            let x: usize = sb.draw(&mut rng);
            counter.entry(x).and_modify(|ct| *ct += 1).or_insert(1);
        }
        // eprintln!("{:?}", counter);
    }

    #[test]
    fn repeatedly_compute_oob_lnf() {
        let sb = StickBreaking::new(0.5, None).unwrap();
        assert_eq!(sb.num_cats(), 0);

        sb.ln_f(&0);
        assert_eq!(sb.num_cats(), 1);

        sb.ln_f(&1);
        assert_eq!(sb.num_cats(), 2);

        sb.ln_f(&1);
        sb.ln_f(&1);
        assert_eq!(sb.num_cats(), 2);

        sb.ln_f(&0);
        assert_eq!(sb.num_cats(), 2);
    }
}
