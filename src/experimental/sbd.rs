use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro128Plus;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use super::sbd_stat::SbdSuffStat;
use crate::dist::Beta;
use crate::misc::argmax;
use crate::misc::ln_pflip;
use crate::prelude::Mode;
use crate::traits::{HasSuffStat, Rv};

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

// We'd like to be able to serialize and deserialize Sbd, but serde can't handle
// `Arc` or `RwLock`. So we use `SbdFmt` as an intermediate type.
#[cfg(feature = "serde1")]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
struct SbdFmt {
    beta: Beta,
    inner: _Inner,
}

#[cfg(feature = "serde1")]
impl From<SbdFmt> for Sbd {
    fn from(fmt: SbdFmt) -> Self {
        Self {
            beta: fmt.beta,
            inner: Arc::new(RwLock::new(fmt.inner)),
        }
    }
}

#[cfg(feature = "serde1")]
impl From<Sbd> for SbdFmt {
    fn from(sbd: Sbd) -> Self {
        Self {
            beta: sbd.beta,
            inner: sbd.inner.read().map(|inner| inner.clone()).unwrap(),
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

    fn extend(&mut self, beta: &Beta) -> f64 {
        let b: f64 = beta.draw(&mut self.rng);
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

    fn extend_until<F>(&mut self, beta: &Beta, p: F) -> &Vec<f64>
    where
        F: Fn(& _Inner) -> bool,
    {
        while !p(self) {
            self.extend(beta);
        }
        &self.ln_weights        
    }
}

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde1",
    serde(rename_all = "snake_case", from = "SbdFmt", into = "SbdFmt")
)]
#[derive(Clone, Debug)]
pub struct Sbd {
    pub beta: Beta,
    pub inner: Arc<RwLock<_Inner>>,
}

impl PartialEq<Sbd> for Sbd {
    fn eq(&self, other: &Sbd) -> bool {
        self.beta == other.beta
            && self.with_inner(|inner| {
                other.with_inner(|other_inner| *inner == *other_inner)
            })
    }
}

impl Sbd {
    pub fn new(alpha: f64, seed: Option<u64>) -> Result<Self, SbdError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            Err(SbdError::InvalidAlpha(alpha))
        } else {
            Ok(Self {
                beta: Beta::new_unchecked(1.0, alpha),
                inner: Arc::new(RwLock::new(_Inner {
                    remaining_mass: 1.0,
                    ln_weights: vec![0.0], // ln(1)
                    rng: seed.map_or_else(
                        Xoshiro128Plus::from_entropy,
                        Xoshiro128Plus::seed_from_u64,
                    ),
                })),
            })
        }
    }

    pub fn with_inner<F, Ans>(&self, f: F) -> Ans
    where
        F: FnOnce(&mut _Inner) -> Ans,
    {
        self.inner.write().map(|mut inner| f(&mut inner)).unwrap()
    }

    pub fn from_ln_weights(
        ln_weights: Vec<f64>,
        alpha: f64,
        seed: Option<u64>,
    ) -> Result<Self, SbdError> {
        let n_weights = ln_weights.len();

        let inner = _Inner {
            remaining_mass: ln_weights.last().unwrap().exp(),
            ln_weights,
            rng: seed.map_or_else(
                Xoshiro128Plus::from_entropy,
                Xoshiro128Plus::seed_from_u64,
            ),
        };

        Ok(Self {
            beta: Beta::new_unchecked(1.0, alpha),
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    pub fn from_weights(
        weights: &[f64],
        alpha: f64,
        seed: Option<u64>,
    ) -> Result<Self, SbdError> {
        let sum = weights.iter().sum::<f64>();
        if (sum - 1.0).abs() > 1e-13 {
            return Err(SbdError::WeightsDoNotSumToOne { sum });
        }
        let ln_weights = weights.iter().map(|w| w.ln()).collect();
        Self::from_ln_weights(ln_weights, alpha, seed)
    }

    pub fn from_canonical_weights(
        weights: &[f64],
        alpha: f64,
        seed: Option<u64>,
    ) -> Result<Self, SbdError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            Err(SbdError::InvalidAlpha(alpha))
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
                beta: Beta::new_unchecked(1.0, alpha),
                inner: Arc::new(RwLock::new(inner)),
            })
        }
    }

    // pub fn p_unobserved(&self) -> f64 {
    //     self.remaining_mass()
    // }

    pub fn alpha(&self) -> f64 {
        self.beta.beta()
    }
}

impl HasSuffStat<usize> for Sbd {
    type Stat = SbdSuffStat;

    fn empty_suffstat(&self) -> Self::Stat {
        SbdSuffStat::new()
    }

    fn ln_f_stat(&self, _stat: &Self::Stat) -> f64 {
        unimplemented!()
    }
}

impl Rv<usize> for Sbd {
    fn ln_f(&self, x: &usize) -> f64 {
        self.with_inner(|inner| {
            inner.extend_until(&self.beta, move |inner| {
                inner.ln_weights.len() > *x + 1
        })[*x]
        }).clone()
    }
         
    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        let u: f64 = rng.gen();

        let beta = self.beta.clone();
        self.with_inner(|inner| {
            let remaining_mass = inner.remaining_mass;
            let k = inner.num_cats();

            if u < 1.0 - remaining_mass {
                // TODO: Since we know the remaining mass, we can easily
                // normalize without needing logsumexp
                ln_pflip(&inner.ln_weights[..k], 1, false, rng)[0]
            } else {
                let ln_ws = inner.extend_until(&beta, |inner| {
                    inner.remaining_mass <= 1.0 - u
                });
                let ix = ln_pflip(&ln_ws, 1, false, rng)[0];
                ix + k
            }
        })
    }
}

impl Mode<usize> for Sbd {
    fn mode(&self) -> Option<usize> {
        let i_max = self.with_inner(|inner| {
            // TODO: Make this more efficient
            inner.extend_until(&self.beta, |inner| {
                argmax(&inner.ln_weights)[0] < inner.ln_weights.len() - 1
            });
            argmax(&inner.ln_weights)[0]
        });

        Some(i_max)
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn canonical_order_ln_f() {
        let sbd = Sbd::new(1.0, None).unwrap();
        let mut rm_mass = sbd.p_unobserved();
        for x in 0..10 {
            let ln_f_1 = sbd.ln_f(&x);
            let k = sbd.k();
            assert!(rm_mass > sbd.p_unobserved());
            rm_mass = sbd.p_unobserved();

            let ln_f_2 = sbd.ln_f(&x);

            assert_eq!(ln_f_1, ln_f_2);
            assert_eq!(k, sbd.k());
        }
    }

    #[test]
    fn static_ln_f_from_new() {
        let sbd = Sbd::new(1.0, None).unwrap();

        assert_eq!(sbd.k(), 0);

        let lnf0 = sbd.ln_f(&0_usize);
        assert::close(lnf0, sbd.ln_f(&0_usize), 1e-12);

        assert_eq!(sbd.k(), 1);

        let lnf1 = sbd.ln_f(&1_usize); // causes new category to form
        assert::close(lnf0, sbd.ln_f(&0_usize), 1e-12);
        assert_eq!(sbd.k(), 2);
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
        assert_eq!(sbd.k(), 0);

        sbd.ln_f(&2);
        assert_eq!(sbd.k(), 1);

        sbd.ln_f(&2);
        sbd.ln_f(&2);
        sbd.ln_f(&2);
        sbd.ln_f(&2);
        assert_eq!(sbd.k(), 1);

        sbd.ln_f(&0);
        assert_eq!(sbd.k(), 2);

        sbd.ln_f(&0);
        sbd.ln_f(&0);
        sbd.ln_f(&0);
        assert_eq!(sbd.k(), 2);
    }
}
