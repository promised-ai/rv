use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

// use super::sticks_stat::StickBreakingSuffStat;
use crate::dist::UnitPowerLaw;
use crate::traits::Rv;

#[derive(Clone, Debug)]
pub enum StickSequenceError {
    InvalidAlpha(f64),
    InvalidNumberOfWeights { n_weights: usize, n_entries: usize },
    WeightsDoNotSumToOne { sum: f64 },
}

impl std::error::Error for StickSequenceError {}

impl std::fmt::Display for StickSequenceError {
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

// We'd like to be able to serialize and deserialize StickSequence, but serde can't handle
// `Arc` or `RwLock`. So we use `StickSequenceFmt` as an intermediate type.
#[cfg(feature = "serde1")]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
struct StickSequenceFmt {
    breaker: UnitPowerLaw,
    inner: _Inner,
}

#[cfg(feature = "serde1")]
impl From<StickSequenceFmt> for StickSequence {
    fn from(fmt: StickSequenceFmt) -> Self {
        Self {
            breaker: fmt.breaker,
            inner: Arc::new(RwLock::new(fmt.inner)),
        }
    }
}

#[cfg(feature = "serde1")]
impl From<StickSequence> for StickSequenceFmt {
    fn from(sticks: StickSequence) -> Self {
        Self {
            breaker: sticks.breaker,
            inner: sticks.inner.read().map(|inner| inner.clone()).unwrap(),
        }
    }
}

// NOTE: We currently derive PartialEq, but this (we think) compares the
// internal state of the RNGs, which is probably not what we want.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct _Inner {
    rng: Xoshiro256Plus,
    ccdf: Vec<f64>,
}

impl _Inner {
    fn new(seed: Option<u64>) -> _Inner {
        _Inner {
            rng: seed.map_or_else(
                Xoshiro256Plus::from_entropy,
                Xoshiro256Plus::seed_from_u64,
            ),
            ccdf: vec![1.0],
        }
    }

    fn extend(&mut self, breaker: &UnitPowerLaw) -> f64 {
        let p: f64 = breaker.draw(&mut self.rng);
        let remaining_mass = self.ccdf.last().unwrap();
        let new_remaining_mass = remaining_mass * p;
        self.ccdf.push(new_remaining_mass);
        new_remaining_mass
    }

    fn extend_until<F>(&mut self, breaker: &UnitPowerLaw, p: F)
    where
        F: Fn(&_Inner) -> bool,
    {
        while !p(self) {
            self.extend(breaker);
        }
    }
}

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde1",
    serde(
        rename_all = "snake_case",
        from = "StickSequenceFmt",
        into = "StickSequenceFmt"
    )
)]
#[derive(Clone, Debug)]
pub struct StickSequence {
    pub breaker: UnitPowerLaw,
    pub inner: Arc<RwLock<_Inner>>,
}

// TODO: Extend to equal length, then check for equality
impl PartialEq<StickSequence> for StickSequence {
    fn eq(&self, _other: &StickSequence) -> bool {
        todo!()
    }
}

impl StickSequence {
    pub fn new(
        alpha: f64,
        seed: Option<u64>,
    ) -> Result<Self, StickSequenceError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            Err(StickSequenceError::InvalidAlpha(alpha))
        } else {
            Ok(Self {
                breaker: UnitPowerLaw::new_unchecked(alpha),
                inner: Arc::new(RwLock::new(_Inner::new(seed))),
            })
        }
    }

    pub fn extendmap_ccdf<P, F, Ans>(&self, pred: P, f: F) -> Ans
    where
        P: Fn(&Vec<f64>) -> bool,
        F: Fn(&Vec<f64>) -> Ans,
    {
        self.extend_until(|inner| pred(&inner.ccdf));
        self.with_inner(|inner| f(&inner.ccdf))
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

    fn ensure_breaks(&self, n: usize) {
        self.extend_until(|inner| inner.ccdf.len() > n);
    }

    pub fn ccdf(&self, n: usize) -> f64 {
        self.ensure_breaks(n);
        self.with_inner(|inner| {
            let ccdf = &inner.ccdf;
            ccdf[n]
        })
    }

    pub fn weight(&self, n: usize) -> f64 {
        self.ensure_breaks(n);
        self.with_inner(|inner| {
            let ccdf = &inner.ccdf;
            ccdf[n + 1] - ccdf[n]
        })
    }

    pub fn weights(&self, n: usize) -> Vec<f64> {
        self.ensure_breaks(n);
        self.with_inner(|inner| {
            let mut last_p = 1.0;
            inner
                .ccdf
                .iter()
                .skip(1)
                .map(|&p| {
                    let w = last_p - p;
                    last_p = p;
                    w
                })
                .collect()
        })
    }

    pub fn alpha(&self) -> f64 {
        self.breaker.alpha()
    }

    pub fn extend_until<F>(&self, p: F)
    where
        F: Fn(&_Inner) -> bool,
    {
        self.with_inner_mut(|inner| inner.extend_until(&self.breaker, p));
    }
}
