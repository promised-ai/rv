use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

// use super::sticks_stat::StickBreakingSuffStat;
use crate::traits::*;

// We'd like to be able to serialize and deserialize StickSequence, but serde can't handle
// `Arc` or `RwLock`. So we use `StickSequenceFmt<B>` as an intermediate type.
#[cfg(feature = "serde1")]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
struct StickSequenceFmt<B> {
    breaker: B,
    inner: _Inner,
}

#[cfg(feature = "serde1")]
impl<B: Rv<f64> + Clone> From<StickSequenceFmt<B>> for StickSequence<B> {
    fn from(fmt: StickSequenceFmt<B>) -> Self {
        Self {
            breaker: fmt.breaker,
            inner: Arc::new(RwLock::new(fmt.inner)),
        }
    }
}

#[cfg(feature = "serde1")]
impl<B: Rv<f64> + Clone> From<StickSequence<B>> for StickSequenceFmt<B> {
    fn from(sticks: StickSequence<B>) -> Self {
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

    fn extend<B: Rv<f64> + Clone>(&mut self, breaker: &B) -> f64 {
        let p: f64 = breaker.draw(&mut self.rng);
        let remaining_mass = self.ccdf.last().unwrap();
        let new_remaining_mass = remaining_mass * p;
        self.ccdf.push(new_remaining_mass);
        new_remaining_mass
    }

    fn extend_until<B, F>(&mut self, breaker: &B, p: F)
    where
        B: Rv<f64> + Clone,
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
        from = "StickSequenceFmt<B>        into = "StickSequenceFmt<B>   )
)]
#[derive(Clone, Debug)]
pub struct StickSequence<B> {
    pub breaker: B,
    pub inner: Arc<RwLock<_Inner>>,
}

// TODO: Extend to equal length, then check for equality
impl<B: Rv<f64> + Clone> PartialEq<StickSequence<B>> for StickSequence<B> {
    fn eq(&self, _other: &StickSequence<B>) -> bool {
        todo!()
    }
}

impl<B: Rv<f64> + Clone> StickSequence<B> {
    pub fn new(breaker: B, seed: Option<u64>) -> Self {
        Self {
            breaker,
            inner: Arc::new(RwLock::new(_Inner::new(seed))),
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

    pub fn breaker(&self) -> B {
        self.breaker.clone()
    }

    pub fn extend_until<F>(&self, p: F)
    where
        F: Fn(&_Inner) -> bool,
    {
        self.with_inner_mut(|inner| inner.extend_until(&self.breaker, p));
    }
}
