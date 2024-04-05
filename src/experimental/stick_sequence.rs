use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

// use super::sticks_stat::StickBreakingSuffStat;
use crate::experimental::stick_breaking::PartialWeights;
use crate::prelude::UnitPowerLaw;
use crate::traits::*;

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
    pub fn new(breaker: UnitPowerLaw, seed: Option<u64>) -> Self {
        Self {
            breaker,
            inner: Arc::new(RwLock::new(_Inner::new(seed))),
        }
    }

    pub fn push_break(&self, p: f64) {
        self.with_inner_mut(|inner| {
            let remaining_mass = *inner.ccdf.last().unwrap();
            let new_remaining_mass = remaining_mass * p;
            inner.ccdf.push(new_remaining_mass);
        });
    }

    pub fn push_to_ccdf(&self, p: f64) {
        self.with_inner_mut(|inner| {
            assert!(p < *inner.ccdf.last().unwrap());
            inner.ccdf.push(p);
        });
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

    pub fn ensure_breaks(&self, n: usize) {
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
        self.ensure_breaks(n + 1);
        self.with_inner(|inner| {
            let ccdf = &inner.ccdf;
            ccdf[n] - ccdf[n + 1]
        })
    }

    /// Returns the weights of the first `n` sticks.
    /// Note that this includes sticks `0..n-1`, but not `n`.
    pub fn weights(&self, n: usize) -> PartialWeights {
        self.ensure_breaks(n);
        let w = self.with_inner(|inner| {
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
        });
        PartialWeights(w)
    }

    pub fn breaker(&self) -> UnitPowerLaw {
        self.breaker.clone()
    }

    pub fn extend_until<F>(&self, p: F)
    where
        F: Fn(&_Inner) -> bool,
    {
        self.with_inner_mut(|inner| inner.extend_until(&self.breaker, p));
    }
}

#[cfg(test)]
mod tests {
    use crate::experimental::StickSequence;
    use crate::prelude::UnitPowerLaw;

    #[test]
    fn test_stickseq_weights() {
        // test that `weights` gives the same as `weight` for all n
        let breaker = UnitPowerLaw::new(10.0).unwrap();
        let sticks = StickSequence::new(breaker, None);
        let weights = sticks.weights(100);
        assert_eq!(weights.0.len(), 100);
        for (n, w) in weights.0.iter().enumerate() {
            assert_eq!(sticks.weight(n), *w);
        }
    }

    #[test]
    fn test_push_to_ccdf() {
        let breaker = UnitPowerLaw::new(10.0).unwrap();
        let sticks = StickSequence::new(breaker, None);
        sticks.push_to_ccdf(0.9);
        sticks.push_to_ccdf(0.8);
        assert_eq!(sticks.ccdf(1), 0.9);
        assert_eq!(sticks.ccdf(2), 0.8);
    }

    #[test]
    fn test_push_break() {
        let breaker = UnitPowerLaw::new(10.0).unwrap();
        let sticks = StickSequence::new(breaker, None);
        sticks.push_break(0.9);
        sticks.push_break(0.8);
        assert::close(sticks.weights(2).0, vec![0.1, 0.9 - 0.9 * 0.8], 1e-10);
    }
}
