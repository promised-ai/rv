use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

// use super::sticks_stat::StickBreakingSuffStat;
use crate::experimental::stick_breaking_process::stick_breaking::PartialWeights;
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

    pub fn ccdf(&self) -> &[f64] {
        &self.ccdf
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
    breaker: UnitPowerLaw,
    inner: Arc<RwLock<_Inner>>,
}

impl PartialEq<StickSequence> for StickSequence {
    fn eq(&self, other: &StickSequence) -> bool {
        self.ensure_breaks(other.num_weights_unstable());
        other.ensure_breaks(self.num_weights_unstable());
        self.breaker == other.breaker
            && self.with_inner(|self_inner| {
                other.with_inner(|other_inner| {
                    self_inner.ccdf == other_inner.ccdf
                        && self_inner.rng == other_inner.rng
                })
            })
    }
}

impl StickSequence {
    /// Creates a new StickSequence with the given breaker and optional seed.
    ///
    /// # Arguments
    ///
    /// * `breaker` - A `UnitPowerLaw` instance used as the breaker.
    /// * `seed` - An optional seed for the random number generator.
    ///
    /// # Returns
    ///
    /// A new instance of `StickSequence`.
    pub fn new(breaker: UnitPowerLaw, seed: Option<u64>) -> Self {
        Self {
            breaker,
            inner: Arc::new(RwLock::new(_Inner::new(seed))),
        }
    }

    /// Pushes a new break to the stick sequence using a given probability `p`.
    ///
    /// # Arguments
    ///
    /// * `p` - The probability used to calculate the new remaining mass.
    pub fn push_break(&self, p: f64) {
        self.with_inner_mut(|inner| {
            let remaining_mass = *inner.ccdf.last().unwrap();
            let new_remaining_mass = remaining_mass * p;
            inner.ccdf.push(new_remaining_mass);
        });
    }

    /// Pushes a new value `p` directly to the ccdf vector if `p` is less than the last element.
    ///
    /// # Arguments
    ///
    /// * `p` - The value to be pushed to the ccdf vector.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not less than the last element of the ccdf vector.
    pub fn push_to_ccdf(&self, p: f64) {
        self.with_inner_mut(|inner| {
            assert!(p < *inner.ccdf.last().unwrap());
            inner.ccdf.push(p);
        });
    }

    /// Extends the ccdf vector until a condition defined by `pred` is met, then applies function `f`.
    ///
    /// # Type Parameters
    ///
    /// * `P` - A predicate function type that takes a reference to a vector of f64 and returns a bool.
    /// * `F` - A function type that takes a reference to a vector of f64 and returns a value of type `Ans`.
    /// * `Ans` - The return type of the function `f`.
    ///
    /// # Arguments
    ///
    /// * `pred` - A predicate function that determines when to stop extending the ccdf vector.
    /// * `f` - A function to apply to the ccdf vector once the condition is met.
    ///
    /// # Returns
    ///
    /// The result of applying function `f` to the ccdf vector.
    pub fn extendmap_ccdf<P, F, Ans>(&self, pred: P, f: F) -> Ans
    where
        P: Fn(&Vec<f64>) -> bool,
        F: Fn(&Vec<f64>) -> Ans,
    {
        self.extend_until(|inner| pred(&inner.ccdf));
        self.with_inner(|inner| f(&inner.ccdf))
    }

    /// Provides read access to the inner `_Inner` structure.
    ///
    /// # Type Parameters
    ///
    /// * `F` - A function type that takes a reference to `_Inner` and returns a value of type `Ans`.
    /// * `Ans` - The return type of the function `f`.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that is applied to the inner `_Inner` structure.
    ///
    /// # Returns
    ///
    /// The result of applying function `f` to the inner `_Inner` structure.
    pub fn with_inner<F, Ans>(&self, f: F) -> Ans
    where
        F: FnOnce(&_Inner) -> Ans,
    {
        self.inner.read().map(|inner| f(&inner)).unwrap()
    }

    /// Provides write access to the inner `_Inner` structure.
    ///
    /// # Type Parameters
    ///
    /// * `F` - A function type that takes a mutable reference to `_Inner` and returns a value of type `Ans`.
    /// * `Ans` - The return type of the function `f`.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that is applied to the inner `_Inner` structure.
    ///
    /// # Returns
    ///
    /// The result of applying function `f` to the inner `_Inner` structure.
    pub fn with_inner_mut<F, Ans>(&self, f: F) -> Ans
    where
        F: FnOnce(&mut _Inner) -> Ans,
    {
        self.inner.write().map(|mut inner| f(&mut inner)).unwrap()
    }

    /// Ensures that the ccdf vector is extended to at least `n + 1` elements.
    ///
    /// # Arguments
    ///
    /// * `n` - The minimum number of elements the ccdf vector should have.
    pub fn ensure_breaks(&self, n: usize) {
        self.extend_until(|inner| inner.ccdf.len() > n);
    }

    /// Returns the `n`th element of the ccdf vector, ensuring the vector is long enough.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the element to retrieve from the ccdf vector.
    ///
    /// # Returns
    ///
    /// The `n`th element of the ccdf vector.
    pub fn ccdf(&self, n: usize) -> f64 {
        self.ensure_breaks(n);
        self.with_inner(|inner| {
            let ccdf = &inner.ccdf;
            ccdf[n]
        })
    }

    /// Returns the number of weights instantiated so far.
    ///
    /// # Returns
    ///
    /// The number of weights. This is "unstable" because it's a detail of the
    /// implementation that should not be depended on.
    pub fn num_weights_unstable(&self) -> usize {
        self.with_inner(|inner| inner.ccdf.len() - 1)
    }

    /// Returns the weight of the `n`th stick.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the stick whose weight is to be returned.
    ///
    /// # Returns
    ///
    /// The weight of the `n`th stick.
    pub fn weight(&self, n: usize) -> f64 {
        self.ensure_breaks(n + 1);
        self.with_inner(|inner| {
            let ccdf = &inner.ccdf;
            ccdf[n] - ccdf[n + 1]
        })
    }

    /// Returns the weights of the first `n` sticks.
    ///
    /// Note that this includes sticks `0..n-1`, but not `n`.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of sticks for which to return the weights.
    ///
    /// # Returns
    ///
    /// A `PartialWeights` instance containing the weights of the first `n` sticks.
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

    /// Returns a clone of the breaker used in this StickSequence.
    ///
    /// # Returns
    ///
    /// A clone of the `UnitPowerLaw` instance used as the breaker.
    pub fn breaker(&self) -> UnitPowerLaw {
        self.breaker.clone()
    }

    /// Extends the ccdf vector until a condition defined by `p` is met.
    ///
    /// # Type Parameters
    ///
    /// * `F` - A function type that takes a reference to `_Inner` and returns a bool.
    ///
    /// # Arguments
    ///
    /// * `p` - A predicate function that determines when to stop extending the ccdf vector.
    pub fn extend_until<F>(&self, p: F)
    where
        F: Fn(&_Inner) -> bool,
    {
        self.with_inner_mut(|inner| inner.extend_until(&self.breaker, p));
    }
}

#[cfg(test)]
mod tests {
    use crate::experimental::stick_breaking_process::StickSequence;
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
        assert::close(sticks.weights(2).0, vec![0.1, 0.18], 1e-10);
    }
}
