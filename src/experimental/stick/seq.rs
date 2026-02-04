use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use crate::prelude::UnitPowerLaw;
use crate::traits::Rv;

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
    // Remaining mass
    pub rm_mass: f64,
    // The weights of the instantiated sticks. The sum of weights is
    // `1.0 - rm_mass`.
    pub weights: Vec<f64>,
}

impl _Inner {
    fn new(seed: Option<u64>) -> _Inner {
        _Inner {
            rng: seed.map_or_else(
                Xoshiro256Plus::from_os_rng,
                Xoshiro256Plus::seed_from_u64,
            ),
            weights: vec![0.0],
            rm_mass: 1.0,
        }
    }

    /// The number of instantiated weights
    #[must_use]
    pub fn n_weights(&self) -> usize {
        self.weights.len()
    }

    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn extend_once<B: Rv<f64> + Clone>(&mut self, breaker: &B) -> f64 {
        let p: f64 = breaker.draw(&mut self.rng);
        let weight = self.rm_mass * p;

        self.rm_mass -= weight;
        self.weights.push(weight);
        self.rm_mass
    }

    /// Extend the stick sequence until the predicate, `p`, is satisfied.
    /// Returns the number of times the sequence was extended
    fn extend_until<B, F>(&mut self, breaker: &B, p: F) -> usize
    where
        B: Rv<f64> + Clone,
        F: Fn(&_Inner) -> bool,
    {
        let mut n_extended = 0;
        while !p(self) {
            self.extend_once(breaker);
            n_extended += 1;
        }
        n_extended
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
                    self_inner.weights == other_inner.weights
                        && self_inner.rng == other_inner.rng
                })
            })
    }
}

impl StickSequence {
    /// Creates a new `StickSequence` with the given breaker and optional seed.
    ///
    /// # Arguments
    /// - `breaker`: A `UnitPowerLaw` instance used as the breaker.
    /// - `seed`: An optional seed for the random number generator.
    ///
    /// # Returns
    /// A new instance of `StickSequence`.
    pub fn new(breaker: UnitPowerLaw, seed: Option<u64>) -> Self {
        Self {
            breaker,
            inner: Arc::new(RwLock::new(_Inner::new(seed))),
        }
    }

    /// Return the remaining mass of non-instantiated sticks
    pub fn rm_mass(&self) -> f64 {
        self.with_inner(|inner| inner.rm_mass)
    }

    /// Pushes a new weight to the stick sequence
    ///
    /// # Arguments
    /// - `w`: The new weight
    ///
    /// # Panics
    /// If `w` is greater than the remaining mass
    pub fn push_weight(&self, w: f64) {
        self.with_inner_mut(|inner| {
            assert!(w <= inner.rm_mass);
            inner.rm_mass -= w;
            inner.weights.push(w);
        });
    }

    /// Pushes a new break to the stick sequence
    ///
    /// # Notes
    /// This is distinct from `push_weight` in that `push_break` constructs the
    /// weight from the break probability.
    ///
    /// # Arguments
    /// - `p`: The new break probability
    pub fn push_break(&self, p: f64) {
        self.with_inner_mut(|inner| {
            let w = inner.rm_mass * p;
            inner.rm_mass -= w;
            inner.weights.push(w);
        });
    }

    /// Provides read access to the inner `_Inner` structure.
    ///
    /// # Type Parameters
    /// - `F`: A function type that takes a reference to `_Inner` and returns a
    ///    value of type `Ans`.
    /// - `Ans`: The return type of the function `f`.
    ///
    /// # Arguments
    /// - `f`: A function that is applied to the inner `_Inner` structure.
    ///
    /// # Returns
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
    /// - `F`: A function type that takes a mutable reference to `_Inner` and
    ///   returns a value of type `Ans`.
    /// - `Ans`: The return type of the function `f`.
    ///
    /// # Arguments
    /// - `f`: A function that is applied to the inner `_Inner` structure.
    ///
    /// # Returns
    /// The result of applying function `f` to the inner `_Inner` structure.
    pub fn with_inner_mut<F, Ans>(&self, f: F) -> Ans
    where
        F: FnOnce(&mut _Inner) -> Ans,
    {
        self.inner.write().map(|mut inner| f(&mut inner)).unwrap()
    }

    /// Extend until the remaining mass is less than p and return the number of
    /// extensions.
    pub fn ensure_rm_mass(&self, p: f64) -> usize {
        self.extend_until(|inner| inner.rm_mass < p)
    }

    /// Ensures that the weights vector is extended to at least `n` elements and
    /// return the number of extensions.
    pub fn ensure_breaks(&self, n: usize) {
        self.extend_until(|inner| inner.weights.len() > n);
    }

    /// Returns the number of weights instantiated so far.
    pub fn num_weights_unstable(&self) -> usize {
        self.with_inner(|inner| inner.weights.len())
    }

    /// Returns the weight of the n<sup>th</sup> stick.
    pub fn weight(&self, n: usize) -> f64 {
        self.with_inner(|inner| inner.weights[n])
    }

    /// Returns the instantiated stick weights in a cloned `Vec`
    ///
    /// If you don't want to clone the weights, use `with_inner`
    pub fn weights(&self) -> Vec<f64> {
        self.with_inner(|inner| inner.weights.clone())
    }

    /// Returns a reference of the breaker used in this `StickSequence`.
    pub fn breaker(&self) -> &UnitPowerLaw {
        &self.breaker
    }

    /// Extends the ccdf vector until a condition defined by `p` is met.
    ///
    /// # Type Parameters
    /// - `F`: A function type that takes a reference to `_Inner` and returns a
    ///   bool.
    ///
    /// # Arguments
    /// - `p`: A predicate function that determines when to stop extending the
    ///   weights vector.
    ///
    /// # Returns
    /// The number of times the stick sequence was extended
    pub fn extend_until<F>(&self, p: F) -> usize
    where
        F: Fn(&_Inner) -> bool,
    {
        self.with_inner_mut(|inner| inner.extend_until(&self.breaker, p))
    }
}
