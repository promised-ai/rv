use arc_swap::ArcSwap;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "rkyv")]
use rkyv::{Archive, Deserialize, Serialize};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use super::HalfBeta;
use crate::experimental::stick::StickWeights;
use crate::traits::Rv;

// This allows us to searlize around all the ArcSwap stuff (in serde)
#[cfg(feature = "serde1")]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
struct StickSequenceFmt {
    breaker: HalfBeta,
    inner: _Inner,
}

#[cfg(feature = "serde1")]
impl From<StickSequenceFmt> for StickSequence {
    fn from(fmt: StickSequenceFmt) -> Self {
        Self {
            breaker: fmt.breaker,
            shared: Arc::new(SharedState::from_pointee(fmt.inner)),
        }
    }
}

#[cfg(feature = "serde1")]
impl From<StickSequence> for StickSequenceFmt {
    fn from(sticks: StickSequence) -> Self {
        Self {
            breaker: sticks.breaker.clone(),
            inner: (**sticks.shared.inner.load()).clone(),
        }
    }
}

// NOTE: We currently derive PartialEq, but this (we think) compares the
// internal state of the RNGs, which is probably not what we want.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(Serialize, Deserialize, Archive))]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct _Inner {
    #[cfg_attr(feature = "rkyv", rkyv(with = rkyv_support::XoshiroRkyvWrapper))]
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
            weights: vec![],
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

    fn push_break(&mut self, p: f64) -> f64 {
        let w = self.rm_mass * p;
        self.rm_mass -= w;
        self.weights.push(w);
        self.rm_mass
    }

    fn extend_once<B: Rv<f64> + Clone>(&mut self, breaker: &B) -> f64 {
        let p: f64 = breaker.draw(&mut self.rng);
        self.push_break(p)
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

#[derive(Debug)]
#[cfg_attr(feature = "rkyv", derive(Serialize, Deserialize, Archive))]
struct SharedState {
    #[cfg_attr(feature="rkyv", rkyv(with = rkyv_support::ArcSwapRkyvWrapper))]
    inner: ArcSwap<_Inner>,
    #[cfg_attr(feature="rkyv", rkyv(with = rkyv::with::Lock))]
    write_lock: Mutex<()>,
}

impl SharedState {
    fn from_pointee(inner: _Inner) -> Self {
        Self {
            inner: ArcSwap::from_pointee(inner),
            write_lock: Mutex::new(()),
        }
    }
}

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Serialize, Deserialize, Archive))]
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
    breaker: HalfBeta,
    shared: Arc<SharedState>,
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
    /// - `breaker`: A `HalfBeta` instance used as the breaker.
    /// - `seed`: An optional seed for the random number generator.
    ///
    /// # Returns
    /// A new instance of `StickSequence`.
    pub fn new(breaker: HalfBeta, seed: Option<u64>) -> Self {
        Self {
            breaker,
            shared: Arc::new(SharedState::from_pointee(_Inner::new(seed))),
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
            inner.push_break(p);
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
        // lock-free, wait-free read.
        f(&self.shared.inner.load())
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
        let _guard = self.shared.write_lock.lock().unwrap();
        let mut new_inner = (**self.shared.inner.load()).clone();

        let ans = f(&mut new_inner);

        self.shared.inner.store(Arc::new(new_inner));
        ans
    }

    /// Extend until the remaining mass is less than p and return the number of
    /// extensions.
    pub fn ensure_rm_mass(&self, p: f64) -> usize {
        if self.shared.inner.load().rm_mass < p {
            return 0;
        }

        let _guard = self.shared.write_lock.lock().unwrap();
        let current = self.shared.inner.load();
        if current.rm_mass < p {
            return 0;
        }

        let mut new_inner = (**current).clone();
        let extensions =
            new_inner.extend_until(&self.breaker, |inner| inner.rm_mass < p);
        self.shared.inner.store(std::sync::Arc::new(new_inner));
        extensions
    }

    pub fn ensure_breaks(&self, n: usize) {
        // FAST PATH: Truly lock-free check. 99.9% of calls will exit here instantly.
        if self.shared.inner.load().weights.len() > n {
            return;
        }

        // SLOW PATH: We need to extend. Serialize writers.
        let _guard = self.shared.write_lock.lock().unwrap();

        // DOUBLE-CHECK: Another thread might have extended the sequence while we waited for the lock.
        let current = self.shared.inner.load();
        if current.weights.len() > n {
            return;
        }

        let mut new_inner = (**current).clone();
        new_inner.extend_until(&self.breaker, |inner| inner.weights.len() > n);
        self.shared.inner.store(std::sync::Arc::new(new_inner));
    }

    /// Returns the number of weights instantiated so far.
    pub fn num_weights_unstable(&self) -> usize {
        self.with_inner(|inner| inner.weights.len())
    }

    /// Returns the weight of the n<sup>th</sup> stick.
    pub fn weight(&self, n: usize) -> f64 {
        self.with_inner(|inner| inner.weights[n])
    }

    /// Returns stick weights in a cloned `Vec`
    ///
    /// If the number of instantiated weights is less than `min_weights`, the
    /// weights will be extended.
    ///
    /// If you don't want to clone the weights, use `with_inner` and get the
    /// slice.
    pub fn weights(&self, min_weights: Option<usize>) -> StickWeights {
        if let Some(n) = min_weights {
            self.ensure_breaks(n);
        }
        self.with_inner(|inner| StickWeights(inner.weights.clone()))
    }

    /// Returns a reference of the breaker used in this `StickSequence`.
    pub fn breaker(&self) -> &HalfBeta {
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

#[cfg(feature = "rkyv")]
mod rkyv_support {
    use super::*;
    use rkyv::Place;
    use rkyv::rancor::Fallible;
    use rkyv::with::{ArchiveWith, DeserializeWith, SerializeWith};

    pub struct XoshiroRkyvWrapper;

    impl ArchiveWith<Xoshiro256Plus> for XoshiroRkyvWrapper {
        type Archived = <[u64; 4] as Archive>::Archived;
        type Resolver = <[u64; 4] as Archive>::Resolver;

        #[inline]
        fn resolve_with(
            field: &Xoshiro256Plus,
            resolver: Self::Resolver,
            out: Place<Self::Archived>,
        ) {
            // Safe: Xoshiro256Plus memory layout is a single [u64; 4]
            let state: &[u64; 4] = unsafe { core::mem::transmute(field) };
            state.resolve(resolver, out);
        }
    }

    impl<S: Fallible + ?Sized> SerializeWith<Xoshiro256Plus, S>
        for XoshiroRkyvWrapper
    where
        [u64; 4]: Serialize<S>,
    {
        #[inline]
        fn serialize_with(
            field: &Xoshiro256Plus,
            serializer: &mut S,
        ) -> Result<Self::Resolver, S::Error> {
            let state: &[u64; 4] = unsafe { core::mem::transmute(field) };
            state.serialize(serializer)
        }
    }

    impl<D: Fallible + ?Sized>
        DeserializeWith<<[u64; 4] as Archive>::Archived, Xoshiro256Plus, D>
        for XoshiroRkyvWrapper
    where
        <[u64; 4] as Archive>::Archived: Deserialize<[u64; 4], D>,
    {
        #[inline]
        fn deserialize_with(
            archived: &<[u64; 4] as Archive>::Archived,
            deserializer: &mut D,
        ) -> Result<Xoshiro256Plus, D::Error> {
            // Deserialize the array, then transmute back to the RNG state
            let state: [u64; 4] = archived.deserialize(deserializer)?;
            Ok(unsafe { core::mem::transmute(state) })
        }
    }

    pub struct ArcSwapRkyvWrapper;

    impl<T> ArchiveWith<ArcSwap<T>> for ArcSwapRkyvWrapper
    where
        Arc<T>: Archive,
    {
        type Archived = <Arc<T> as Archive>::Archived;
        type Resolver = <Arc<T> as Archive>::Resolver;

        #[inline]
        fn resolve_with(
            field: &ArcSwap<T>,
            resolver: Self::Resolver,
            out: Place<Self::Archived>,
        ) {
            // Grab a clone of the underlying Arc<T>
            let arc = field.load_full();
            arc.resolve(resolver, out);
        }
    }

    impl<T, S> SerializeWith<ArcSwap<T>, S> for ArcSwapRkyvWrapper
    where
        S: Fallible + ?Sized,
        Arc<T>: Serialize<S>,
    {
        #[inline]
        fn serialize_with(
            field: &ArcSwap<T>,
            serializer: &mut S,
        ) -> Result<Self::Resolver, S::Error> {
            let arc = field.load_full();
            arc.serialize(serializer)
        }
    }

    impl<T: Archive, D>
        DeserializeWith<<Arc<T> as Archive>::Archived, ArcSwap<T>, D>
        for ArcSwapRkyvWrapper
    where
        D: Fallible + ?Sized,
        <Arc<T> as Archive>::Archived: Deserialize<Arc<T>, D>,
    {
        #[inline]
        fn deserialize_with(
            archived: &<Arc<T> as Archive>::Archived,
            deserializer: &mut D,
        ) -> Result<ArcSwap<T>, D::Error> {
            // Deserialize the Arc<T>, then wrap it in a new ArcSwap
            let arc: Arc<T> = archived.deserialize(deserializer)?;
            Ok(ArcSwap::new(arc))
        }
    }
}
