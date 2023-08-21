use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro128Plus;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use super::sbd_stat::SbdSuffStat;
use crate::dist::Beta;
use crate::misc::argmax;
use crate::misc::ln_pflip;
use crate::prelude::Mode;
use crate::traits::{HasSuffStat, Rv};

#[derive(Clone, Debug)]
pub enum SbdError {
    InvalidAlpha(f64),
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
        }
    }
}

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

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct _Inner {
    remaining_mass: f64,
    lookup: HashMap<usize, usize>,
    rev_lookup: HashMap<usize, usize>,
    // the bin weights. the last entry is ln(remaining_mass)
    pub ln_weights: Vec<f64>,
    rng: rand_xoshiro::Xoshiro128Plus,
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
            && self
                .inner
                .read()
                .and_then(|lhs| other.inner.read().map(|rhs| *rhs == *lhs))
                .unwrap()
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
                    lookup: HashMap::new(),
                    rev_lookup: HashMap::new(),
                    rng: seed.map_or_else(
                        Xoshiro128Plus::from_entropy,
                        Xoshiro128Plus::seed_from_u64,
                    ),
                })),
            })
        }
    }

    pub fn from_weights_and_lookup(
        weights: &[f64],
        lookup: HashMap<usize, usize>,
        alpha: f64,
        seed: Option<u64>,
    ) -> Result<Self, SbdError> {
        let k = weights.len() - 1;
        let inner = _Inner {
            remaining_mass: weights[k],
            ln_weights: weights.iter().map(|&w| w.ln()).collect(),
            rev_lookup: lookup.iter().map(|(a, b)| (*b, *a)).collect(),
            lookup,
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
                lookup: (0..k - 1).map(|x| (x, x)).collect(),
                rev_lookup: (0..k - 1).map(|x| (x, x)).collect(),
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
    }

    pub fn p_unobserved(&self) -> f64 {
        self.inner.read().map(|obj| obj.remaining_mass).unwrap()
    }

    pub fn alpha(&self) -> f64 {
        self.beta.beta()
    }

    pub fn k(&self) -> usize {
        self.inner
            .read()
            .map(|obj| obj.ln_weights.len() - 1)
            .unwrap()
    }

    fn extend(&self, x: usize) -> f64 {
        let p: f64 = self
            .inner
            .write()
            .map(|mut obj| self.beta.draw(&mut obj.rng))
            .unwrap();
        let rm_mass = self.inner.read().map(|obj| obj.remaining_mass).unwrap();
        let w = rm_mass * p;
        let rm_mass = rm_mass - w;

        let ln_w = w.ln();
        let k = self.k();

        self.inner
            .write()
            .map(|mut obj| {
                obj.remaining_mass = rm_mass;
                obj.lookup.insert(x, k);
                obj.rev_lookup.insert(k, x);
                obj.ln_weights
                    .last_mut()
                    .map(|last| *last = ln_w)
                    .expect("empty ln_weights");
                obj.ln_weights.push(rm_mass.ln());
            })
            .unwrap();

        ln_w
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
        let ix_opt = self
            .inner
            .read()
            .map(|obj| obj.lookup.get(x).copied())
            .unwrap();

        match ix_opt {
            Some(ix) => {
                self.inner.read().map(|obj| obj.ln_weights[ix]).unwrap()
            }
            None => self.extend(*x),
        }
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        let x = self
            .inner
            .read()
            .map(|obj| ln_pflip(&obj.ln_weights, 1, false, rng)[0])
            .unwrap();
        if x == self.k() {
            self.k() // FIXME: better way
        } else {
            self.inner.read().map(|obj| obj.rev_lookup[&x]).unwrap()
        }
    }
}

impl Mode<usize> for Sbd {
    fn mode(&self) -> Option<usize> {
        let k = self.k();
        Some(
            self.inner
                .read()
                .map(|inner| {
                    let ix = argmax(&inner.ln_weights[..k])[0];
                    inner.rev_lookup[&ix]
                })
                .unwrap(),
        )
    }
}

#[cfg(test)]
mod test {
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
}
