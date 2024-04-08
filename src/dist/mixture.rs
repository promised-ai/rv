#[cfg(feature = "serde1")]
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
#[cfg(feature = "serde1")]
use serde::ser::{SerializeStruct, Serializer};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::dist::{Categorical, Gaussian, Poisson};
use crate::misc::{logsumexp, pflip};
use crate::traits::*;
use rand::Rng;
use std::fmt;
use std::sync::OnceLock;

/// [Mixture distribution](https://en.wikipedia.org/wiki/Mixture_model)
/// Σ w<sub>i</sub> f(x|θ<sub>i</sub>)
///
/// A mixture distribution is a convex combination of distributions.
///
/// # Example
///
/// A bimodal Gaussian mixture model
///
/// ```
/// use rv::prelude::*;
///
/// let g1 = Gaussian::new(-2.5, 1.0).unwrap();
/// let g2 = Gaussian::new(2.0, 2.1).unwrap();
///
/// // f(x) = 0.6 * N(-2.5, 1.0) + 0.4 * N(2.0, 2.1)
/// let mm = Mixture::new(vec![0.6, 0.4], vec![g1, g2]).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Mixture<Fx> {
    /// The weights for each component distribution. All entries must be
    /// positive and sum to 1.
    weights: Vec<f64>,
    /// The component distributions.
    components: Vec<Fx>,
    // Cached ln(weights)
    ln_weights: OnceLock<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum MixtureError {
    /// The weights vector is empty
    WeightsEmpty,
    /// The weights to not sum to one
    WeightsDoNotSumToOne { sum: f64 },
    /// One or more weights is less than zero
    WeightTooLow { ix: usize, weight: f64 },
    /// The components vector is empty
    ComponentsEmpty,
    /// The components vector and the weights vector are different lengths
    ComponentWeightLengthMismatch {
        /// length of the weights vector
        n_weights: usize,
        /// length of the components vector
        n_components: usize,
    },
}

#[inline]
fn validate_weights(weights: &[f64]) -> Result<(), MixtureError> {
    if weights.is_empty() {
        return Err(MixtureError::WeightsEmpty);
    }

    weights
        .iter()
        .enumerate()
        .try_fold(0.0, |sum, (ix, &weight)| {
            if weight < 0.0 {
                Err(MixtureError::WeightTooLow { ix, weight })
            } else {
                Ok(sum + weight)
            }
        })
        .and_then(|sum| {
            if (sum - 1.0).abs() > 1E-12 {
                Err(MixtureError::WeightsDoNotSumToOne { sum })
            } else {
                Ok(())
            }
        })
}

impl<Fx> Mixture<Fx> {
    /// Create a new mixture distribution
    ///
    /// # Arguments
    /// - weights: The weights for each component distribution. All entries
    ///   must be positive and sum to 1.
    /// - components: The component distributions.
    pub fn new(
        weights: Vec<f64>,
        components: Vec<Fx>,
    ) -> Result<Self, MixtureError> {
        if weights.is_empty() {
            Err(MixtureError::WeightsEmpty)
        } else if components.is_empty() {
            Err(MixtureError::ComponentsEmpty)
        } else if components.len() != weights.len() {
            Err(MixtureError::ComponentWeightLengthMismatch {
                n_weights: weights.len(),
                n_components: components.len(),
            })
        } else {
            Ok(())
        }?;

        validate_weights(&weights)?;

        Ok(Mixture {
            weights,
            components,
            ln_weights: OnceLock::new(),
        })
    }

    pub fn ln_weights(&self) -> &[f64] {
        self.ln_weights
            .get_or_init(|| self.weights.iter().map(|&w| w.ln()).collect())
    }

    /// Creates a new Mixture without checking whether the parameters are valid.
    #[inline]
    pub fn new_unchecked(weights: Vec<f64>, components: Vec<Fx>) -> Self {
        Mixture {
            weights,
            components,
            ln_weights: OnceLock::new(),
        }
    }

    /// Assume uniform component weights
    ///
    /// Given a n-length vector of components, automatically sets the component
    /// weights to 1/n.
    #[inline]
    pub fn uniform(components: Vec<Fx>) -> Result<Self, MixtureError> {
        if components.is_empty() {
            Err(MixtureError::ComponentsEmpty)
        } else {
            let k = components.len();
            let weights = vec![1.0 / k as f64; k];
            Ok(Mixture {
                weights,
                components,
                ln_weights: OnceLock::new(),
            })
        }
    }

    /// Combines many mixtures into one big mixture
    ///
    /// # Notes
    ///
    /// Assumes mixtures are valid.
    pub fn combine(mut mixtures: Vec<Mixture<Fx>>) -> Self {
        // The total number of components
        let k_total: usize = mixtures.iter().fold(0, |acc, mm| acc + mm.k());

        // The number of non-empty mixtures
        let n: u32 = mixtures
            .iter()
            .map(|mm| if mm.k() == 0 { 0 } else { 1 })
            .sum();

        if n == 0 {
            // everything is empty, return empty
            return Mixture::new_unchecked(vec![], vec![]);
        }

        let nf = f64::from(n);

        let mut weights: Vec<f64> = Vec::with_capacity(k_total);
        let mut components: Vec<Fx> = Vec::with_capacity(k_total);

        mixtures.iter_mut().for_each(|mm| {
            mm.weights.drain(..).zip(mm.components.drain(..)).for_each(
                |(w, cpnt)| {
                    weights.push(w / nf);
                    components.push(cpnt);
                },
            );
        });

        Mixture::new_unchecked(weights, components)
    }

    /// Number of components
    #[inline]
    pub fn k(&self) -> usize {
        self.components.len()
    }

    /// Get a reference to the component weights
    #[inline]
    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }

    /// Get a reference to the components
    #[inline]
    pub fn components(&self) -> &Vec<Fx> {
        &self.components
    }

    /// Set the mixture weights
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Mixture;
    /// use rv::dist::Gaussian;
    ///
    /// let components = vec![
    ///     Gaussian::new(-2.0, 1.0).unwrap(),
    ///     Gaussian::new(2.0, 1.0).unwrap(),
    /// ];
    ///
    /// let mut mm = Mixture::uniform(components).unwrap();
    /// assert_eq!(mm.weights(), &vec![0.5, 0.5]);
    ///
    /// mm.set_weights(vec![0.2, 0.8]).unwrap();
    /// assert_eq!(mm.weights(), &vec![0.2, 0.8]);
    /// ```
    ///
    /// Will error for invalid weights
    ///
    /// ```rust
    /// # use rv::dist::Mixture;
    /// # use rv::dist::Gaussian;
    /// # let components = vec![
    /// #     Gaussian::new(-2.0, 1.0).unwrap(),
    /// #     Gaussian::new(2.0, 1.0).unwrap(),
    /// # ];
    /// # let mut mm = Mixture::uniform(components).unwrap();
    /// // This is fine
    /// assert!(mm.set_weights(vec![0.2, 0.8]).is_ok());
    ///
    /// // Does not sum to 1
    /// assert!(mm.set_weights(vec![0.1, 0.8]).is_err());
    ///
    /// // Wrong number of weights
    /// assert!(mm.set_weights(vec![0.1, 0.1, 0.8]).is_err());
    ///
    /// // Negative weight
    /// assert!(mm.set_weights(vec![-0.1, 1.1]).is_err());
    ///
    /// // Zero weight are ok
    /// assert!(mm.set_weights(vec![0.0, 1.0]).is_ok());
    /// ```
    #[inline]
    pub fn set_weights(
        &mut self,
        weights: Vec<f64>,
    ) -> Result<(), MixtureError> {
        if weights.len() != self.components.len() {
            return Err(MixtureError::ComponentWeightLengthMismatch {
                n_components: self.components.len(),
                n_weights: weights.len(),
            });
        };

        validate_weights(&weights)?;

        self.ln_weights = OnceLock::new();
        self.weights = weights;
        Ok(())
    }

    #[inline]
    pub fn set_weights_unchecked(&mut self, weights: Vec<f64>) {
        self.ln_weights = OnceLock::new();
        self.weights = weights;
    }

    #[inline]
    /// Set the mixture components
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Mixture;
    /// use rv::dist::Gaussian;
    /// use rv::traits::Mean;
    ///
    /// let components = vec![
    ///     Gaussian::new(-2.0, 1.0).unwrap(),
    ///     Gaussian::new(2.0, 1.0).unwrap(),
    /// ];
    ///
    /// let mut mm = Mixture::uniform(components).unwrap();
    /// let mean_1: f64 = mm.mean().unwrap();
    /// assert_eq!(mean_1, 0.0);
    ///
    /// let components_2 = vec![
    ///     Gaussian::new(-3.0, 1.0).unwrap(),
    ///     Gaussian::new(2.0, 1.0).unwrap(),
    /// ];
    /// mm.set_components(components_2).unwrap();
    /// let mean_2: f64 = mm.mean().unwrap();
    /// assert_eq!(mean_2, -0.5);
    /// ```
    pub fn set_components(
        &mut self,
        components: Vec<Fx>,
    ) -> Result<(), MixtureError> {
        if components.len() != self.components.len() {
            Err(MixtureError::ComponentWeightLengthMismatch {
                n_components: components.len(),
                n_weights: self.weights.len(),
            })
        } else {
            self.components = components;
            Ok(())
        }
    }

    #[inline]
    pub fn set_components_unchecked(&mut self, components: Vec<Fx>) {
        self.components = components;
    }
}

impl<Fx> TryFrom<Vec<(f64, Fx)>> for Mixture<Fx> {
    type Error = MixtureError;

    fn try_from(
        mut weighted_components: Vec<(f64, Fx)>,
    ) -> Result<Self, Self::Error> {
        let k = weighted_components.len();
        let mut components = Vec::with_capacity(k);
        let mut weights = Vec::with_capacity(k);

        weighted_components.drain(..).for_each(|(w, cpnt)| {
            weights.push(w);
            components.push(cpnt);
        });

        Mixture::new(weights, components)
    }
}

impl<Fx> From<Mixture<Fx>> for Vec<(f64, Fx)> {
    fn from(mut mm: Mixture<Fx>) -> Self {
        mm.weights.drain(..).zip(mm.components.drain(..)).collect()
    }
}

impl<X, Fx> HasDensity<X> for Mixture<Fx>
where
    Fx: Rv<X>,
{
    fn ln_f(&self, x: &X) -> f64 {
        let lfs: Vec<f64> = self
            .ln_weights()
            .iter()
            .zip(self.components.iter())
            .map(|(&w, cpnt)| w + cpnt.ln_f(x))
            .collect();

        logsumexp(&lfs)
    }

    fn f(&self, x: &X) -> f64 {
        self.weights
            .iter()
            .zip(self.components.iter())
            .fold(0.0, |acc, (&w, cpnt)| cpnt.f(x).mul_add(w, acc))
    }
}

impl<X, Fx> Sampleable<X> for Mixture<Fx>
where
    Fx: Rv<X>,
{
    fn draw<R: Rng>(&self, mut rng: &mut R) -> X {
        let k: usize = pflip(&self.weights, 1, &mut rng)[0];
        self.components[k].draw(&mut rng)
    }

    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<X> {
        pflip(&self.weights, n, &mut rng)
            .iter()
            .map(|&k| self.components[k].draw(&mut rng))
            .collect()
    }
}

// XXX: Not quite sure how this should work. I'd like to have mixtures of
// things with different support.
impl<X, Fx> Support<X> for Mixture<Fx>
where
    Fx: Rv<X> + Support<X>,
{
    fn supports(&self, x: &X) -> bool {
        self.components.iter().any(|cpnt| cpnt.supports(x))
    }
}

impl<X, Fx> Cdf<X> for Mixture<Fx>
where
    Fx: Rv<X> + Cdf<X>,
{
    fn cdf(&self, x: &X) -> f64 {
        self.weights
            .iter()
            .zip(self.components.iter())
            .fold(0.0_f64, |acc, (&w, cpnt)| cpnt.cdf(x).mul_add(w, acc))
    }
}

impl<X, Fx> ContinuousDistr<X> for Mixture<Fx>
where
    Fx: Rv<X> + ContinuousDistr<X>,
{
    fn pdf(&self, x: &X) -> f64 {
        self.weights.iter().zip(self.components.iter()).fold(
            0.0,
            |acc, (&w, cpnt)| {
                if cpnt.supports(x) {
                    w.mul_add(cpnt.f(x), acc)
                } else {
                    acc
                }
            },
        )
    }

    fn ln_pdf(&self, x: &X) -> f64 {
        self.pdf(x).ln()
    }
}

impl<X, Fx> DiscreteDistr<X> for Mixture<Fx>
where
    Fx: Rv<X> + DiscreteDistr<X>,
{
    fn pmf(&self, x: &X) -> f64 {
        self.weights.iter().zip(self.components.iter()).fold(
            0.0,
            |acc, (&w, cpnt)| {
                if cpnt.supports(x) {
                    w.mul_add(cpnt.f(x), acc)
                } else {
                    acc
                }
            },
        )
    }

    fn ln_pmf(&self, x: &X) -> f64 {
        self.pmf(x).ln()
    }
}

macro_rules! uv_mean_and_var {
    ($kind: ty) => {
        impl<Fx> Mean<$kind> for Mixture<Fx>
        where
            Fx: Mean<$kind>,
        {
            fn mean(&self) -> Option<$kind> {
                self.weights
                    .iter()
                    .zip(self.components.iter())
                    .try_fold(0_f64, |grand_mean, (&w, cpnt)| {
                        cpnt.mean()
                            .map(|mean| w.mul_add(mean as f64, grand_mean))
                    })
                    .map(|mean| mean as $kind)
            }
        }

        // https://stats.stackexchange.com/a/16609/36044
        impl<Fx> Variance<$kind> for Mixture<Fx>
        where
            Fx: Mean<$kind> + Variance<$kind>,
        {
            fn variance(&self) -> Option<$kind> {
                let mut p1: f64 = 0.0;
                let mut p2: f64 = 0.0;
                let mut p3: f64 = 0.0;
                for (w, cpnt) in self.weights.iter().zip(self.components.iter())
                {
                    match cpnt.mean() {
                        Some(m) => {
                            let mf = m as f64;
                            p1 += w * mf * mf;
                            p3 += w * mf;
                        }
                        None => return None,
                    }
                    match cpnt.variance() {
                        Some(v) => p2 += w * (v as f64),
                        None => return None,
                    }
                }
                let out: f64 = p3.mul_add(-p3, p1 + p2);
                Some(out as $kind)
            }
        }
    };
}

uv_mean_and_var!(f32);
uv_mean_and_var!(f64);

#[cfg(feature = "serde1")]
impl<Fx> Serialize for Mixture<Fx>
where
    Fx: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Mixture", 2)?;
        state.serialize_field("weights", &self.weights)?;
        state.serialize_field("components", &self.components)?;
        state.end()
    }
}

#[cfg(feature = "serde1")]
impl<'de, Fx> Deserialize<'de> for Mixture<Fx>
where
    Fx: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use std::marker::PhantomData;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Weights,
            Components,
        }

        struct MixtureVisitor<Fx> {
            _fx: PhantomData<Fx>,
        }

        impl<Fx> MixtureVisitor<Fx> {
            fn new() -> Self {
                MixtureVisitor { _fx: PhantomData }
            }
        }

        impl<'de, Fx> Visitor<'de> for MixtureVisitor<Fx>
        where
            Fx: Deserialize<'de>,
        {
            type Value = Mixture<Fx>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Mixture")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Mixture<Fx>, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let weights = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let components = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(Mixture {
                    weights,
                    components,
                    ln_weights: OnceLock::new(),
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<Mixture<Fx>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut weights = None;
                let mut components = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Weights => {
                            if weights.is_some() {
                                return Err(de::Error::duplicate_field(
                                    "weights",
                                ));
                            }
                            weights = Some(map.next_value()?);
                        }
                        Field::Components => {
                            if components.is_some() {
                                return Err(de::Error::duplicate_field(
                                    "components",
                                ));
                            }
                            components = Some(map.next_value()?);
                        }
                    }
                }
                let weights = weights
                    .ok_or_else(|| de::Error::missing_field("weights"))?;
                let components = components
                    .ok_or_else(|| de::Error::missing_field("components"))?;

                Ok(Mixture {
                    weights,
                    components,
                    ln_weights: OnceLock::new(),
                })
            }
        }

        const FIELDS: &[&str] = &["weights", "components"];
        deserializer.deserialize_struct(
            "Mixture",
            FIELDS,
            MixtureVisitor::new(),
        )
    }
}

impl std::error::Error for MixtureError {}

impl fmt::Display for MixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WeightsEmpty => write!(f, "empty weights vector"),
            Self::ComponentsEmpty => write!(f, "empty components vector"),
            Self::WeightsDoNotSumToOne { sum } => {
                write!(f, "weights sum to {} but should sum to one", sum)
            }
            Self::WeightTooLow { ix, weight } => {
                write!(f, "weight at index {} was too low: {} <= 0", ix, weight)
            }
            Self::ComponentWeightLengthMismatch {
                n_weights,
                n_components,
            } => write!(
                f,
                "weights and components had a different number of \
                    entries. weights had {} entries but components had {} \
                    entries",
                n_weights, n_components
            ),
        }
    }
}

macro_rules! bernmix_entropy {
    ($type:ty) => {
        impl Entropy for Mixture<$type> {
            fn entropy(&self) -> f64 {
                let ln_f_true = self.ln_f(&true);
                let ln_f_false = self.ln_f(&false);

                ln_f_true
                    .exp()
                    .mul_add(ln_f_true, ln_f_false.exp() * ln_f_false)
            }
        }
    };
}

bernmix_entropy!(crate::dist::Bernoulli);

macro_rules! catmix_entropy {
    ($type:ty) => {
        // Exact computation for categorical
        impl Entropy for Mixture<$type> {
            fn entropy(&self) -> f64 {
                (0..self.components()[0].k()).fold(0.0, |acc, x| {
                    let ln_f = self.ln_f(&x);
                    ln_f.exp().mul_add(-ln_f, acc)
                })
            }
        }
    };
}

catmix_entropy!(Categorical);

macro_rules! countmix_entropy {
    ($type:ty) => {
        // Approximation for count-type distribution
        impl Entropy for Mixture<$type> {
            fn entropy(&self) -> f64 {
                if self.k() == 1 {
                    crate::misc::entropy::count_entropy(
                        self,
                        self.mean().unwrap() as u32,
                    )
                } else {
                    let (min_mean, max_mean) = {
                        let mut ma = self.components[0].mean().unwrap();
                        let mut mb = self.components[1].mean().unwrap();
                        if ma > mb {
                            std::mem::swap(&mut ma, &mut mb);
                        }
                        (ma, mb)
                    };
                    let (lower, upper) = self.components.iter().fold(
                        (min_mean, max_mean),
                        |(lower, upper), cpnt| {
                            let mean =
                                cpnt.mean().expect("distr always has a mean");
                            if mean > upper {
                                (lower, mean)
                            } else if mean < lower {
                                (mean, upper)
                            } else {
                                (lower, upper)
                            }
                        },
                    );
                    crate::misc::entropy::count_entropy_range(
                        self,
                        ((lower + upper) / 2.0) as u32,
                        lower as u32,
                        upper as u32,
                    )
                }
            }
        }
    };
}

countmix_entropy!(Poisson);

macro_rules! dual_step_quad_bounds {
    ($kind: ty) => {
        impl QuadBounds for $kind {
            fn quad_bounds(&self) -> (f64, f64) {
                let center: f64 = self.mean().unwrap();
                self.components().iter().fold(
                    (center, center),
                    |(mut left, mut right), cpnt| {
                        let (a, b) = cpnt.quad_bounds();
                        if a < left {
                            left = a;
                        }
                        if b > right {
                            right = b;
                        }
                        (left, right)
                    },
                )
            }
        }
    };
}

fn sort_mixture_by_mode<Fx>(mm: Mixture<Fx>) -> Mixture<Fx>
where
    Fx: Mode<f64>,
{
    let mut components: Vec<(f64, Fx)> = mm.into();
    components.sort_by(|a, b| {
        a.1.mode()
            .partial_cmp(&b.1.mode())
            .unwrap_or(std::cmp::Ordering::Less)
    });
    Mixture::<Fx>::try_from(components).unwrap()
}

fn continuous_mixture_quad_points<Fx>(mm: &Mixture<Fx>) -> Vec<f64>
where
    Fx: Mode<f64> + Variance<f64>,
{
    let mut state = (None, None);

    mm.components()
        .iter()
        .filter_map(|cpnt| {
            let mode = cpnt.mode();
            let std = cpnt.variance().map(|v| v.sqrt());
            match (&state, (mode, std)) {
                ((Some(m1), s1), (Some(m2), s2)) => {
                    if (m2 - *m1)
                        > s1.unwrap_or(f64::INFINITY).min(s2.unwrap_or(f64::INFINITY))
                    {
                        state = (mode, std);
                        Some(m2)
                    } else {
                        None
                    }
                }
                ((None, _), (Some(m2), _)) => {
                    state = (mode, std);
                    Some(m2)
                }
                _ => None,
            }
        })
        .collect()
}

/// Computes an integral over a continuous mixture
pub fn cm_quad<F, Fx>(f: &F, mm: &Mixture<Fx>) -> f64
where
    F: Fn(f64) -> f64,
    Fx: Mode<f64> + Variance<f64>,
    Mixture<Fx>: Clone + QuadBounds,
{
    use crate::misc::{gauss_legendre_quadrature_cached, gauss_legendre_table};

    let quad_level = 16;
    let (weights, roots) = gauss_legendre_table(quad_level);

    let (lower, upper) = mm.quad_bounds();

    let points = {
        let tmp_mm = sort_mixture_by_mode(mm.clone());
        continuous_mixture_quad_points(&tmp_mm)
    };

    let last_ix = points.len() - 1;
    let q_a = gauss_legendre_quadrature_cached(
        f,
        (lower, points[0]),
        &weights,
        &roots,
    );
    let q_b = gauss_legendre_quadrature_cached(
        f,
        (points[last_ix], upper),
        &weights,
        &roots,
    );

    let mut right = points[0];
    let q_m = points
        .iter()
        .skip(1)
        .map(|&x| {
            let q = gauss_legendre_quadrature_cached(
                f,
                (right, x),
                &weights,
                &roots,
            );
            right = x;
            q
        })
        .sum::<f64>();

    q_a + q_m + q_b
}

// Entropy by quadrature. Should be faster and more accurate than monte carlo
macro_rules! quadrature_entropy {
    ($kind: ty) => {
        impl Entropy for $kind {
            fn entropy(&self) -> f64 {
                let f = |x| {
                    let ln_f = self.ln_f(&x);
                    ln_f.exp() * ln_f
                };

                -cm_quad(&f, self)
            }
        }
    };
}

dual_step_quad_bounds!(Mixture<Gaussian>);

quadrature_entropy!(Mixture<Gaussian>);

macro_rules! ds_discrete_quad_bounds {
    ($fxtype:ty, $xtype:ty, $minval:expr, $maxval:expr) => {
        impl QuadBounds for $fxtype {
            fn quad_bounds(&self) -> (f64, f64) {
                let mean = self.mean().unwrap();
                let (mut left, mut right) = {
                    let left = mean.floor() as $xtype;
                    let mut right = mean.ceil() as $xtype;
                    if left == right {
                        right += 1;
                    }
                    (left, right)
                };

                while self.f(&left) > 1e-16 && left > $minval {
                    left -= 1;
                }

                while self.f(&right) > 1e-16 && right < $maxval {
                    right -= 1;
                }

                (left as f64, right as f64)
            }
        }
    };
}

ds_discrete_quad_bounds!(Mixture<Poisson>, u32, 0, u32::MAX);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{Gaussian, Poisson};

    const TOL: f64 = 1E-12;

    #[test]
    fn uniform_ctor() {
        let components = vec![Gaussian::standard(), Gaussian::standard()];
        let mm = Mixture::uniform(components).unwrap();

        assert_eq!(mm.weights.len(), 2);
        assert_eq!(mm.k(), 2);
        assert::close(mm.weights[0], 0.5, TOL);
        assert::close(mm.weights[1], 0.5, TOL);
    }

    #[test]
    fn new_should_not_allow_bad_weights() {
        let components = vec![Gaussian::standard(), Gaussian::standard()];

        assert!(Mixture::new(vec![0.5, 0.51], components.clone()).is_err());
        assert!(Mixture::new(vec![0.5, 0.49], components.clone()).is_err());
        assert!(Mixture::new(vec![0.5, 0.5], components).is_ok());
    }

    #[test]
    fn new_should_not_allow_empty_weights() {
        let components = vec![Gaussian::standard(), Gaussian::standard()];
        let empty_components: Vec<Gaussian> = vec![];

        assert!(Mixture::new(vec![], components).is_err());
        assert!(Mixture::new(vec![], empty_components).is_err());
    }

    #[test]
    fn new_should_not_allow_empty_components() {
        let empty_components: Vec<Gaussian> = vec![];
        assert!(Mixture::new(vec![0.5, 0.5], empty_components.clone()).is_err());
        assert!(Mixture::new(vec![], empty_components).is_err());
    }

    #[test]
    fn new_should_not_allow_mismatched_inputs() {
        let components = vec![Gaussian::standard(), Gaussian::standard()];
        assert!(Mixture::new(vec![0.5, 0.3, 0.2], components).is_err());
    }

    #[test]
    fn combine() {
        let mm1 = {
            let components = vec![
                Gaussian::new(0.0, 1.0).unwrap(),
                Gaussian::new(1.0, 1.0).unwrap(),
            ];
            Mixture::new(vec![0.2, 0.8], components).unwrap()
        };
        let mm2 = {
            let components = vec![
                Gaussian::new(2.0, 1.0).unwrap(),
                Gaussian::new(3.0, 1.0).unwrap(),
            ];
            Mixture::new(vec![0.4, 0.6], components).unwrap()
        };

        let mmc = Mixture::combine(vec![mm1, mm2]);

        assert::close(mmc.weights, vec![0.1, 0.4, 0.2, 0.3], TOL);
        assert::close(mmc.components[0].mu(), 0.0, TOL);
        assert::close(mmc.components[1].mu(), 1.0, TOL);
        assert::close(mmc.components[2].mu(), 2.0, TOL);
        assert::close(mmc.components[3].mu(), 3.0, TOL);
    }

    #[test]
    fn combine_with_some_empty() {
        let mm1 = {
            let components = vec![
                Gaussian::new(0.0, 1.0).unwrap(),
                Gaussian::new(1.0, 1.0).unwrap(),
            ];
            Mixture::new(vec![0.2, 0.8], components).unwrap()
        };

        // Force an empty mixture
        let mm2 = Mixture::<Gaussian>::new_unchecked(vec![], vec![]);

        let mmc = Mixture::combine(vec![mm1, mm2]);

        assert::close(mmc.weights, vec![0.2, 0.8], TOL);
        assert::close(mmc.components[0].mu(), 0.0, TOL);
        assert::close(mmc.components[1].mu(), 1.0, TOL);
    }

    #[test]
    fn combine_with_all_empty() {
        // Force an empty mixture
        let mm1 = Mixture::<Gaussian>::new_unchecked(vec![], vec![]);
        let mm2 = Mixture::<Gaussian>::new_unchecked(vec![], vec![]);

        let mmc = Mixture::combine(vec![mm1, mm2]);

        assert!(mmc.weights.is_empty());
        assert!(mmc.components.is_empty());
    }

    #[test]
    fn mean_of_sample_should_be_weighted_dist_means_uniform() {
        let mut rng = rand::thread_rng();
        let mm = Mixture::new(
            vec![0.5, 0.5],
            vec![
                Gaussian::new(-3.0, 1.0).unwrap(),
                Gaussian::new(3.0, 3.0).unwrap(),
            ],
        )
        .unwrap();

        // using sample
        let xbar: f64 = mm
            .sample(100_000, &mut rng)
            .iter()
            .fold(0.0_f64, |acc, &x: &f64| acc + x)
            / 100_000.0;
        assert::close(xbar, 0.0, 0.05);

        // using draw
        let ybar: f64 = (0..100_000)
            .map(|_| mm.draw(&mut rng))
            .fold(0.0_f64, |acc, x: f64| acc + x)
            / 100_000.0;
        assert::close(ybar, 0.0, 0.05);
    }

    #[test]
    fn mean_of_sample_should_be_weighted_dist_means_nonuniform() {
        let mut rng = rand::thread_rng();
        let mm = Mixture::new(
            vec![0.9, 0.1],
            vec![
                Gaussian::new(-3.0, 1.0).unwrap(),
                Gaussian::new(3.0, 3.0).unwrap(),
            ],
        )
        .unwrap();

        // using sample
        let xbar: f64 = mm
            .sample(100_000, &mut rng)
            .iter()
            .fold(0.0_f64, |acc, &x: &f64| acc + x)
            / 100_000.0;
        assert::close(xbar, -2.4, 0.05);

        // using draw
        let ybar: f64 = (0..100_000)
            .map(|_| mm.draw(&mut rng))
            .fold(0.0_f64, |acc, x: f64| acc + x)
            / 100_000.0;
        assert::close(ybar, -2.4, 0.05);
    }

    #[test]
    fn continuous_pdf() {
        let xs: Vec<f64> = vec![
            -2.0,
            -1.710_526_315_789_473_7,
            -1.421_052_631_578_947_3,
            -1.131_578_947_368_421,
            -0.842_105_263_157_894_7,
            -0.552_631_578_947_368_4,
            -0.263_157_894_736_842_04,
            0.026_315_789_473_684_07,
            0.315_789_473_684_210_6,
            0.605_263_157_894_737_2,
            0.894_736_842_105_263_3,
            1.184_210_526_315_789_4,
            1.473_684_210_526_316,
            1.763_157_894_736_842_5,
            2.052_631_578_947_368,
            2.342_105_263_157_894_7,
            2.631_578_947_368_421_2,
            2.921_052_631_578_948,
            3.210_526_315_789_474_3,
            3.5,
        ];

        let target: Vec<f64> = vec![
            0.064_733_806_025_893_35,
            0.084_978_828_897_900_14,
            0.112_176_547_082_522_75,
            0.145_010_733_520_742_48,
            0.179_762_942_597_814_2,
            0.210_676_441_152_711_95,
            0.231_391_353_356_829_23,
            0.237_029_546_149_983_23,
            0.226_016_381_241_992_2,
            0.200_779_196_670_985_8,
            0.167_275_199_122_399,
            0.134_546_082_026_410_27,
            0.115_151_528_716_021_27,
            0.122_130_652_966_175_45,
            0.154_975_251_646_779_5,
            0.186_109_460_028_085_07,
            0.179_572_789_899_413_44,
            0.131_719_333_542_605_3,
            0.074_190_433_575_055_34,
            0.034_984_487_519_387_28,
        ];

        let components = vec![
            Gaussian::new(0.0, 1.0).unwrap(),
            Gaussian::new(-1.0, 3.0).unwrap(),
            Gaussian::new(2.5, 0.5).unwrap(),
        ];
        let weights = vec![0.5, 0.3, 0.2];

        let mm = Mixture::new(weights, components).unwrap();

        let fx: Vec<f64> = xs.iter().map(|x| mm.pdf(x)).collect();

        assert::close(fx, target, TOL);
    }

    #[test]
    fn discrete_pdf() {
        let xs: Vec<u32> = (0..20).map(|i| i as u32).collect();

        let target: Vec<f64> = vec![
            0.189_443_492_238_293_93,
            0.206_009_287_111_727_17,
            0.136_381_392_923_447_45,
            0.090_779_995_533_652_37,
            0.070_057_526_948_561_47,
            0.055_987_521_520_454_68,
            0.044_125_253_804_943_39,
            0.035_914_449_903_067_774,
            0.031_455_422_818_301_3,
            0.028_991_871_540_101_65,
            0.026_609_800_836_008_76,
            0.023_324_644_922_075_175,
            0.019_148_520_099_684_46,
            0.014_640_805_796_680_027,
            0.010_432_339_897_673_344,
            0.006_948_125_661_306_361,
            0.004_340_886_637_601_523,
            0.002_553_064_633_728_986,
            0.001_418_280_775_567_577_2,
            0.000_746_444_941_794_977_2,
        ];

        let components = vec![
            Poisson::new(1.0).unwrap(),
            Poisson::new(4.0).unwrap(),
            Poisson::new(10.0).unwrap(),
        ];
        let weights = vec![0.5, 0.3, 0.2];

        let mm = Mixture::new(weights, components).unwrap();

        let fx: Vec<f64> = xs.iter().map(|x| mm.pmf(x)).collect();

        assert::close(fx, target, TOL);
    }

    #[test]
    fn cdf_for_overlapping_gaussians() {
        let components = vec![
            Gaussian::standard(),
            Gaussian::standard(),
            Gaussian::standard(),
        ];
        let weights = vec![1.0 / 3.0; 3];
        let mm = Mixture::new(weights, components).unwrap();

        assert::close(mm.cdf(&0.0_f64), 0.5, TOL);
    }

    #[test]
    fn cdf_for_symmetric_gaussian_mixture() {
        let components = vec![
            Gaussian::new(-1.0, 3.0).unwrap(),
            Gaussian::new(1.0, 1.0).unwrap(),
            Gaussian::new(3.0, 3.0).unwrap(),
        ];
        let weights = vec![1.0 / 3.0; 3];
        let mm = Mixture::new(weights, components).unwrap();

        assert::close(mm.cdf(&1.0_f64), 0.5, TOL);
    }

    #[test]
    fn mean_even_weight() {
        let components = vec![
            Gaussian::new(-1.0, 3.0).unwrap(),
            Gaussian::new(1.0, 1.0).unwrap(),
            Gaussian::new(3.0, 3.0).unwrap(),
        ];
        let weights = vec![1.0 / 3.0; 3];
        let mm = Mixture::new(weights, components).unwrap();

        let m: f64 = mm.mean().unwrap();
        assert::close(m, 1.0, TOL);
    }

    #[test]
    fn mean_uneven_weight() {
        let components = vec![
            Gaussian::new(-1.0, 3.0).unwrap(),
            Gaussian::new(1.0, 1.0).unwrap(),
            Gaussian::new(3.0, 3.0).unwrap(),
        ];
        let weights = vec![0.5, 0.25, 0.25];
        let mm = Mixture::new(weights, components).unwrap();

        let m: f64 = mm.mean().unwrap();
        assert::close(m, 0.5, TOL);
    }

    #[test]
    fn variance_even_weight() {
        let components = vec![
            Gaussian::new(1.0, 3.0).unwrap(),
            Gaussian::new(3.0, 1.0).unwrap(),
        ];
        let weights = vec![0.5, 0.5];
        let mm = Mixture::new(weights, components).unwrap();

        let v: f64 = mm.variance().unwrap();
        assert::close(v, 6.0, TOL);
    }

    #[test]
    fn variance_uneven_weight() {
        let components = vec![
            Gaussian::new(1.0, 3.0).unwrap(),
            Gaussian::new(3.0, 1.0).unwrap(),
        ];
        let weights = vec![0.25, 0.75];
        let mm = Mixture::new(weights, components).unwrap();

        let v: f64 = mm.variance().unwrap();
        assert::close(v, 3.75, TOL);
    }

    #[cfg(feature = "serde1")]
    #[test]
    fn mixture_serde() {
        let components = vec![
            Gaussian::new(1.0, 3.0).unwrap(),
            Gaussian::new(3.0, 1.0).unwrap(),
        ];
        let weights = vec![0.25, 0.75];
        let mm1 = Mixture::new(weights, components).unwrap();

        let s1 = serde_yaml::to_string(&mm1).unwrap();
        let mm2: Mixture<Gaussian> =
            serde_yaml::from_slice(s1.as_bytes()).unwrap();
        let s2 = serde_yaml::to_string(&mm2).unwrap();

        assert_eq!(s1, s2);
    }

    #[cfg(test)]
    mod mixture_impls {
        use super::*;
        const TOL: f64 = 1E-8;

        #[test]
        fn gauss_mixture_entropy() {
            let components = vec![Gaussian::standard(), Gaussian::standard()];
            let weights = vec![0.5, 0.5];
            let mm = Mixture::new(weights, components).unwrap();

            let h: f64 = mm.entropy();
            assert::close(h, 1.418_938_533_204_672_7, 1E-10);
        }

        #[test]
        fn categorical_mixture_entropy_0() {
            let components = vec![
                {
                    let weights: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
                    Categorical::new(&weights).unwrap()
                },
                {
                    let weights: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
                    Categorical::new(&weights).unwrap()
                },
            ];
            let weights = vec![0.5, 0.5];
            let mm = Mixture::new(weights, components).unwrap();
            let h: f64 = mm.entropy();
            assert::close(h, 1.279_854_225_833_667_6, TOL);
        }

        #[test]
        fn categorical_mixture_entropy_1() {
            let components = vec![
                {
                    let weights: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
                    Categorical::new(&weights).unwrap()
                },
                {
                    let weights: Vec<f64> = vec![4.0, 3.0, 2.0, 1.0];
                    Categorical::new(&weights).unwrap()
                },
            ];
            let weights = vec![0.5, 0.5];
            let mm = Mixture::new(weights, components).unwrap();
            let h: f64 = mm.entropy();
            assert::close(h, -(0.25_f64.ln()), TOL);
        }

        #[test]
        fn categorical_mixture_entropy_after_combine() {
            let m1 = {
                let weights: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
                let cat = Categorical::new(&weights).unwrap();
                let components = vec![cat];
                Mixture::new(vec![1.0], components).unwrap()
            };

            let m2 = {
                let weights: Vec<f64> = vec![4.0, 3.0, 2.0, 1.0];
                let cat = Categorical::new(&weights).unwrap();
                let components = vec![cat];
                Mixture::new(vec![1.0], components).unwrap()
            };

            let mm = Mixture::combine(vec![m1, m2]);

            let h: f64 = mm.entropy();
            assert::close(h, -(0.25_f64.ln()), TOL);
        }

        #[test]
        fn gauss_mixture_quad_bounds_have_zero_pdf() {
            use crate::dist::{InvGamma, Poisson};

            let mut rng = rand::thread_rng();
            let pois = Poisson::new(7.0).unwrap();

            let mu_prior = Gaussian::new(0.0, 5.0).unwrap();
            let sigma_prior = InvGamma::new(2.0, 3.0).unwrap();
            let bad_bounds = (0..100).find(|_| {
                let n: usize = pois.draw(&mut rng);

                let components: Vec<Gaussian> = (0..=n)
                    .map(|_| {
                        let mu: f64 = mu_prior.draw(&mut rng);
                        let sigma: f64 = sigma_prior.draw(&mut rng);
                        Gaussian::new(mu, sigma).unwrap()
                    })
                    .collect();

                let mm = Mixture::uniform(components).unwrap();
                let (a, b) = mm.quad_bounds();
                let pdf_a = mm.pdf(&a);
                let pdf_b = mm.pdf(&b);

                pdf_a > 1E-10 || pdf_b > 1E-10
            });

            assert!(bad_bounds.is_none());
        }

        #[test]
        fn spread_out_gauss_mixture_quad_bounds() {
            let g1 = Gaussian::new(0.0, 0.1).unwrap();
            let g2 = Gaussian::new(10.0, 0.5).unwrap();
            let g3 = Gaussian::new(20.0, 0.2).unwrap();

            let mm = Mixture::uniform(vec![g1, g2, g3]).unwrap();

            let (a, b) = mm.quad_bounds();

            assert!(a < 0.0);
            assert!(b > 20.0);
        }

        #[test]
        fn gauss_2_component_mixture_entropy() {
            let components = vec![
                Gaussian::new(-2.0, 1.0).unwrap(),
                Gaussian::new(2.0, 1.0).unwrap(),
            ];
            let weights = vec![0.5, 0.5];
            let mm = Mixture::new(weights, components).unwrap();

            let h: f64 = mm.entropy();
            // Answer from numerical integration in python
            assert::close(h, 2.051_658_739_391_058, 1E-7);
        }

        #[cfg(feature = "serde1")]
        #[test]
        fn messy_jsd_should_be_positive() {
            // recreates bug in dependent crate where JSD was negative
            let mm_str =
                std::fs::read_to_string("resources/gauss-mm.yaml").unwrap();
            let mm: Mixture<Gaussian> = serde_yaml::from_str(&mm_str).unwrap();
            let sum_h = mm
                .weights()
                .iter()
                .zip(mm.components().iter())
                .map(|(&w, cpnt)| w * cpnt.entropy())
                .sum::<f64>();
            let jsd = mm.entropy() - sum_h;
            assert!(0.0 < jsd);
        }

        #[test]
        fn poisson_2_entropy() {
            let components =
                vec![Poisson::new(1.2).unwrap(), Poisson::new(20.0).unwrap()];
            let mm = Mixture::uniform(components).unwrap();

            let entropy =
                -(0_u32..10_000).map(|x| mm.ln_f(&x) * mm.f(&x)).sum::<f64>();
            assert::close(entropy, mm.entropy(), 1e-3);
        }

        #[cfg(feature = "serde1")]
        #[test]
        fn super_messy_jsd_should_be_positive() {
            // recreates bug in dependent crate where JSD was negative
            let mm_str =
                std::fs::read_to_string("resources/satellites-mixture.json")
                    .unwrap();
            let mm: Mixture<Gaussian> = serde_json::from_str(&mm_str).unwrap();
            let sum_h = mm
                .weights()
                .iter()
                .zip(mm.components().iter())
                .map(|(&w, cpnt)| w * cpnt.entropy())
                .sum::<f64>();

            let mm_entropy = mm.entropy();
            let jsd = mm_entropy - sum_h;
            assert!(0.0 < jsd);
        }
    }

    macro_rules! mm_unit_variance {
        ($distr: ident, $f: expr) => {
            mod $distr {
                use super::*;
                use $crate::dist::$distr::*;

                #[test]
                fn unit_variance() {
                    let f1 = $f.unwrap();
                    let f2 = $f.unwrap();

                    let v1: f64 = f1.variance().unwrap();

                    let mm =
                        $crate::dist::Mixture::uniform(vec![f1, f2]).unwrap();

                    let v2: f64 = mm.variance().unwrap();

                    assert::close(v1, v2, 1E-12);
                }
            }
        };
    }

    mm_unit_variance!(beta, Beta::new(1.5, 2.3));
    mm_unit_variance!(beta_binom, BetaBinomial::new(20, 1.5, 2.3));
    mm_unit_variance!(chi_squared, ChiSquared::new(3.2));
    mm_unit_variance!(exponential, Exponential::new(4.3));
    mm_unit_variance!(gamma, Gamma::new(2.1, 1.5));
    mm_unit_variance!(gaussian, Gaussian::new(-2.0, 1.0));
    mm_unit_variance!(poisson, Poisson::new(7.21));
}
