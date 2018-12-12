extern crate rand;

use self::rand::Rng;

use crate::misc::{logsumexp, pflip};
use crate::result;
use crate::traits::*;

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
pub struct Mixture<Fx> {
    /// The weights for each component distribution. All entries must be
    /// positive and sum to 1.
    pub weights: Vec<f64>,
    /// The component distributions.
    pub components: Vec<Fx>,
}

impl<Fx> Mixture<Fx> {
    pub fn new(weights: Vec<f64>, components: Vec<Fx>) -> result::Result<Self> {
        let weights_ok = weights.iter().all(|&w| w >= 0.0)
            && (weights.iter().fold(0.0, |acc, &w| acc + w) - 1.0).abs()
                < 1E-12;

        if weights.is_empty() || components.is_empty() {
            let err_kind = result::ErrorKind::EmptyContainerError;
            let msg = "weights or components was empty";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        } else if weights.len() != components.len() {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "weights.len() != components.len()";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        } else if !weights_ok {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "weights must be positive and sum to 1";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        } else {
            Ok(Mixture {
                weights,
                components,
            })
        }
    }

    /// Assume uniform component weights
    ///
    /// Given a n-length vector of components, automatically sets the component
    /// weights to 1/n.
    pub fn uniform(components: Vec<Fx>) -> result::Result<Self> {
        let k = components.len();
        let weights = vec![1.0 / k as f64; k];
        Ok(Mixture {
            weights,
            components,
        })
    }

    /// Combines many mixtures into one big mixture
    pub fn combine(mut mixtures: Vec<Mixture<Fx>>) -> result::Result<Self> {
        let k_total: usize = mixtures.iter().fold(0, |acc, mm| acc + mm.k());
        let nf = mixtures.len() as f64;

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

        Mixture::new(weights, components)
    }

    /// Number of components
    pub fn k(&self) -> usize {
        self.components.len()
    }
}

impl<X, Fx> Rv<X> for Mixture<Fx>
where
    Fx: Rv<X>,
{
    fn ln_f(&self, x: &X) -> f64 {
        let lfs: Vec<f64> = self
            .weights
            .iter()
            .zip(self.components.iter())
            .map(|(&w, cpnt)| w.ln() + cpnt.ln_f(&x))
            .collect();

        logsumexp(&lfs)
    }

    fn f(&self, x: &X) -> f64 {
        self.weights
            .iter()
            .zip(self.components.iter())
            .fold(0.0, |acc, (&w, cpnt)| acc + w * cpnt.f(&x))
    }

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
        self.components.iter().any(|cpnt| cpnt.supports(&x))
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
            .fold(0.0_f64, |acc, (&w, cpnt)| acc + w * cpnt.cdf(&x))
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
                if cpnt.supports(&x) {
                    acc + w * cpnt.f(&x)
                } else {
                    acc
                }
            },
        )
    }

    fn ln_pdf(&self, x: &X) -> f64 {
        self.pdf(&x).ln()
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
                if cpnt.supports(&x) {
                    acc + w * cpnt.f(&x)
                } else {
                    acc
                }
            },
        )
    }

    fn ln_pmf(&self, x: &X) -> f64 {
        self.pmf(&x).ln()
    }
}

macro_rules! continuous_uv_mean_and_var {
    ($kind: ty) => {
        impl<Fx> Mean<$kind> for Mixture<Fx>
        where
            Fx: ContinuousDistr<$kind> + Mean<$kind>,
        {
            fn mean(&self) -> Option<$kind> {
                let mut out: f64 = 0.0;
                for (w, cpnt) in self.weights.iter().zip(self.components.iter())
                {
                    match cpnt.mean() {
                        Some(m) => out += w * (m as f64),
                        None => return None,
                    }
                }
                Some(out as $kind / (self.k() as $kind))
            }
        }

        // https://stats.stackexchange.com/a/16609/36044
        impl<Fx> Variance<$kind> for Mixture<Fx>
        where
            Fx: ContinuousDistr<$kind> + Mean<$kind> + Variance<$kind>,
        {
            fn variance(&self) -> Option<$kind> {
                let mut p1: f64 = 0.0;
                let mut p2: f64 = 0.0;
                let mut p3: f64 = 0.0;
                for (w, cpnt) in self.weights.iter().zip(self.components.iter())
                {
                    match cpnt.mean() {
                        Some(m) => {
                            p1 += w * (m as f64).powi(2);
                            p3 += w * (m as f64);
                        }
                        None => return None,
                    }
                    match cpnt.variance() {
                        Some(v) => p2 += w * (v as f64),
                        None => return None,
                    }
                }
                let out: f64 = p1 + p2 - p3.powi(2);
                Some(out as $kind)
            }
        }
    };
}

continuous_uv_mean_and_var!(f32);
continuous_uv_mean_and_var!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    extern crate assert;
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
        assert!(Mixture::new(vec![0.5, 0.5], components.clone()).is_ok());
    }

    #[test]
    fn new_should_not_allow_empty_weights() {
        let components = vec![Gaussian::standard(), Gaussian::standard()];
        let empty_components: Vec<Gaussian> = vec![];

        assert!(Mixture::new(vec![], components.clone()).is_err());
        assert!(Mixture::new(vec![], empty_components).is_err());
    }

    #[test]
    fn new_should_not_allow_empty_components() {
        let empty_components: Vec<Gaussian> = vec![];
        assert!(Mixture::new(vec![0.5, 0.5], empty_components.clone()).is_err());
        assert!(Mixture::new(vec![], empty_components.clone()).is_err());
    }

    #[test]
    fn new_should_not_allow_mismatched_inputs() {
        let components = vec![Gaussian::standard(), Gaussian::standard()];
        assert!(Mixture::new(vec![0.5, 0.3, 0.2], components.clone()).is_err());
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

        let mmc = Mixture::combine(vec![mm1, mm2]).unwrap();

        assert::close(mmc.weights, vec![0.1, 0.4, 0.2, 0.3], TOL);
        assert::close(mmc.components[0].mu, 0.0, TOL);
        assert::close(mmc.components[1].mu, 1.0, TOL);
        assert::close(mmc.components[2].mu, 2.0, TOL);
        assert::close(mmc.components[3].mu, 3.0, TOL);
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
            -1.7105263157894737,
            -1.4210526315789473,
            -1.131578947368421,
            -0.8421052631578947,
            -0.5526315789473684,
            -0.26315789473684204,
            0.02631578947368407,
            0.3157894736842106,
            0.6052631578947372,
            0.8947368421052633,
            1.1842105263157894,
            1.473684210526316,
            1.7631578947368425,
            2.052631578947368,
            2.3421052631578947,
            2.6315789473684212,
            2.921052631578948,
            3.2105263157894743,
            3.5,
        ];

        let target: Vec<f64> = vec![
            0.06473380602589335,
            0.08497882889790014,
            0.11217654708252275,
            0.14501073352074248,
            0.1797629425978142,
            0.21067644115271195,
            0.23139135335682923,
            0.23702954614998323,
            0.2260163812419922,
            0.2007791966709858,
            0.167275199122399,
            0.13454608202641027,
            0.11515152871602127,
            0.12213065296617545,
            0.1549752516467795,
            0.18610946002808507,
            0.17957278989941344,
            0.1317193335426053,
            0.07419043357505534,
            0.03498448751938728,
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
            0.18944349223829393,
            0.20600928711172717,
            0.13638139292344745,
            0.09077999553365237,
            0.07005752694856147,
            0.05598752152045468,
            0.04412525380494339,
            0.035914449903067774,
            0.0314554228183013,
            0.02899187154010165,
            0.02660980083600876,
            0.023324644922075175,
            0.01914852009968446,
            0.014640805796680027,
            0.010432339897673344,
            0.006948125661306361,
            0.004340886637601523,
            0.002553064633728986,
            0.0014182807755675772,
            0.0007464449417949772,
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
        assert::close(m, 0.3333333333333333, TOL);
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
        assert::close(m, 0.16666666666666666, TOL);
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
}
