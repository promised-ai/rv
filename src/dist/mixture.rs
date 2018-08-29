extern crate rand;

use self::rand::Rng;
use misc::{logsumexp, pflip};
use std::io;
use traits::*;

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
    pub fn new(weights: Vec<f64>, components: Vec<Fx>) -> io::Result<Self> {
        let weights_ok = weights.iter().all(|&w| w >= 0.0)
            && (weights.iter().fold(0.0, |acc, &w| acc + w) - 1.0).abs()
                < 1E-12;
        let length_mismatch = weights.len() != components.len();
        if length_mismatch {
            let err_kind = io::ErrorKind::InvalidInput;
            let msg = "weights.len() != components.len()";
            let err = io::Error::new(err_kind, msg);
            Err(err)
        } else if !weights_ok {
            let err_kind = io::ErrorKind::InvalidInput;
            let msg = "weights must be positive and sum to 1";
            let err = io::Error::new(err_kind, msg);
            Err(err)
        } else {
            Ok(Mixture {
                weights,
                components,
            })
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    extern crate assert;
    use dist::{Gaussian, Poisson};

    const TOL: f64 = 1E-12;

    #[test]
    fn new_should_not_allow_bad_weights() {
        let components = vec![Gaussian::standard(), Gaussian::standard()];

        assert!(Mixture::new(vec![0.5, 0.51], components.clone()).is_err());
        assert!(Mixture::new(vec![0.5, 0.49], components.clone()).is_err());
        assert!(Mixture::new(vec![0.5, 0.5], components.clone()).is_ok());
    }

    #[test]
    fn new_should_not_allow_mismatched_inputs() {
        let components = vec![Gaussian::standard(), Gaussian::standard()];
        assert!(Mixture::new(vec![0.5, 0.3, 0.2], components.clone()).is_err());
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
        ).unwrap();

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
        ).unwrap();

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
}
