extern crate rand;

use self::rand::Rng;
use misc::{logsumexp, pflip};
use std::io;
use traits::*;

pub struct Mixture<Fx> {
    pub weights: Vec<f64>,
    pub components: Vec<Fx>,
}

impl<Fx> Mixture<Fx> {
    pub fn new(weights: Vec<f64>, components: Vec<Fx>) -> io::Result<Self> {
        let weights_sum = weights.iter().fold(0.0, |acc, &w| acc + w);
        let length_mismatch = weights.len() != components.len();
        if length_mismatch {
            let err_kind = io::ErrorKind::InvalidInput;
            let msg = "weights.len() != components.len()";
            let err = io::Error::new(err_kind, msg);
            Err(err)
        } else if (weights_sum - 1.0).abs() > 1E-12 {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "weights must sum to 1");
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
    use dist::Gaussian;

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
}
