use crate::dist::Categorical;
use crate::dist::Gaussian;
use crate::dist::Mixture;
use crate::traits::*;

trait QuadBounds {
    fn quad_bounds(&self) -> (f64, f64);
}

impl QuadBounds for Gaussian {
    fn quad_bounds(&self) -> (f64, f64) {
        let span = self.mu + 6.0 * self.sigma;
        (-span, span)
    }
}

// Exact computation for categorical
impl Entropy for Mixture<Categorical> {
    fn entropy(&self) -> f64 {
        (0..self.components[0].k()).fold(0.0, |acc, x| {
            let ln_f = self.ln_f(&x);
            acc - ln_f.exp() * ln_f
        })
    }
}

// Exact computation for categorical
impl Entropy for Mixture<Mixture<Categorical>> {
    fn entropy(&self) -> f64 {
        let k = self.components[0].components[0].k();
        (0..k).fold(0.0, |acc, x| {
            let ln_f = self.ln_f(&x);
            acc - ln_f.exp() * ln_f
        })
    }
}

macro_rules! dual_step_quad_bounds {
    ($kind: ty) => {
        impl QuadBounds for $kind {
            fn quad_bounds(&self) -> (f64, f64) {
                let center: f64 = self.mean().unwrap();
                let step: f64 = self.variance().unwrap().sqrt();
                let mut lower = center - 4.0 * step;
                let mut upper = center + 4.0 * step;

                loop {
                    if self.pdf(&lower) < 1E-16 {
                        break;
                    } else {
                        lower -= step;
                    }
                }

                loop {
                    if self.pdf(&upper) < 1E-16 {
                        break;
                    } else {
                        upper += step;
                    }
                }

                (lower, upper)
            }
        }
    };
}

// Entropy by quadrature. Should be faster and more accurate than monte carlo
macro_rules! quadrature_entropy {
    ($kind: ty) => {
        impl Entropy for $kind {
            fn entropy(&self) -> f64 {
                let (lower, upper) = self.quad_bounds();
                let f = |x| {
                    let ln_f = self.ln_f(&x);
                    ln_f.exp() * ln_f
                };
                let result = quadrature::integrate(f, lower, upper, 1E-16);
                -result.integral
            }
        }
    };
}

dual_step_quad_bounds!(Mixture<Gaussian>);
dual_step_quad_bounds!(Mixture<Mixture<Gaussian>>);

quadrature_entropy!(Mixture<Gaussian>);
quadrature_entropy!(Mixture<Mixture<Gaussian>>);

#[cfg(test)]
mod tests {
    extern crate assert;
    use super::*;

    const TOL: f64 = 1E-8;

    #[test]
    fn gauss_mixture_entropy() {
        let components = vec![Gaussian::standard(), Gaussian::standard()];
        let weights = vec![0.5, 0.5];
        let mm = Mixture::new(weights, components).unwrap();

        let h: f64 = mm.entropy();
        assert::close(h, 1.4189385332046727, TOL);
    }

    #[test]
    fn gauss_mixture_mixture_entropy() {
        let components = vec![
            {
                let components =
                    vec![Gaussian::standard(), Gaussian::standard()];
                let weights = vec![0.5, 0.5];
                Mixture::new(weights, components).unwrap()
            },
            {
                let components =
                    vec![Gaussian::standard(), Gaussian::standard()];
                let weights = vec![0.5, 0.5];
                Mixture::new(weights, components).unwrap()
            },
        ];
        let weights = vec![0.5, 0.5];
        let mm = Mixture::new(weights, components).unwrap();

        let h: f64 = mm.entropy();
        assert::close(h, 1.4189385332046727, TOL);
    }

    #[test]
    fn categorical_mixture_entropy() {
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
        assert::close(h, 1.2798542258336676, TOL);
    }

    #[test]
    fn gauss_mixture_quad_bounds_have_zero_pdf() {
        use crate::dist::{InvGamma, Poisson};
        use crate::traits::Rv;

        let mut rng = rand::thread_rng();
        let pois = Poisson::new(7.0).unwrap();

        let mu_prior = Gaussian::new(0.0, 5.0).unwrap();
        let sigma_prior = InvGamma::new(2.0, 3.0).unwrap();
        let bad_bounds = (0..100).find(|_| {
            // TODO: should probably implement Rv<usize> for Poisson...
            let n: usize = <Poisson as Rv<u32>>::draw(&pois, &mut rng) as usize;

            let components: Vec<Gaussian> = (0..n)
                .map(|_| {
                    let mu: f64 = mu_prior.draw(&mut rng);
                    let sigma: f64 = sigma_prior.draw(&mut rng);
                    Gaussian::new(mu, sigma).unwrap()
                })
                .collect();

            let mm = Mixture::uniform(components).unwrap();
            let (a, b) = mm.quad_bounds();

            mm.pdf(&a).abs() > 1E-16 || mm.pdf(&b).abs() > 1E-16
        });

        assert!(bad_bounds.is_none());
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
        assert::close(h, 2.051658739391058, 1E-7);
    }
}
