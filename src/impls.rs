extern crate quadrature;

use dist::Categorical;
use dist::Gaussian;
use dist::Mixture;
use traits::*;

trait QuadBounds {
    fn quad_bounds(&self) -> (f64, f64);
}

impl QuadBounds for Gaussian {
    fn quad_bounds(&self) -> (f64, f64) {
        let span = self.mu + 6.0 * self.sigma;
        (-span, span)
    }
}

impl Entropy for Mixture<Categorical> {
    fn entropy(&self) -> f64 {
        (0..self.k()).fold(0.0, |acc, x| {
            let ln_f = self.ln_f(&x);
            acc - ln_f.exp() * ln_f
        })
    }
}

impl Entropy for Mixture<Mixture<Categorical>> {
    fn entropy(&self) -> f64 {
        let k = self.components[0].k();
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
}
