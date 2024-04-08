#[cfg(test)]
use approx::RelativeEq;

use crate::traits::Rv;
use std::collections::BTreeMap;

// tests that Clone, Debug, and PartialEq are implemented for a distribution
// Tests that partial eq is not sensitive to OnceCell initialization, which
// often happens in ln_f is called
#[macro_export]
macro_rules! test_basic_impls {
    ([continuous] $fx: expr) => {
        test_basic_impls!($fx, 0.5_f64, impls);
    };
    ([categorical] $fx: expr) => {
        test_basic_impls!($fx, 0_usize, impls);
    };
    ([count] $fx: expr) => {
        test_basic_impls!($fx, 3_u32, impls);
    };
    ([binary] $fx: expr) => {
        test_basic_impls!($fx, true, impls);
    };
    ($fx: expr, $x: expr) => {
        test_basic_impls!($fx, $x, impls);
    };
    ($fx: expr, $x: expr, $mod: ident) => {
        mod $mod {
            use super::*;

            #[test]
            fn should_impl_debug_clone_and_partialeq() {
                // make the expression a thing. If we don't do this, calling $fx
                // reconstructs the distribution which means we don't do caching
                let fx = $fx;

                // clone a copy of fn before any computation of cached values is
                // done
                let fx2 = fx.clone();
                assert_eq!($fx, fx2);

                // Computing ln_f normally initializes all cached values
                let y1 = fx.ln_f(&$x);
                let y2 = fx.ln_f(&$x);
                assert!((y1 - y2).abs() < f64::EPSILON);

                // check the fx == fx2 despite fx having its cached values initalized
                assert_eq!(fx2, fx);

                // Make sure Debug is implemented for fx
                let _s1 = format!("{:?}", fx);
            }
        }
    };
}

#[macro_export]
macro_rules! verify_cache_resets {
    ([unchecked],
     $fn_name: ident,
     $set_fn: ident,
     $start_dist: expr,
     $x: expr,
     $start_value: expr,
     $change_value: expr
     ) => {
        #[test]
        fn $fn_name() {
            let mut dist = $start_dist;
            let x = $x;

            // cache should initialize during this call
            let ln_f_0 = dist.ln_f(&x);
            // this call should use the cache
            let ln_f_1 = dist.ln_f(&x);

            assert!((ln_f_0 - ln_f_1).abs() < 1e-10);

            // set the cache to the wrong thing
            dist.$set_fn($change_value);
            let _ = dist.ln_f(&x);

            // reset alpha and empty cache
            dist.$set_fn($start_value);

            // this call should use the cache
            let ln_f_2 = dist.ln_f(&x);
            assert!((ln_f_2 - ln_f_1).abs() < 1e-10);
        }
    };
    ([checked],
     $fn_name: ident,
     $set_fn: ident,
     $start_dist: expr,
     $x: expr,
     $start_value: expr,
     $change_value: expr
     ) => {
        #[test]
        fn $fn_name() {
            let mut dist = $start_dist;
            let x = $x;

            // cache should initialize during this call
            let ln_f_0 = dist.ln_f(&x);
            // this call should use the cache
            let ln_f_1 = dist.ln_f(&x);

            assert!((ln_f_0 - ln_f_1).abs() < 1e-10);

            // set the cache to the wrong thing
            dist.$set_fn($change_value).unwrap();
            let _ = dist.ln_f(&x);

            // reset alpha and empty cache
            dist.$set_fn($start_value).unwrap();

            // this call should use the cache
            let ln_f_2 = dist.ln_f(&x);
            assert!((ln_f_1 - ln_f_2).abs() < 1e-10);
        }
    };
}

#[cfg(test)]
#[allow(dead_code)]
/// Assert Relative Eq for sequences
pub fn relative_eq<T, I>(
    left: I,
    right: I,
    epsilon: T::Epsilon,
    max_relative: T::Epsilon,
) -> bool
where
    T: RelativeEq,
    T::Epsilon: Copy,
    I: IntoIterator<Item = T>,
    <I as IntoIterator>::IntoIter: ExactSizeIterator,
{
    let a = left.into_iter();
    let b = right.into_iter();

    if a.len() != b.len() {
        return false;
    }

    a.zip(b)
        .all(|(a, b)| a.relative_eq(&b, epsilon, max_relative))
}

pub trait GewekeTestable<Fx, X> {
    fn prior_draw<R: rand::Rng>(&self, rng: &mut R) -> Fx;
    fn update_params<R: rand::Rng>(&self, data: &[X], rng: &mut R) -> Fx;
    fn geweke_stats(&self, fx: &Fx, xs: &[X]) -> BTreeMap<String, f64>;
}

pub struct GewekeTester<Pr, Fx, X>
where
    Pr: GewekeTestable<Fx, X>,
    Pr: Rv<Fx>,
    Fx: Rv<X>,
{
    pub pr: Pr,
    pub nx: usize,
    pub xs: Vec<X>,
    pub prior_chain_stats: BTreeMap<String, Vec<f64>>,
    pub posterior_chain_stats: BTreeMap<String, Vec<f64>>,
    _phantom: std::marker::PhantomData<Fx>,
}

fn append_stats(
    n: usize,
    src: &BTreeMap<String, f64>,
    sink: &mut BTreeMap<String, Vec<f64>>,
) {
    if sink.is_empty() {
        for k in src.keys() {
            sink.insert(k.clone(), Vec::with_capacity(n));
        }
    }

    for (k, v) in src.iter() {
        sink.get_mut(k)
            .map(|vals| vals.push(*v))
            .expect("failed to push")
    }
}

impl<Pr, Fx, X> GewekeTester<Pr, Fx, X>
where
    Pr: GewekeTestable<Fx, X>,
    Pr: Rv<Fx>,
    Fx: Rv<X>,
{
    pub fn new(pr: Pr, nx: usize) -> Self {
        GewekeTester {
            pr,
            nx,
            xs: Vec::new(),
            prior_chain_stats: BTreeMap::new(),
            posterior_chain_stats: BTreeMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn eval(&self, max_err: f64) -> Result<(), String> {
        let errors = self.errs();
        errors.iter().try_for_each(|(name, err)| {
            if *err > max_err {
                Err(format!(
                    "P-P Error {} ({}) exceeds max ({})",
                    name, err, max_err
                ))
            } else {
                Ok(())
            }
        })
    }

    /// Two-tailed test on prior and posterior stats
    pub fn errs(&self) -> Vec<(String, f64)> {
        use crate::dist::Empirical;
        let mut errors: Vec<(String, f64)> = Vec::new();
        for (stat_name, prior_stats) in self.prior_chain_stats.iter() {
            let post_stats = &self.posterior_chain_stats[stat_name];
            let emp_prior = Empirical::new(prior_stats.clone());
            let emp_post = Empirical::new(post_stats.clone());
            let err = emp_prior.err(&emp_post);
            errors.push((stat_name.clone(), err));
        }
        errors
    }

    pub fn run_chains<R: rand::Rng>(
        &mut self,
        n: usize,
        thinning: usize,
        rng: &mut R,
    ) {
        self.run_prior_chain(n, rng);
        self.run_posterior_chain(n, thinning, rng);
    }

    pub fn run_prior_chain<R: rand::Rng>(&mut self, n: usize, rng: &mut R) {
        (0..n).for_each(|_| {
            let fx = self.pr.prior_draw(rng);
            let xs: Vec<X> = fx.sample(self.nx, rng);
            let stats = self.pr.geweke_stats(&fx, &xs);

            append_stats(n, &stats, &mut self.prior_chain_stats)
        })
    }

    pub fn run_posterior_chain<R: rand::Rng>(
        &mut self,
        n: usize,
        thinning: usize,
        rng: &mut R,
    ) {
        let mut fx = self.pr.prior_draw(rng);
        let mut xs = fx.sample(self.nx, rng);
        (0..n).for_each(|_| {
            (0..thinning).for_each(|_| {
                fx = self.pr.update_params(&xs, rng);
                xs = fx.sample(self.nx, rng);
            });

            let stats = self.pr.geweke_stats(&fx, &xs);

            append_stats(n, &stats, &mut self.posterior_chain_stats)
        })
    }
}

#[macro_export]
macro_rules! gaussian_prior_geweke_testable {
    ($prior: ty, $fx: ty) => {
        impl GewekeTestable<Gaussian, f64> for $prior {
            fn prior_draw<R: rand::Rng>(&self, rng: &mut R) -> Gaussian {
                self.draw(rng)
            }

            fn update_params<R: rand::Rng>(
                &self,
                data: &[f64],
                rng: &mut R,
            ) -> Gaussian {
                let post = <$prior as ConjugatePrior<f64, $fx>>::posterior(
                    &self,
                    &DataOrSuffStat::from(data),
                );
                post.draw(rng)
            }

            fn geweke_stats(
                &self,
                fx: &Gaussian,
                xs: &[f64],
            ) -> BTreeMap<String, f64> {
                let mut stats: BTreeMap<String, f64> = BTreeMap::new();

                stats.insert(String::from("mu"), fx.mu());
                stats.insert(String::from("sigma"), fx.sigma());

                let mean = xs.iter().map(|&x| x).sum::<f64>() / xs.len() as f64;
                let mse = xs
                    .iter()
                    .map(|&x| {
                        let err = (x - mean);
                        err * err
                    })
                    .sum::<f64>()
                    / xs.len() as f64;

                stats.insert(String::from("x_mean"), mean);
                stats.insert(String::from("x_mse"), mse);

                stats
            }
        }
    };
}
