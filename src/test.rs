#[cfg(test)]
use approx::RelativeEq;

use crate::traits::Rv;
use std::collections::BTreeMap;

// tests that Clone, Debug, and PartialEq are implemented for a distribution
// Tests that partial eq is not sensitive to OnceCell initialization, which
// often happens in ln_f is called
#[macro_export]
macro_rules! test_basic_impls {
    ($X:ty, $Fx:ty) => {
        test_basic_impls!($X, $Fx, <$Fx>::default());
    };
    ($X:ty, $Fx:ty, $fx:expr) => {
        mod rv_impl {
            use super::*;

            #[test]
            fn should_impl_debug_clone_and_partialeq() {
                let mut rng = rand::thread_rng();
                // make the expression a thing. If we don't do this, calling $fx
                // reconstructs the distribution which means we don't do caching
                let fx = $fx;
                let x: $X = fx.draw(&mut rng);

                // clone a copy of fn before any computation of cached values is
                // done
                let fx2 = fx.clone();
                assert_eq!($fx, fx2);

                // Computing ln_f normally initializes all cached values
                let y1 = fx.ln_f(&x);
                let y2 = fx.ln_f(&x);
                assert!((y1 - y2).abs() < f64::EPSILON);

                // check the fx == fx2 despite fx having its cached values
                // initialized
                assert_eq!(fx2, fx);

                // Make sure Debug is implemented for fx
                let _s1 = format!("{:?}", fx);
            }

            #[test]
            fn should_impl_parameterized() {
                let mut rng = rand::thread_rng();

                let fx_1 = $fx;
                let params = fx_1.emit_params();
                let fx_2 = <$Fx>::from_params(params);

                for _ in 0..100 {
                    let x: $X = fx_1.draw(&mut rng);

                    let ln_f_1 = fx_1.ln_f(&x);
                    let ln_f_2 = fx_2.ln_f(&x);

                    assert::close(ln_f_1, ln_f_2, 1e-14);
                }
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

#[macro_export]
macro_rules! test_conjugate_prior {
    ($X: ty, $Fx: ty, $Pr: ident, $prior: expr) => {
        test_conjugate_prior!(
            $X,
            $Fx,
            $Pr,
            $prior,
            mctol = 1e-3,
            n = 1_000_000
        );
    };
    ($X: ty, $Fx: ty, $Pr: ident, $prior: expr, n=$n: expr) => {
        test_conjugate_prior!($X, $Fx, $Pr, $prior, mctol = 1e-3, n = $n);
    };
    ($X: ty, $Fx: ty, $Pr: ident, $prior: expr, mctol=$tol: expr) => {
        test_conjugate_prior!(
            $X,
            $Fx,
            $Pr,
            $prior,
            mctol = $tol,
            n = 1_000_000
        );
    };
    ($X: ty, $Fx: ty, $Pr: ident, $prior: expr, mctol=$tol: expr, n=$n: expr) => {
        mod conjugate_prior {
            use super::*;

            fn random_xs(
                fx: &$Fx,
                n: usize,
                mut rng: &mut impl rand::Rng,
            ) -> <$Fx as $crate::traits::HasSuffStat<$X>>::Stat {
                let mut stat =
                    <$Fx as $crate::traits::HasSuffStat<$X>>::empty_suffstat(
                        &fx,
                    );
                let xs: Vec<$X> = fx.sample(n, &mut rng);
                stat.observe_many(&xs);
                stat
            }

            #[test]
            fn ln_p_is_ratio_of_ln_m() {
                // test that p(y|x) = p(y, x) / p(x)
                // If this doesn't work, one of two things could be wrong:
                // 1. prior.ln_m is wrong
                // 2. prior.ln_pp is wrong
                let mut rng = rand::thread_rng();

                let pr = $prior;
                let fx: $Fx = pr.draw(&mut rng);

                let mut stat = random_xs(&fx, 3, &mut rng);

                let y: $X = fx.draw(&mut rng);

                let ln_pp = <$Pr as ConjugatePrior<$X, $Fx>>::ln_pp(
                    &pr,
                    &y,
                    &DataOrSuffStat::SuffStat(&stat),
                );
                let ln_m_lower = <$Pr as ConjugatePrior<$X, $Fx>>::ln_m(
                    &pr,
                    &DataOrSuffStat::SuffStat(&stat),
                );

                stat.observe(&y);

                let ln_m_upper = <$Pr as ConjugatePrior<$X, $Fx>>::ln_m(
                    &pr,
                    &DataOrSuffStat::SuffStat(&stat),
                );

                assert::close(ln_pp, ln_m_upper - ln_m_lower, 1e-12);
            }

            #[test]
            fn bayes_law() {
                // test that p(θ|x) == p(x|θ)p(θ)/p(x)
                // If this doesn't work, one of the following is wrong
                // 1. prior.posterior.ln_f(fx)
                // 2. fx.ln_f(x)
                // 3. prior.ln_f(fx)
                // 4. prior.ln_m(x)
                let mut rng = rand::thread_rng();

                let pr = $prior;
                let fx: $Fx = pr.draw(&mut rng);
                let stat = random_xs(&fx, 3, &mut rng);

                let ln_like =
                    <$Fx as $crate::traits::HasSuffStat<$X>>::ln_f_stat(
                        &fx, &stat,
                    );
                let ln_prior = pr.ln_f(&fx);
                let ln_m = <$Pr as ConjugatePrior<$X, $Fx>>::ln_m(
                    &pr,
                    &DataOrSuffStat::SuffStat(&stat),
                );

                let posterior = <$Pr as ConjugatePrior<$X, $Fx>>::posterior(
                    &pr,
                    &DataOrSuffStat::SuffStat(&stat),
                );
                let ln_post = posterior.ln_f(&fx);

                eprintln!("bayes_law stat: {:?}", stat);
                eprintln!("bayes_law prior: {pr}");
                eprintln!("bayes_law fx: {fx}");
                eprintln!("bayes_law ln_like: {ln_like}");
                eprintln!("bayes_law ln_prior: {ln_prior}");
                eprintln!("bayes_law ln_m: {ln_m}");
                eprintln!("bayes_law ln_post: {ln_post}");

                assert::close(ln_post, ln_like + ln_prior - ln_m, 1e-10);
            }

            #[test]
            fn monte_carlo_ln_m() {
                // tests that the Monte Carlo estimate of the evidence converges
                // to m(x)
                // If this doesn't work one of three things could be wrong:
                // 1. prior.draw (from sample_stream) is wrong
                // 2. fx.ln_f_stat is wrong
                // 3. prior.m is wrong
                let n_tries = 5;
                let mut rng = rand::thread_rng();

                let pr = $prior;

                let stat = random_xs(&pr.draw(&mut rng), 3, &mut rng);

                let m = <$Pr as ConjugatePrior<$X, $Fx>>::m(
                    &pr,
                    &DataOrSuffStat::SuffStat(&stat),
                );

                let mut min_err = f64::INFINITY;

                for _ in 0..n_tries {
                    let stream =
                        <$Pr as $crate::traits::Sampleable<$Fx>>::sample_stream(
                            &pr, &mut rng,
                        );
                    let est = stream
                        .take($n)
                        .map(|fx| {
                            <$Fx as $crate::traits::HasSuffStat<$X>>::ln_f_stat(
                                &fx, &stat,
                            )
                            .exp()
                        })
                        .sum::<f64>()
                        / ($n as f64);

                    let err = (est - m).abs();
                    let close_enough = err < $tol;

                    if err < min_err {
                        min_err = err;
                    }

                    if close_enough {
                        return;
                    }
                }
                panic!(
                    "MC estimate of M failed under {pr}. Min err: {min_err}"
                );
            }
        }
    };
}

use crate::prelude::ChiSquared;
use crate::traits::Cdf;
/// # Arguments
/// * `samples` - The data samples to test
/// * `density_fn` - A function that returns the unnormalized density at a given point
/// * `num_bins` - Number of constant-width bins to use
/// * `normalized` - Whether the given density is normalized
///
/// # Returns
/// * Result<f64, Box<dyn std::error::Error>>
pub fn density_histogram_test<F>(
    samples: &[f64],
    num_bins: usize,
    density_fn: F,
    normalized: bool,
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: Fn(f64) -> f64,
{
    if samples.is_empty() {
        return Err("Sample set is empty".into());
    }
    if num_bins < 2 {
        return Err("Need at least 2 bins for chi-square test".into());
    }

    // Find min and max of the samples
    let min_val = samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    // Create histogram with constant width bins
    let mut hist: Vec<usize> = vec![0; num_bins];

    // Fill histogram with samples
    for &sample in samples {
        let u = (sample - min_val) / range; // 0 ≤ u ≤ 1
        let histix = (u * num_bins as f64) as usize;
        hist[histix] += 1;
    }

    // Calculate bin width for Simpson's rule
    let bin_width = (max_val - min_val) / num_bins as f64;

    // Calculate expected frequencies using Simpson's rule to integrate the density function
    let mut expected_counts = Vec::with_capacity(num_bins);
    let mut total_integral = 0.0;

    for bin_ix in 0..num_bins {
        let bin_start = min_val + bin_ix as f64 * bin_width;
        let bin_mid = bin_start + bin_width / 2.0;
        let bin_end = bin_start + bin_width;

        // Apply Simpson's rule for each bin
        let integral = (bin_width / 6.0)
            * (density_fn(bin_start)
                + 4.0 * density_fn(bin_mid)
                + density_fn(bin_end));

        expected_counts.push(integral);
        total_integral += integral;
    }

    // Scale the expected counts so the total matches the number of samples
    let scale_factor = if normalized {
        samples.len() as f64
    } else {
        samples.len() as f64 / total_integral
    };
    for expected in &mut expected_counts {
        *expected *= scale_factor;
    }

    // Calculate chi-square statistic
    let mut test_stat = 0.0;
    let mut valid_bins = 0;

    for bin_ix in 0..num_bins {
        let observed = hist[bin_ix];
        let expected = expected_counts[bin_ix];

        // Skip bins with expected count less than 5 (chi-square assumption)
        if expected >= 5.0 {
            test_stat += (observed as f64 - expected).powi(2) / expected;
            valid_bins += 1;
        }
    }

    // Degrees of freedom = number of bins - 1
    let df = valid_bins - 1;

    if df <= 0 {
        return Err("Not enough valid bins (with expected count >= 5) for chi-square test".into());
    }

    // Calculate p-value
    let chi_dist = ChiSquared::new(df as f64)?;
    let p_value = 1.0 - chi_dist.cdf(&test_stat);

    Ok(p_value)
}
