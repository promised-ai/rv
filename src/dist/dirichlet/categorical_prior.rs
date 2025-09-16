use rand::Rng;

use crate::data::{extract_stat_then, CategoricalDatum, CategoricalSuffStat};
use crate::dist::{Categorical, Dirichlet, SymmetricDirichlet};
use crate::misc::ln_gammafn;
use crate::prelude::CategoricalData;
use crate::traits::*;

impl HasDensity<Categorical> for SymmetricDirichlet {
    fn ln_f(&self, x: &Categorical) -> f64 {
        self.ln_f(&x.weights())
    }
}

impl Sampleable<Categorical> for SymmetricDirichlet {
    fn draw<R: Rng>(&self, mut rng: &mut R) -> Categorical {
        let weights: Vec<f64> = self.draw(&mut rng);
        Categorical::new(&weights).expect("Invalid draw")
    }
}

impl<X: CategoricalDatum> ConjugatePrior<X, Categorical>
    for SymmetricDirichlet
{
    type Posterior = Dirichlet;
    type MCache = f64;
    type PpCache = (Vec<f64>, f64);

    fn posterior(&self, x: &CategoricalData<X>) -> Self::Posterior {
        extract_stat_then(
            x,
            || CategoricalSuffStat::new(self.k()),
            |stat: CategoricalSuffStat| {
                let alphas: Vec<f64> =
                    stat.counts().iter().map(|&ct| self.alpha() + ct).collect();

                Dirichlet::new(alphas).unwrap()
            },
        )
    }

    #[inline]
    fn ln_m_cache(&self) -> Self::MCache {
        let sum_alpha = self.alpha() * self.k() as f64;
        let a = ln_gammafn(sum_alpha);
        let d = ln_gammafn(self.alpha()) * self.k() as f64;
        a - d
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::MCache,
        x: &CategoricalData<X>,
    ) -> f64 {
        let sum_alpha = self.alpha() * self.k() as f64;

        extract_stat_then(
            x,
            || CategoricalSuffStat::new(self.k()),
            |stat: CategoricalSuffStat| {
                // terms
                let b = ln_gammafn(sum_alpha + stat.n() as f64);
                let c = stat
                    .counts()
                    .iter()
                    .fold(0.0, |acc, &ct| acc + ln_gammafn(self.alpha() + ct));

                -b + c + cache
            },
        )
    }

    #[inline]
    fn ln_pp_cache(&self, x: &CategoricalData<X>) -> Self::PpCache {
        let post = self.posterior(x);
        let norm = post.alphas().iter().fold(0.0, |acc, &a| acc + a);
        (post.alphas, norm.ln())
    }

    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &X) -> f64 {
        let ix = y.into_usize();
        cache.0[ix].ln() - cache.1
    }
}

impl HasDensity<Categorical> for Dirichlet {
    fn ln_f(&self, x: &Categorical) -> f64 {
        self.ln_f(&x.weights())
    }
}

impl Sampleable<Categorical> for Dirichlet {
    fn draw<R: Rng>(&self, mut rng: &mut R) -> Categorical {
        let weights: Vec<f64> = self.draw(&mut rng);
        Categorical::new(&weights).expect("Invalid draw")
    }
}

impl<X: CategoricalDatum> ConjugatePrior<X, Categorical> for Dirichlet {
    type Posterior = Self;
    type MCache = (f64, f64);
    type PpCache = (Vec<f64>, f64);

    fn posterior(&self, x: &CategoricalData<X>) -> Self::Posterior {
        extract_stat_then(
            x,
            || CategoricalSuffStat::new(self.k()),
            |stat: CategoricalSuffStat| {
                let alphas: Vec<f64> = self
                    .alphas()
                    .iter()
                    .zip(stat.counts().iter())
                    .map(|(&a, &ct)| a + ct)
                    .collect();

                Dirichlet::new(alphas).unwrap()
            },
        )
    }

    #[inline]
    fn ln_m_cache(&self) -> Self::MCache {
        let sum_alpha = self.alphas().iter().fold(0.0, |acc, &a| acc + a);
        let a = ln_gammafn(sum_alpha);
        let d = self
            .alphas()
            .iter()
            .fold(0.0, |acc, &a| acc + ln_gammafn(a));
        (sum_alpha, a - d)
    }

    fn ln_m_with_cache(
        &self,
        cache: &Self::MCache,
        x: &CategoricalData<X>,
    ) -> f64 {
        let (sum_alpha, ln_norm) = cache;
        extract_stat_then(
            x,
            || CategoricalSuffStat::new(self.k()),
            |stat: CategoricalSuffStat| {
                // terms
                let b = ln_gammafn(sum_alpha + stat.n() as f64);
                let c = self
                    .alphas()
                    .iter()
                    .zip(stat.counts().iter())
                    .map(|(&a, &ct)| ln_gammafn(a + ct))
                    .sum::<f64>();

                -b + c + ln_norm
            },
        )
    }

    #[inline]
    fn ln_pp_cache(&self, x: &CategoricalData<X>) -> Self::PpCache {
        let post = self.posterior(x);
        let norm = post.alphas().iter().fold(0.0, |acc, &a| acc + a);
        (post.alphas, norm.ln())
    }

    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &X) -> f64 {
        let ix = y.into_usize();
        cache.0[ix].ln() - cache.1
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::data::DataOrSuffStat;
    use crate::test_conjugate_prior;

    const TOL: f64 = 1E-12;

    type CategoricalData<'a, X> = DataOrSuffStat<'a, X, Categorical>;

    mod dir {
        use super::*;

        test_conjugate_prior!(
            u8,
            Categorical,
            Dirichlet,
            Dirichlet::new(vec![1.0, 2.0]).unwrap(),
            n = 1_000_000
        );
    }

    mod symmetric {
        use super::*;

        test_conjugate_prior!(
            u8,
            Categorical,
            SymmetricDirichlet,
            SymmetricDirichlet::jeffreys(2).unwrap(),
            n = 1_000_000
        );

        #[test]
        fn marginal_likelihood_u8_1() {
            let alpha = 1.0;
            let k = 3;
            let xs: Vec<u8> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let csd = SymmetricDirichlet::new(alpha, k).unwrap();
            let m = csd.ln_m(&data);

            assert::close(-11.328_521_741_971_9, m, TOL);
        }

        #[test]
        fn marginal_likelihood_u8_2() {
            let alpha = 0.8;
            let k = 3;
            let mut xs: Vec<u8> = vec![0; 2];
            let mut xs1: Vec<u8> = vec![1; 7];
            let mut xs2: Vec<u8> = vec![2; 13];

            xs.append(&mut xs1);
            xs.append(&mut xs2);

            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let csd = SymmetricDirichlet::new(alpha, k).unwrap();
            let m = csd.ln_m(&data);

            assert::close(-22.437_719_300_855_2, m, TOL);
        }

        #[test]
        fn marginal_likelihood_u8_3() {
            let alpha = 4.5;
            let k = 3;
            let mut xs: Vec<u8> = vec![0; 2];
            let mut xs1: Vec<u8> = vec![1; 7];
            let mut xs2: Vec<u8> = vec![2; 13];

            xs.append(&mut xs1);
            xs.append(&mut xs2);

            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let csd = SymmetricDirichlet::new(alpha, k).unwrap();
            let m = csd.ln_m(&data);

            assert::close(-22.420_386_389_729_3, m, TOL);
        }

        #[test]
        fn symmetric_prior_draw_log_weights_should_all_be_negative() {
            let mut rng = rand::rng();
            let csd = SymmetricDirichlet::new(1.0, 4).unwrap();
            let ctgrl: Categorical = csd.draw(&mut rng);

            assert!(ctgrl.ln_weights().iter().all(|lw| *lw < 0.0));
        }

        #[test]
        fn symmetric_prior_draw_log_weights_should_be_unique() {
            let mut rng = rand::rng();
            let csd = SymmetricDirichlet::new(1.0, 4).unwrap();
            let ctgrl: Categorical = csd.draw(&mut rng);

            let ln_weights = ctgrl.ln_weights();

            assert!((ln_weights[0] - ln_weights[1]).abs() > TOL);
            assert!((ln_weights[1] - ln_weights[2]).abs() > TOL);
            assert!((ln_weights[2] - ln_weights[3]).abs() > TOL);
            assert!((ln_weights[0] - ln_weights[2]).abs() > TOL);
            assert!((ln_weights[0] - ln_weights[3]).abs() > TOL);
            assert!((ln_weights[1] - ln_weights[3]).abs() > TOL);
        }

        #[test]
        fn symmetric_posterior_draw_log_weights_should_all_be_negative() {
            let mut rng = rand::rng();

            let xs: Vec<u8> = vec![0, 1, 2, 1, 2, 3, 0, 1, 1];
            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let csd = SymmetricDirichlet::new(1.0, 4).unwrap();
            let cd = csd.posterior(&data);
            let ctgrl: Categorical = cd.draw(&mut rng);

            assert!(ctgrl.ln_weights().iter().all(|lw| *lw < 0.0));
        }

        #[test]
        fn symmetric_posterior_draw_log_weights_should_be_unique() {
            let mut rng = rand::rng();

            let xs: Vec<u8> = vec![0, 1, 2, 1, 2, 3, 0, 1, 1];
            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let csd = SymmetricDirichlet::new(1.0, 4).unwrap();
            let cd = csd.posterior(&data);
            let ctgrl: Categorical = cd.draw(&mut rng);

            let ln_weights = ctgrl.ln_weights();

            assert!((ln_weights[0] - ln_weights[1]).abs() > TOL);
            assert!((ln_weights[1] - ln_weights[2]).abs() > TOL);
            assert!((ln_weights[2] - ln_weights[3]).abs() > TOL);
            assert!((ln_weights[0] - ln_weights[2]).abs() > TOL);
            assert!((ln_weights[0] - ln_weights[3]).abs() > TOL);
            assert!((ln_weights[1] - ln_weights[3]).abs() > TOL);
        }

        #[test]
        fn predictive_probability_value_1() {
            let csd = SymmetricDirichlet::new(1.0, 3).unwrap();

            let xs: Vec<u8> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let lp = csd.ln_pp(&0, &data);
            assert::close(lp, -1.871_802_176_901_59, TOL);
        }

        #[test]
        fn predictive_probability_value_2() {
            let csd = SymmetricDirichlet::new(1.0, 3).unwrap();

            let xs: Vec<u8> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let lp = csd.ln_pp(&1, &data);
            assert::close(lp, -0.955_511_445_027_44, TOL);
        }

        #[test]
        fn predictive_probability_value_3() {
            let csd = SymmetricDirichlet::new(2.5, 3).unwrap();
            let xs: Vec<u8> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let lp = csd.ln_pp(&0, &data);
            assert::close(lp, -1.609_437_912_434_1, TOL);
        }

        #[test]
        fn predictive_probability_value_4() {
            let csd = SymmetricDirichlet::new(0.25, 3).unwrap();
            let xs: Vec<u8> = vec![
                0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2,
            ];
            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let lp = csd.ln_pp(&0, &data);
            assert::close(lp, -2.313_634_929_180_62, TOL);
        }

        #[test]
        fn csd_loglike_value_1() {
            let csd = SymmetricDirichlet::new(0.5, 3).unwrap();
            let cat = Categorical::new(&[0.2, 0.3, 0.5]).unwrap();
            let lf = csd.ln_f(&cat);
            assert::close(lf, -0.084_598_117_749_354_22, TOL);
        }
    }
}
