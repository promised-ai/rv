use rand::Rng;
use special::Gamma as SGamma;

use crate::data::{CategoricalDatum, CategoricalSuffStat, DataOrSuffStat};
use crate::dist::{Categorical, Dirichlet, SymmetricDirichlet};
use crate::prelude::CategoricalData;
use crate::traits::*;

impl Rv<Categorical> for SymmetricDirichlet {
    fn ln_f(&self, x: &Categorical) -> f64 {
        self.ln_f(&x.weights())
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Categorical {
        let weights: Vec<f64> = self.draw(&mut rng);
        Categorical::new(&weights).expect("Invalid draw")
    }
}

impl<'a, X: CategoricalDatum> ConjugatePrior<'a, X, Categorical>
    for SymmetricDirichlet
{
    type Posterior = Dirichlet;

    fn posterior<S>(&self, x: &S) -> Self::Posterior
    where
        S: Into<&'a CategoricalSuffStat>,
    {
        let stat: CategoricalSuffStat = x.into();
        let alphas: Vec<f64> =
            stat.counts().iter().map(|&ct| self.alpha() + ct).collect();

        Dirichlet::new(alphas).unwrap()
    }

    fn ln_m<S>(&self, x: &S) -> f64
    where
        S: Into<&'a CategoricalSuffStat>,
    {
        let sum_alpha = self.alpha() * self.k() as f64;

        let stat: CategoricalSuffStat = x.into();
        let a = sum_alpha.ln_gamma().0;
        let b = (sum_alpha + stat.n() as f64).ln_gamma().0;
        let c = stat
            .counts()
            .iter()
            .fold(0.0, |acc, &ct| acc + (self.alpha() + ct).ln_gamma().0);
        let d = self.alpha().ln_gamma().0 * self.k() as f64;

        a - b + c - d
    }

    fn ln_pp<S>(&self, y: &X, x: &S) -> f64
    where
        S: Into<&'a CategoricalSuffStat>,
    {
        let post = <Self as ConjugatePrior<X, Categorical>>::posterior(self, x);
        let norm = post.alphas().iter().fold(0.0, |acc, &a| acc + a);
        let ix = y.into_usize();
        post.alphas()[ix].ln() - norm.ln()
    }
}

impl Rv<Categorical> for Dirichlet {
    fn ln_f(&self, x: &Categorical) -> f64 {
        self.ln_f(&x.weights())
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Categorical {
        let weights: Vec<f64> = self.draw(&mut rng);
        Categorical::new(&weights).expect("Invalid draw")
    }
}

impl<'a, X: CategoricalDatum> ConjugatePrior<'a, X, Categorical> for Dirichlet {
    type Posterior = Self;
    fn posterior<S>(&self, x: &S) -> Self::Posterior
    where
        S: Into<&'a CategoricalSuffStat>,
    {
        let stat: CategoricalSuffStat = x.into();
        let alphas: Vec<f64> = self
            .alphas()
            .iter()
            .zip(stat.counts().iter())
            .map(|(&a, &ct)| a + ct)
            .collect();

        Dirichlet::new(alphas).unwrap()
    }

    fn ln_m<S>(&self, x: &S) -> f64
    where
        S: Into<&'a CategoricalSuffStat>,
    {
        let sum_alpha = self.alphas().iter().fold(0.0, |acc, &a| acc + a);
        let stat: CategoricalSuffStat = x.into();

        // terms
        let a = sum_alpha.ln_gamma().0;
        let b = (sum_alpha + stat.n() as f64).ln_gamma().0;
        let c = self
            .alphas()
            .iter()
            .zip(stat.counts().iter())
            .fold(0.0, |acc, (&a, &ct)| acc + (a + ct).ln_gamma().0);
        let d = self
            .alphas()
            .iter()
            .fold(0.0, |acc, &a| acc + a.ln_gamma().0);

        a - b + c - d
    }

    fn ln_pp<S>(&self, y: &X, x: &S) -> f64
    where
        S: Into<&'a CategoricalSuffStat>,
    {
        let post = <Self as ConjugatePrior<X, Categorical>>::posterior(self, x);
        let norm = post.alphas().iter().fold(0.0, |acc, &a| acc + a);
        let ix: usize = y.into_usize();
        post.alphas()[ix].ln() - norm.ln()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const TOL: f64 = 1E-12;

    type CategoricalData<'a, X> = DataOrSuffStat<'a, X, Categorical>;

    mod symmetric {
        use super::*;

        #[test]
        fn marginal_likelihood_u8_1() {
            let alpha = 1.0;
            let k = 3;
            let xs: Vec<u8> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let csd = SymmetricDirichlet::new(alpha, k).unwrap();
            let m = csd.ln_m(&data);

            assert::close(-11.3285217419719, m, TOL);
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

            assert::close(-22.4377193008552, m, TOL);
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

            assert::close(-22.4203863897293, m, TOL);
        }

        #[test]
        fn symmetric_prior_draw_log_weights_should_all_be_negative() {
            let mut rng = rand::thread_rng();
            let csd = SymmetricDirichlet::new(1.0, 4).unwrap();
            let ctgrl: Categorical = csd.draw(&mut rng);

            assert!(ctgrl.ln_weights().iter().all(|lw| *lw < 0.0));
        }

        #[test]
        fn symmetric_prior_draw_log_weights_should_be_unique() {
            let mut rng = rand::thread_rng();
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
            let mut rng = rand::thread_rng();

            let xs: Vec<u8> = vec![0, 1, 2, 1, 2, 3, 0, 1, 1];
            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let csd = SymmetricDirichlet::new(1.0, 4).unwrap();
            let cd = csd.posterior(&data);
            let ctgrl: Categorical = cd.draw(&mut rng);

            assert!(ctgrl.ln_weights().iter().all(|lw| *lw < 0.0));
        }

        #[test]
        fn symmetric_posterior_draw_log_weights_should_be_unique() {
            let mut rng = rand::thread_rng();

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
            assert::close(lp, -1.87180217690159, TOL);
        }

        #[test]
        fn predictive_probability_value_2() {
            let csd = SymmetricDirichlet::new(1.0, 3).unwrap();

            let xs: Vec<u8> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let lp = csd.ln_pp(&1, &data);
            assert::close(lp, -0.95551144502744, TOL);
        }

        #[test]
        fn predictive_probability_value_3() {
            let csd = SymmetricDirichlet::new(2.5, 3).unwrap();
            let xs: Vec<u8> = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let data: CategoricalData<u8> = DataOrSuffStat::Data(&xs);

            let lp = csd.ln_pp(&0, &data);
            assert::close(lp, -1.6094379124341, TOL);
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
            assert::close(lp, -2.31363492918062, TOL);
        }

        #[test]
        fn csd_loglike_value_1() {
            let csd = SymmetricDirichlet::new(0.5, 3).unwrap();
            let cat = Categorical::new(&vec![0.2, 0.3, 0.5]).unwrap();
            let lf = csd.ln_f(&cat);
            assert::close(lf, -0.084598117749354218, TOL);
        }
    }
}
