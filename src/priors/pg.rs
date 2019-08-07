use rand::Rng;

use crate::data::{DataOrSuffStat, PoissonSuffStat};
use crate::dist::{Gamma, Poisson};
use crate::misc::ln_binom;
use crate::traits::*;
use special::Gamma as SGamma;

impl Rv<Poisson> for Gamma {
    fn ln_f(&self, x: &Poisson) -> f64 {
        match x.mean() {
            Some(mean) => self.ln_f(&mean),
            None => std::f64::NEG_INFINITY,
        }
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Poisson {
        let mean: f64 = self.draw(&mut rng);
        Poisson::new(mean).expect("Failed to draw a valid mean")
    }
}

impl Support<Poisson> for Gamma {
    fn supports(&self, x: &Poisson) -> bool {
        match x.mean() {
            Some(mean) => mean > 0.0 && !mean.is_infinite(),
            None => false,
        }
    }
}

impl ContinuousDistr<Poisson> for Gamma {}

macro_rules! impl_traits {
    ($kind: ty) => {
        impl ConjugatePrior<$kind, Poisson> for Gamma {
            type Posterior = Self;

            fn posterior(&self, x: &DataOrSuffStat<$kind, Poisson>) -> Self {
                let (n, sum) = match x {
                    DataOrSuffStat::Data(ref xs) => {
                        let mut stat = PoissonSuffStat::new();
                        xs.iter().for_each(|x| stat.observe(x));
                        (stat.n(), stat.sum())
                    }
                    DataOrSuffStat::SuffStat(ref stat) => {
                        (stat.n(), stat.sum())
                    }
                    DataOrSuffStat::None => (0, 0.0),
                };

                let a = self.shape() + sum;
                let b = self.rate() + (n as f64);
                Self::new(a, b).expect("Invalid posterior parameters")
            }

            fn ln_m(&self, x: &DataOrSuffStat<$kind, Poisson>) -> f64 {
                let stat: PoissonSuffStat = match x {
                    DataOrSuffStat::Data(ref xs) => {
                        let mut stat = PoissonSuffStat::new();
                        xs.iter().for_each(|x| stat.observe(x));
                        stat
                    }
                    DataOrSuffStat::SuffStat(ref stat) => (*stat).clone(),
                    DataOrSuffStat::None => PoissonSuffStat::new(),
                };

                let data_or_suff: DataOrSuffStat<$kind, Poisson> =
                    DataOrSuffStat::SuffStat(&stat);
                let post = self.posterior(&data_or_suff);

                let z0 =
                    self.shape().ln_gamma().0 - self.shape() * self.rate().ln();
                let zn =
                    post.shape().ln_gamma().0 - post.shape() * post.rate().ln();

                zn - z0 - stat.sum_log_fact()
            }

            fn ln_pp(
                &self,
                y: &$kind,
                x: &DataOrSuffStat<$kind, Poisson>,
            ) -> f64 {
                let post = self.posterior(x);
                let r = post.shape();
                let p = 1.0 / (1.0 + post.rate());

                let k = f64::from(*y);
                let bnp = ln_binom(k + r - 1.0, k);
                bnp + (1.0 - p).ln() * r + k * p.ln()
            }
        }
    };
}

impl_traits!(u8);
impl_traits!(u16);
impl_traits!(u32);

#[cfg(test)]
mod tests {
    use super::*;
    const TOL: f64 = 1E-12;

    #[test]
    fn posterior_from_data() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5];
        let xs = DataOrSuffStat::Data::<u8, Poisson>(&data);
        let posterior = Gamma::new(1.0, 1.0).unwrap().posterior(&xs);

        assert::close(posterior.shape(), 16.0, TOL);
        assert::close(posterior.rate(), 6.0, TOL);
    }

    #[test]
    fn ln_m_no_data() {
        let dist = Gamma::new(1.0, 1.0).unwrap();
        let data: DataOrSuffStat<u8, Poisson> = DataOrSuffStat::None;
        assert::close(dist.ln_m(&data), 0.0, TOL);
    }

    #[test]
    fn ln_m_data() {
        let dist = Gamma::new(1.0, 1.0).unwrap();
        let inputs: [u8; 5] = [0, 1, 2, 3, 4];
        let expected: [f64; 5] = [
            -0.6931471805599453,
            -2.1972245773362196,
            -4.446565155811452,
            -7.171720824816601,
            -10.267902068569033,
        ];

        // Then test on the sequence of inputs
        let suff_stats: Vec<PoissonSuffStat> = inputs
            .iter()
            .scan(PoissonSuffStat::new(), |acc, x| {
                acc.observe(x);
                Some(acc.clone())
            })
            .collect();

        suff_stats
            .iter()
            .zip(expected.iter())
            .for_each(|(ss, exp)| {
                let data: DataOrSuffStat<u8, Poisson> =
                    DataOrSuffStat::SuffStat(ss);
                let r = dist.ln_m(&data);
                assert::close(r, *exp, TOL);
            });
    }

    #[test]
    fn ln_pp_no_data() {
        let dist = Gamma::new(1.0, 1.0).unwrap();
        let inputs: [u8; 5] = [0, 1, 2, 3, 4];
        let expected: [f64; 5] = [
            -0.6931471805599453,
            -1.3862943611198906,
            -2.0794415416798357,
            -2.772588722239781,
            -3.4657359027997265,
        ];

        for i in 0..inputs.len() {
            assert::close(
                dist.ln_pp(&inputs[i], &DataOrSuffStat::None),
                expected[i],
                TOL,
            )
        }
    }

    #[test]
    fn ln_pp_data() {
        let data: [u8; 10] = [5, 7, 8, 1, 0, 2, 2, 5, 1, 4];
        let mut suff_stat = PoissonSuffStat::new();
        data.iter().for_each(|d| suff_stat.observe(d));

        let doss = DataOrSuffStat::SuffStat::<u8, Poisson>(&suff_stat);

        let dist = Gamma::new(1.0, 1.0).unwrap();
        let inputs: [u8; 5] = [0, 1, 2, 3, 4];
        let expected: [f64; 5] = [
            -3.132409571626673,
            -2.0337972829585635,
            -1.6009332006622845,
            -1.546865979392009,
            -1.7545053441702536,
        ];

        for (i, e) in inputs.iter().zip(expected.iter()) {
            assert::close(dist.ln_pp(i, &doss), *e, TOL);
        }
    }
}
