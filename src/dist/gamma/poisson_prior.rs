use std::f64::EPSILON;

use rand::Rng;

use crate::data::PoissonSuffStat;
use crate::dist::poisson::PoissonError;
use crate::dist::{Gamma, Poisson};
use crate::misc::ln_binom;
use crate::suffstat_traits::*;
use crate::traits::*;

impl HasDensity<Poisson> for Gamma {
    fn ln_f(&self, x: &Poisson) -> f64 {
        match x.mean() {
            Some(mean) => self.ln_f(&mean),
            None => std::f64::NEG_INFINITY,
        }
    }
}

impl Sampleable<Poisson> for Gamma {
    fn draw<R: Rng>(&self, mut rng: &mut R) -> Poisson {
        let mean: f64 = self.draw(&mut rng);
        match Poisson::new(mean) {
            Ok(pois) => pois,
            Err(PoissonError::RateTooLow { .. }) => {
                Poisson::new_unchecked(EPSILON)
            }
            Err(err) => panic!("Failed to draw Possion: {}", err),
        }
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
            type MCache = f64;
            type PpCache = (f64, f64, f64);

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
                };

                let a = self.shape() + sum;
                let b = self.rate() + (n as f64);
                Self::new(a, b).expect("Invalid posterior parameters")
            }

            #[inline]
            fn ln_m_cache(&self) -> Self::MCache {
                let z0 = self
                    .shape()
                    .mul_add(-self.ln_rate(), self.ln_gamma_shape());
                z0
            }

            fn ln_m_with_cache(
                &self,
                cache: &Self::MCache,
                x: &DataOrSuffStat<$kind, Poisson>,
            ) -> f64 {
                let stat: PoissonSuffStat = match x {
                    DataOrSuffStat::Data(ref xs) => {
                        let mut stat = PoissonSuffStat::new();
                        xs.iter().for_each(|x| stat.observe(x));
                        stat
                    }
                    DataOrSuffStat::SuffStat(ref stat) => (*stat).clone(),
                };

                let data_or_suff: DataOrSuffStat<$kind, Poisson> =
                    DataOrSuffStat::SuffStat(&stat);
                let post = self.posterior(&data_or_suff);

                let zn = post
                    .shape()
                    .mul_add(-post.ln_rate(), post.ln_gamma_shape());

                zn - cache - stat.sum_ln_fact()
            }

            #[inline]
            fn ln_pp_cache(
                &self,
                x: &DataOrSuffStat<$kind, Poisson>,
            ) -> Self::PpCache {
                let post = self.posterior(x);
                let r = post.shape();
                let p = 1.0 / (1.0 + post.rate());
                (r, p, p.ln())
            }

            fn ln_pp_with_cache(
                &self,
                cache: &Self::PpCache,
                y: &$kind,
            ) -> f64 {
                let (r, p, ln_p) = cache;
                let k = f64::from(*y);
                let bnp = ln_binom(k + r - 1.0, k);
                bnp + (1.0 - p).ln() * r + k * ln_p
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
        let new_vec = Vec::new();
        let data: DataOrSuffStat<u8, Poisson> = DataOrSuffStat::from(&new_vec);
        assert::close(dist.ln_m(&data), 0.0, TOL);
    }

    #[test]
    fn ln_m_data() {
        let dist = Gamma::new(1.0, 1.0).unwrap();
        let inputs: [u8; 5] = [0, 1, 2, 3, 4];
        let expected: [f64; 5] = [
            -std::f64::consts::LN_2,
            -2.197_224_577_336_219_6,
            -4.446_565_155_811_452,
            -7.171_720_824_816_601,
            -10.267_902_068_569_033,
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
            -std::f64::consts::LN_2,
            -1.386_294_361_119_890_6,
            -2.079_441_541_679_835_7,
            -2.772_588_722_239_781,
            -3.465_735_902_799_726_5,
        ];

        for i in 0..inputs.len() {
            assert::close(
                dist.ln_pp(&inputs[i], &DataOrSuffStat::from(&vec![])),
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
            -3.132_409_571_626_673,
            -2.033_797_282_958_563_5,
            -1.600_933_200_662_284_5,
            -1.546_865_979_392_009,
            -1.754_505_344_170_253_6,
        ];

        for (i, e) in inputs.iter().zip(expected.iter()) {
            assert::close(dist.ln_pp(i, &doss), *e, TOL);
        }
    }

    #[test]
    fn cannot_draw_zero_rate() {
        let mut rng = rand::thread_rng();
        let dist = Gamma::new(1.0, 1e-10).unwrap();
        let stream =
            <Gamma as Sampleable<Poisson>>::sample_stream(&dist, &mut rng);
        assert!(stream.take(10_000).all(|pois| pois.rate() > 0.0));
    }
}
