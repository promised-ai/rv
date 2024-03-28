use crate::data::DataOrSuffStat;
use crate::dist::{Beta, Dirichlet};
use crate::traits::{ConjugatePrior, Rv, SuffStat};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use super::sbd::Sbd;
use super::sbd_stat::SbdSuffStat;

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct Sb {
    alpha: f64,
    seed: Option<u64>,
}

impl Sb {
    pub fn new(alpha: f64, seed: Option<u64>) -> Self {
        Self { alpha, seed }
    }

    pub fn set_alpha_unchecked(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug)]
pub struct SbPosterior {
    alpha: f64,
    dir: Dirichlet,
}

impl Rv<Sbd> for Sb {
    fn ln_f(&self, x: &Sbd) -> f64 {
        let k = x.k();
        if k == 0 {
            return 0.0;
        }

        let weights = x.observed_weights();
        let beta = Beta::new_unchecked(1.0, self.alpha);
        let mut total_mass = 1.0;
        weights
            .iter()
            .take(k)
            .map(|w| {
                let x = w / total_mass;
                let ln_f = beta.ln_f(&x);
                total_mass -= w;
                ln_f
            })
            .sum::<f64>()
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Sbd {
        let seed: u64 = rng.gen();
        Sbd::new(self.alpha, Some(seed)).unwrap()
    }
}

impl Rv<Sbd> for SbPosterior {
    fn ln_f(&self, x: &Sbd) -> f64 {
        let weights = x.observed_weights();
        let k = self.dir.k();
        let mut total_mass = 1.0;
        self.dir
            .alphas
            .iter()
            .take(k - 1)
            .zip(weights.iter())
            .map(|(&alpha, &w)| {
                let beta = Beta::new_unchecked(1.0, alpha);
                let ln_f = beta.ln_f(&(w / total_mass));
                // let g = crate::dist::Gamma::new_unchecked(alpha, 1.0);
                // let ln_f = g.ln_f(&(1.0 - (w / total_mass)));
                total_mass -= w;
                ln_f
            })
            .sum::<f64>()
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Sbd {
        let weights: Vec<f64> = self.dir.draw(rng);
        Sbd::from_weights(&weights, self.alpha, Some(rng.gen())).unwrap()
    }
}

fn sbpost_from_stat(alpha: f64, stat: &SbdSuffStat) -> SbPosterior {
    let dir = {
        let alphas: Vec<f64> = stat
            .counts()
            .iter()
            .map(|&ct| ct as f64 + alpha)
            .chain(std::iter::once(alpha))
            .collect();

        Dirichlet::new(alphas).unwrap()
    };

    SbPosterior { alpha, dir }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct SbCache {
    ln_weights: Vec<f64>,
    ln_f_new: f64,
}

impl SbCache {
    pub fn k(&self) -> usize {
        self.ln_weights.len()
    }
}

fn gks(stat: &SbdSuffStat) -> Vec<f64> {
    let n = stat.n();
    let mut acc = n;

    stat.counts()
        .iter()
        .map(|ct| {
            let g = acc;
            acc -= ct;
            g as f64
        })
        .collect()
}

fn sbm_from_stat(stat: &SbdSuffStat, alpha: f64) -> f64 {
    let gk = gks(stat);
    let n = stat.n() as f64;
    let ln_alpha = alpha.ln();

    assert_eq!(gk.len(), stat.counts().len());
    // assert_eq!(stat.n(), stat.counts().iter().sum::<usize>());

    // TODO: simplify
    // a = loggamma(alpha) - loggamma(alpha + n)
    let term_a = Gamma::ln_gamma(alpha).0 - Gamma::ln_gamma(alpha + n).0;
    // b = sum(log_alpha - np.log(gk + alpha) for gk in gks)
    let term_b = gk.iter().map(|g| ln_alpha - (g + alpha).ln()).sum::<f64>();
    // c = sum(loggamma(ct + 1) for ct in counts)
    let term_c = stat
        .counts()
        .iter()
        .map(|&ct| Gamma::ln_gamma(ct as f64 + 1.0).0)
        .sum::<f64>();

    term_a + term_b + term_c
}

fn sbpp_cache(x: &SbdSuffStat, alpha: f64) -> SbCache {
    let gk = gks(x);
    let n = x.n() as f64;
    let k = gk.len();

    let mut ln_weights = Vec::with_capacity(k);

    let term_a = -(n + alpha).ln();

    for y in 0..k {
        let term_b = ((x.counts()[y] + 1) as f64).ln();
        let term_c = gk
            .iter()
            .take(y + 1)
            .map(|&g| ((g + alpha) / (g + alpha + 1.0)).ln())
            .sum::<f64>();

        ln_weights.push(term_a + term_b + term_c)
    }

    let ln_f_new = {
        let term_b = gk
            .iter()
            .map(|&g| ((g + alpha) / (g + alpha + 1.0)).ln())
            .sum::<f64>();

        term_a + term_b + alpha.ln() - (1.0 + alpha).ln()
    };

    SbCache {
        ln_weights,
        ln_f_new,
    }
}

impl ConjugatePrior<usize, Sbd> for Sb {
    type Posterior = SbPosterior;
    type LnMCache = ();
    type LnPpCache = SbCache;

    fn ln_m_cache(&self) -> Self::LnMCache {}

    fn ln_pp_cache(&self, x: &DataOrSuffStat<usize, Sbd>) -> Self::LnPpCache {
        match x {
            DataOrSuffStat::Data(xs) => {
                let mut stat = SbdSuffStat::new();
                stat.observe_many(xs);
                sbpp_cache(&stat, self.alpha)
            }
            DataOrSuffStat::SuffStat(ref stat) => sbpp_cache(stat, self.alpha),
            DataOrSuffStat::None => SbCache {
                ln_weights: Vec::new(),
                ln_f_new: 0.0,
            },
        }
    }

    fn posterior(&self, x: &DataOrSuffStat<usize, Sbd>) -> Self::Posterior {
        match x {
            DataOrSuffStat::Data(xs) => {
                let mut stat = SbdSuffStat::new();
                stat.observe_many(xs);
                sbpost_from_stat(self.alpha, &stat)
            }
            DataOrSuffStat::SuffStat(stat) => {
                sbpost_from_stat(self.alpha, stat)
            }
            DataOrSuffStat::None => panic!("Need data for posterior"),
        }
    }

    fn ln_m_with_cache(
        &self,
        _cache: &Self::LnMCache,
        x: &DataOrSuffStat<usize, Sbd>,
    ) -> f64 {
        match x {
            DataOrSuffStat::Data(xs) => {
                let mut stat = SbdSuffStat::new();
                stat.observe_many(xs);
                sbm_from_stat(&stat, self.alpha)
            }
            DataOrSuffStat::SuffStat(stat) => sbm_from_stat(&stat, self.alpha),
            DataOrSuffStat::None => panic!("Need data for ln_m"),
        }
    }

    fn ln_pp_with_cache(&self, cache: &Self::LnPpCache, y: &usize) -> f64 {
        if *y >= cache.k() {
            cache.ln_f_new
        } else {
            cache.ln_weights[*y]
        }
    }
}

use special::Gamma;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::CategoricalSuffStat;
    use crate::dist::{Categorical, SymmetricDirichlet};

    #[test]
    fn rv_sbd_smoke() {
        let alpha = 0.5;
        let prior = Sb::new(alpha, None);
        let sbd = Sbd::new(alpha, None).unwrap();
        // populate sbd
        (0..1).for_each(|x| {
            sbd.ln_f(&x);
        });

        let ln_f = prior.ln_f(&sbd);
        eprintln!("{ln_f}");
    }

    #[test]
    fn sbd_posterior_smoke() {
        let alpha = 0.5;
        let prior = Sb::new(alpha, None);
        let mut stat = SbdSuffStat::new();
        // populate sbd
        (0..10_usize)
            .cycle()
            .take(1000)
            .for_each(|x| stat.observe(&x));

        let mut rng = rand::thread_rng();
        let post = prior.posterior(&DataOrSuffStat::SuffStat(&stat));
        let sbd = post.draw(&mut rng);

        assert_eq!(sbd.k(), 10);

        eprintln!("POST: {:?}\n", post);
        eprintln!("SBD: {:?}", sbd);
    }

    #[test]
    fn breaking_ex1() {
        // This example used to result in NaN weight
        let ln_weights = [
            (-0.627_512_722_454_201_5_f64).exp(),
            (-0.763_394_023_914_272_2_f64).exp(),
        ];
        let alpha = 1.0;
        let prior = Sb::new(alpha, None);
        let sbd =
            Sbd::from_canonical_weights(&ln_weights, alpha, None).unwrap();
        let ln_f = prior.ln_f(&sbd);
        eprintln!("{:?}", ln_weights);
        eprintln!("{ln_f}");
    }

    #[test]
    fn posterior_sanity() {
        let data: Vec<usize> = vec![0, 3, 3, 3, 3, 3, 4, 4, 4, 6];

        let mut stat = SbdSuffStat::new();
        stat.observe_many(&data);

        let prior = Sb::new(0.1, None);

        let mut rng = rand::thread_rng();
        let post = prior.posterior(&DataOrSuffStat::SuffStat(&stat));

        let mut sums = vec![0.0; 4];

        for _ in 0..100 {
            let sbd = post.draw(&mut rng);
            let f0 = sbd.f(&0_usize);
            let f3 = sbd.f(&3_usize);
            let f4 = sbd.f(&4_usize);
            let f6 = sbd.f(&6_usize);

            sums[0] += f0;
            sums[1] += f3;
            sums[2] += f4;
            sums[3] += f6;
        }
        assert!(sums[1] > sums[2]);
        assert!(sums[2] > sums[0]);
        assert!(sums[2] > sums[3]);
    }

    #[test]
    fn sbd_vs_canonical_cat_posterior() {
        let alpha = 1.2;
        let data: Vec<usize> = vec![0, 1, 1, 1, 1, 1, 2, 2, 2, 3];

        let mut sbd_stat = SbdSuffStat::new();
        sbd_stat.observe_many(&data);

        let mut cat_stat = CategoricalSuffStat::new(5);
        cat_stat.observe_many(&data);

        let sb = Sb::new(alpha, None);
        let dir = SymmetricDirichlet::new(alpha, 5).unwrap();

        let sbd_post = sb.posterior(&DataOrSuffStat::SuffStat(&sbd_stat));
        let cat_post = <SymmetricDirichlet as ConjugatePrior<
            usize,
            Categorical,
        >>::posterior(
            &dir, &DataOrSuffStat::SuffStat(&cat_stat)
        );

        assert_eq!(sbd_post.dir.alphas.len(), cat_post.alphas.len());

        for (w1, w2) in sbd_post.dir.alphas.iter().zip(cat_post.alphas.iter()) {
            assert::close(*w1, *w2, 1e-10);
        }
    }

    #[test]
    fn predictive_quotient_observed() {
        let alpha = 0.82;
        let data: Vec<usize> = vec![0, 1, 1, 1, 1, 1, 2, 2, 2, 3];

        let mut x = SbdSuffStat::new();
        x.observe_many(&data);

        let sb = Sb::new(alpha, None);

        let pp_0 = sb.pp(&0_usize, &DataOrSuffStat::SuffStat(&x));
        let pp_1 = sb.pp(&1_usize, &DataOrSuffStat::SuffStat(&x));
        let pp_2 = sb.pp(&2_usize, &DataOrSuffStat::SuffStat(&x));
        let pp_3 = sb.pp(&3_usize, &DataOrSuffStat::SuffStat(&x));

        let m = sb.m(&DataOrSuffStat::SuffStat(&x));

        fn m_x(x: usize, sb: &Sb, stat: &SbdSuffStat) -> f64 {
            let mut stat_cpy = stat.clone();
            stat_cpy.observe(&x);
            sb.m(&DataOrSuffStat::SuffStat(&stat_cpy))
        }

        let m_0 = m_x(0, &sb, &x);
        let m_1 = m_x(1, &sb, &x);
        let m_2 = m_x(2, &sb, &x);
        let m_3 = m_x(3, &sb, &x);

        assert::close(m_0 / m, pp_0, 1e-12);
        assert::close(m_1 / m, pp_1, 1e-12);
        assert::close(m_2 / m, pp_2, 1e-12);
        assert::close(m_3 / m, pp_3, 1e-12);
    }

    #[test]
    fn gks_values() {
        let data: Vec<usize> = vec![0, 0, 0, 1, 1, 2, 3, 3];

        let mut stat = SbdSuffStat::new();
        stat.observe_many(&data);

        let gk = gks(&stat);

        assert_eq!(gk.len(), 4);
        assert_eq!(gk[0], 8.0);
        assert_eq!(gk[1], 5.0);
        assert_eq!(gk[2], 3.0);
        assert_eq!(gk[3], 2.0);
    }

    #[test]
    fn sbd_marginal_values() {
        let alpha = 0.8;
        let data: Vec<usize> = vec![0, 0, 0, 1, 1, 2, 3, 3];

        let mut stat = SbdSuffStat::new();
        stat.observe_many(&data);

        let x = DataOrSuffStat::SuffStat(&stat);

        let sb = Sb::new(alpha, None);

        let ln_m = sb.ln_m(&x);

        assert::close(ln_m, -14.038534276704162, 1e-10);
    }

    #[test]
    fn sbd_predictive_values() {
        let alpha = 0.8;
        let data: Vec<usize> = vec![0, 0, 0, 1, 1, 2, 3, 3];

        let mut stat = SbdSuffStat::new();
        stat.observe_many(&data);

        let x = DataOrSuffStat::SuffStat(&stat);

        let sb = Sb::new(alpha, None);

        assert::close(sb.ln_pp(&0, &x), -0.8960880245566358, 1e-12);
        assert::close(sb.ln_pp(&1, &x), -1.342834791638104, 1e-12);
        assert::close(sb.ln_pp(&2, &x), -1.9819147509277737, 1e-12);
        assert::close(sb.ln_pp(&3, &x), -1.8818312923707912, 1e-12);
        assert::close(sb.ln_pp(&4, &x), -3.7913737972552295, 1e-12);
        assert::close(sb.ln_pp(&5, &x), -3.7913737972552295, 1e-12);
    }

    #[test]
    fn sbd_bayes_law() {
        let alpha = 3.0;
        let data: Vec<usize> = vec![0, 0, 0, 1, 1, 2, 3, 3];
        let weights: Vec<f64> = vec![0.5, 0.2, 0.1, 0.1, 0.05, 0.05];

        let mut stat = SbdSuffStat::new();
        stat.observe_many(&data);

        let x = DataOrSuffStat::SuffStat(&stat);

        let sb = Sb::new(alpha, None);
        let posterior = sb.posterior(&x);
        let sbd: Sbd = Sbd::from_weights(&weights, alpha, None).unwrap();

        let ln_post = posterior.ln_f(&sbd);
        let ln_like = data.iter().map(|x_i| sbd.ln_f(x_i)).sum::<f64>();
        let ln_prior = sb.ln_f(&sbd);
        let ln_m = sb.ln_m(&x);

        eprintln!("ln_like: {ln_like}");
        eprintln!("ln_prior: {ln_prior}");
        eprintln!("ln_marg: {ln_m}");
        eprintln!("ln_post: {ln_post}");
        eprintln!("ln_like + ln_prior - ln_m : {}", ln_like + ln_prior - ln_m);

        assert::close(ln_post, ln_like + ln_prior - ln_m, 1e-10);
    }

    #[test]
    fn sbd_prior_values() {
        let alpha = 0.8;
        let weights = vec![0.4, 0.1, 0.2, 0.1, 0.2];
        let sbd = Sbd::from_weights(&weights, alpha, None).unwrap();
        let sb = Sb::new(alpha, None);
        let ln_prior = sb.ln_f(&sbd);

        assert::close(ln_prior, -0.5706866227700182, 1e-10);
    }
}
