use crate::dist::{Geometric, Poisson, Uniform};
use crate::impl_display;
use crate::traits::*;
use crate::data::PoissonSuffStat;
use getset::Setters;
use once_cell::sync::OnceCell;
use rand::Rng;
#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};
use special::Gamma as _;
use std::convert::TryInto;
use std::f64::consts::PI;

const MAX_REJECTIONS: usize = 1000;

const ASYM_POLY_COEFFS: [[f64; 7]; 7] = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [23.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [11237.0, -298.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    [2482411.0, -241041.0, -1887.0, 5.0, 0.0, 0.0, 0.0],
    [
        1363929895.0,
        -220083004.0,
        1451274.0,
        -7420.0,
        7.0,
        0.0,
        0.0,
    ],
    [
        175309343349.0,
        -915974552561.0,
        25171388146.0,
        76299326.0,
        -78295.0,
        35.0,
        0.0,
    ],
    [
        525035501918789.0,
        -142838662997982.0,
        7134232164555.0,
        -19956117988.0,
        45700491.0,
        -20190.0,
        5.0,
    ],
];

const ASYM_POLY_DIVS: [f64; 7] = [
    24.0,
    1152.0,
    414720.0,
    39813120.0,
    6688604160.0,
    4815794995200.0,
    115579079884800.0,
];

#[derive(Clone, Debug)]
struct CmpCache {
    /// Log Normalizing Constant
    ln_norm: OnceCell<f64>,
    /// Rejection sampler B_{f/g}^{\nu < 1}
    b_geom: OnceCell<f64>,
    /// Rejection sampler B_{f/g}^{\nu >= 1}
    b_pois: OnceCell<f64>,
    /// Parameter for geometric envelope for nu < 1
    p_geom: OnceCell<f64>,
    /// Mean value
    mean: OnceCell<f64>,
    /// Variance
    variance: OnceCell<f64>,
}

impl Default for CmpCache {
    fn default() -> Self {
        CmpCache {
            ln_norm: OnceCell::new(),
            b_geom: OnceCell::new(),
            b_pois: OnceCell::new(),
            p_geom: OnceCell::new(),
            mean: OnceCell::new(),
            variance: OnceCell::new(),
        }
    }
}

impl CmpCache {
    /// Get the log normalizing constant
    pub fn ln_norm(&self, lambda: f64, nu: f64) -> f64 {
        *self
            .ln_norm
            .get_or_init(|| Cmp::normalizer(lambda, nu).ln())
    }

    /// Get the value for the geometric envelop
    pub fn p_geom(&self, lambda: f64, nu: f64) -> f64 {
        *self.p_geom.get_or_init(|| {
            let mu = lambda.powf(nu.recip());
            2.0 * nu / (2.0 * mu * nu + 1.0 + nu)
        })
    }

    /// Get the value for B_{f/g}^{\nu < 1}
    pub fn b_geom(&self, lambda: f64, nu: f64) -> f64 {
        debug_assert!(nu < 1.0);
        *self.b_geom.get_or_init(|| {
            let p = self.p_geom(lambda, nu);
            let mu = lambda.powf(nu.recip());
            let q = (mu / (1.0 - p).powf(nu.recip())).floor();
            p.recip() * lambda.powf(q)
                / ((1.0 - p).powf(q) * ((q + 1.0).gamma()).powf(nu))
        })
    }

    /// Get the value for B_{f/g}^{\nu >= 1}
    pub fn b_pois(&self, lambda: f64, nu: f64) -> f64 {
        debug_assert!(nu >= 1.0);
        *self.b_pois.get_or_init(|| {
            let mu = lambda.powf(nu.recip());
            (mu.powf(mu.floor()) / (mu + 1.0).gamma().floor()).powf(nu - 1.0)
        })
    }

    /// Get the mean for the distribution
    pub fn mean(&self, lambda: f64, nu: f64) -> f64 {
        *self.mean.get_or_init(|| {

            if nu <= 1.0 || lambda > 10.0_f64.powf(nu) {
                // Use approximations if appropiate 
                // See section 4.1 of
                // https://www.researchgate.net/profile/Seng-Huat_Ong/publication/271737480_Analysis_of_discrete_data_by_Conway-Maxwell_Poisson_distribution/links/57469dc908ae9ace84243f75.pdf
                lambda.powf(nu.recip()) - (nu - 1.0) / (2.0 * nu)
            } else {
                // Calculate directly
                let z = self.ln_norm(lambda, nu).exp();
                let mut term = lambda;
                let mut sum = term;
                let mut i: i32 = 1;
                while term.abs() > std::f64::EPSILON {
                    i += 1;
                    let fi = i as f64;
                    term = fi * lambda.powi(i)
                        / (z * ((i - 1) as f64).gamma().powf(nu));
                    sum += term;
                }
                sum
            }
        })
    }

    /// Get the variance for the distribution
    pub fn variance(&self, lambda: f64, nu: f64) -> f64 {
        *self.variance.get_or_init(|| {
            if nu <= 1.0 || lambda > 10.0_f64.powf(nu) {
                // Use approximations if appropiate 
                // See section 4.1 of
                // https://www.researchgate.net/profile/Seng-Huat_Ong/publication/271737480_Analysis_of_discrete_data_by_Conway-Maxwell_Poisson_distribution/links/57469dc908ae9ace84243f75.pdf
                (lambda.powf(nu.recip()) / nu) * (
                    1.0 
                    + (nu * nu - 1.0) / (24.0 * nu * nu) * lambda.powf(-2.0 / nu)
                )
            } else {
                let z = self.ln_norm(lambda, nu).exp();
                let mut term = lambda;
                let mut sum = term;
                let mut i: i32 = 1;
                while term.abs() > std::f64::EPSILON {
                    i += 1;
                    let fi = i as f64;
                    term = fi * fi * lambda.powi(i)
                        / (z * ((i - 1) as f64).gamma().powf(nu));
                    sum += term;
                }
                sum - self.mean(lambda, nu)
            }
        })
    }
}

/// [Conway-Maxwell-Possion
/// distribution](https://en.wikipedia.org/wiki/Conway%E2%80%93Maxwell%E2%80%93Poisson_distribution)
/// over x in {0, 1, ... }.
///
/// # Example
///
/// ```
/// use rv::prelude::*;
///
/// // Create Cmp(λ=5.3, ν=1.2)
/// let cmp = Cmp::new(5.3, 1.2).unwrap();
///
/// // Draw 100 samples
/// let mut rng = rand::thread_rng();
/// let xs: Vec<u32> = cmp.sample(100, &mut rng);
/// assert_eq!(xs.len(), 100)
/// ```
#[derive(Debug, Setters)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Cmp {
    #[set = "pub"]
    lambda: f64,
    #[set = "pub"]
    nu: f64,
    #[cfg_attr(feature = "serde_support", serde(skip))]
    cache: CmpCache,
}

impl PartialEq<Cmp> for Cmp {
    fn eq(&self, other: &Self) -> bool {
        self.lambda == other.lambda && self.nu == self.nu
    }
}

impl Clone for Cmp {
    fn clone(&self) -> Self {
        Self {
            lambda: self.lambda,
            nu: self.nu,
            cache: self.cache.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum CmpError {
    /// The lambda parameter is less than or equal to zero
    LambdaTooLowError,
    /// The lambda parameter is infinite or NaN
    LambdaNotFiniteError,
    /// The nu parameter is less than or equal to zero
    NuTooLowError,
    /// The Nu parameter is infinite or NaN
    NuNotFiniteError,
}

impl Cmp {
    /// Create a new Cmp distribution with given rate
    pub fn new(lambda: f64, nu: f64) -> Result<Self, CmpError> {
        if lambda <= 0.0 {
            Err(CmpError::LambdaTooLowError)
        } else if !lambda.is_finite() {
            Err(CmpError::LambdaNotFiniteError)
        } else if nu <= 0.0 {
            Err(CmpError::NuTooLowError)
        } else if !nu.is_finite() {
            Err(CmpError::NuNotFiniteError)
        } else {
            Ok(Self::new_unchecked(lambda, nu))
        }
    }

    /// Creates a new Cmp without checking whether the parameters are valid.
    pub fn new_unchecked(lambda: f64, nu: f64) -> Self {
        Cmp {
            lambda,
            nu,
            cache: CmpCache::default(),
        }
    }

    /// Get the lambda parameter
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::Cmp;
    /// let cmp = Cmp::new(2.0, 3.0).unwrap();
    /// assert_eq!(cmp.lambda(), 2.0);
    /// ```
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Get the nu parameter
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::Cmp;
    /// let cmp = Cmp::new(2.0, 3.0).unwrap();
    /// assert_eq!(cmp.nu(), 3.0);
    /// ```
    pub fn nu(&self) -> f64 {
        self.nu
    }

    /// Get the normalizing constant
    pub fn normalizer(lambda: f64, nu: f64) -> f64 {
        if nu > 30.0 {
            1.0 + lambda
        } else {
            Self::norm_exact(lambda, nu)
                .unwrap_or_else(|_| Self::norm_asym(lambda, nu))
        }
    }

    /// Normalizing factor exact result.
    ///
    /// This is prone to overflows and numerical instability for large lambdas.
    fn norm_exact(lambda: f64, nu: f64) -> Result<f64, String> {
        let mut i: usize = 0;
        let mut term: f64 = 1.0;
        let mut fac: f64 = 1.0;
        let mut nom: f64 = 1.0;
        let mut sum: f64 = 1.0;

        while term.abs() > std::f64::EPSILON {
            i += 1;
            fac *= i as f64;
            nom *= lambda;
            term = nom / fac.powf(nu);
            sum += term;
            if i > 1000 {
                return Err("Failed to converge".to_string());
            }
        }
        Ok(sum)
    }

    /// Asymptotic (wrt lambda) Normalizing factor
    ///
    /// See [arXiv:1612.06618v2  [math.ST]  16 Oct 2017](https://arxiv.org/pdf/1612.06618.pdf)
    fn norm_asym(lambda: f64, nu: f64) -> f64 {
        let mult = (nu * lambda.powf(nu.recip())).exp()
            / (lambda.powf((nu - 1.0) / (2.0 * nu))
                * (2.0 * PI).powf((nu - 1.0) / 2.0)
                * nu.sqrt());
        let z = nu * lambda.powf(nu.recip());
        let mut sum = 0.0;
        for i in 0..ASYM_POLY_COEFFS.len() {
            let mut part = 0.0;
            for j in 0..ASYM_POLY_COEFFS[i].len() {
                part += ASYM_POLY_COEFFS[i][j] * nu.powf(2.0 * (j as f64));
            }
            sum += part * (nu * nu - 1.0) * z.powi((i + 1) as i32)
                / ASYM_POLY_DIVS[i];
        }
        mult * sum
    }
}

impl From<&Cmp> for String {
    fn from(cmp: &Cmp) -> String {
        format!("Cmp(λ: {}, ν: {})", cmp.lambda, cmp.nu)
    }
}

impl_display!(Cmp);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Cmp {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xf = f64::from(*x);
                (xf * self.lambda.ln() - self.nu * (xf + 1.0).ln_gamma().0)
                    - self.cache.ln_norm(self.lambda, self.nu)
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                // Uses rejection sampling with Poisson and Geometric envelopes.
                // See Algorithm 2 in https://arxiv.org/pdf/1709.03471.pdf
                let unif = Uniform::new_unchecked(0.0, 1.0);

                if self.nu >= 1.0 {
                    // Use Poisson Envelope
                    let mu = self.lambda.powf(self.nu.recip());
                    let pois = Poisson::new_unchecked(mu);
                    let b = self.cache.b_pois(self.lambda, self.nu);
                    for _i in 0..MAX_REJECTIONS {
                        let y: u32 = pois.draw(rng);
                        let num = self.lambda.powi(y as i32)
                            / ((y + 1) as f64).gamma().powf(self.nu);
                        let denom =
                            mu.powi(y as i32) / ((y + 1) as f64).gamma();
                        let alpha = num / (b * denom);
                        let u: f64 = unif.draw(rng);
                        if u <= alpha {
                            return y as $kind;
                        }
                    }
                } else {
                    // Use Geometric Envelope
                    let p = self.cache.p_geom(self.lambda, self.nu);
                    let geom = Geometric::new_unchecked(p);
                    let mu = self.lambda.powf(self.nu.recip());
                    for _i in 0..MAX_REJECTIONS {
                        let y: u32 = geom.draw(rng);
                        let f = mu.powi(y as i32) / ((y + 1) as f64).gamma();
                        let alpha = f.powf(self.nu)
                            / (self.cache.b_geom(self.lambda, self.nu)
                                * (1.0 - p).powi(y.try_into().unwrap())
                                * p);
                        let u: f64 = unif.draw(rng);
                        if u <= alpha {
                            return y as $kind;
                        }
                    }
                }
                panic!("Failed to generate a drawn value");
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                (0..n).map(|_| self.draw(rng)).collect()
            }
        }

        impl Support<$kind> for Cmp {
            #[allow(unused_comparisons)]
            fn supports(&self, x: &$kind) -> bool {
                *x >= 0
            }
        }

        impl DiscreteDistr<$kind> for Cmp {}

        impl HasSuffStat<$kind> for Cmp {
            type Stat = PoissonSuffStat;
            fn empty_suffstat(&self) -> Self::Stat {
                PoissonSuffStat::new()
            }
        }
    };
}

impl Mean<f64> for Cmp {
    fn mean(&self) -> Option<f64> {
        Some(self.cache.mean(self.lambda, self.nu))
    }
}

impl Variance<f64> for Cmp {
    fn variance(&self) -> Option<f64> {
        Some(self.cache.variance(self.lambda, self.nu))
    }
}

impl_traits!(u8);
impl_traits!(u16);
impl_traits!(u32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::x2_test;
    use std::f64;

    const TOL: f64 = 1E-12;
    const N_TRIES: usize = 5;
    const X2_PVAL: f64 = 0.2;

    #[test]
    fn new() {
        assert::close(Cmp::new(0.001, 3.456).unwrap().lambda, 0.001, TOL);
        assert::close(Cmp::new(1.234, 4.567).unwrap().nu, 4.567, TOL);
    }

    #[test]
    fn new_should_reject_non_finite_rate() {
        assert!(Cmp::new(f64::INFINITY, 1.0).is_err());
        assert!(Cmp::new(1.0, f64::INFINITY).is_err());
        assert!(Cmp::new(1.0, f64::NAN).is_err());
        assert!(Cmp::new(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn new_should_reject_rate_lteq_zero() {
        assert!(Cmp::new(-1.0, 1.0).is_err());
        assert!(Cmp::new(1.0, -1.0).is_err());
    }

    #[test]
    fn ln_pdf_poisson() {
        let cmp = Cmp::new(5.3, 1.0).unwrap();
        assert::close(cmp.ln_pmf(&1_u32), -3.6322931794419238, TOL);
        assert::close(cmp.ln_pmf(&5_u32), -1.7489576399916658, TOL);
        assert::close(cmp.ln_pmf(&11_u32), -4.4575328197350492, TOL);
    }

    #[test]
    fn ln_pdf() {
        let cmp = Cmp::new(6.0, 2.0).unwrap();
        assert::close(cmp.ln_pmf(&0_u32), -3.214551887365854, TOL);
        assert::close(cmp.ln_pmf(&1_u32), -1.4227924181377989, TOL);
        assert::close(cmp.ln_pmf(&5_u32), -3.830738026789671, TOL);
        assert::close(cmp.ln_pmf(&11_u32), -18.509813417605024, TOL);
    }

    #[test]
    fn ln_pdf_asymp_lambda() {
        let cmp = Cmp::new(1000.0, 10.0).unwrap();
        assert::close(cmp.ln_pmf(&0_u32), -7.5979563466765656088, TOL);
        assert::close(cmp.ln_pmf(&1_u32), -0.69020106769442879369, TOL);
        assert::close(cmp.ln_pmf(&5_u32), -20.934097379586336984, TOL);
    }

    #[test]
    fn ln_pdf_asymp_nu() {
        let cmp = Cmp::new(10.0, 100.0).unwrap();
        assert::close(cmp.ln_pmf(&0_u32), -2.3978952727983706694, TOL);
        assert::close(cmp.ln_pmf(&1_u32), -0.095310179804324768327, TOL);
        assert::close(cmp.ln_pmf(&2_u32), -67.107443142804811487, TOL);
    }

    #[test]
    fn draw_test_poisson() {
        let mut rng = rand::thread_rng();
        let cmp = Cmp::new_unchecked(2.0, 1.0);

        // How many bins do we need?
        let k: usize = (0..100)
            .position(|x| cmp.pmf(&(x as u32)) < f64::EPSILON)
            .unwrap_or(99)
            + 1;

        let ps: Vec<f64> = (0..k).map(|x| cmp.pmf(&(x as u32))).collect();

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; k];
            let xs: Vec<u32> = cmp.sample(1000, &mut rng);
            xs.iter().for_each(|&x| f_obs[x as usize] += 1);
            let (_, p) = x2_test(&f_obs, &ps);
            if p > X2_PVAL {
                acc + 1
            } else {
                acc
            }
        });
        assert!(passes > 0);
    }

    #[test]
    fn draw_test_small_nu() {
        let mut rng = rand::thread_rng();
        let cmp = Cmp::new_unchecked(0.8, 0.15);

        // How many bins do we need?
        let k: usize = (0..100)
            .position(|x| cmp.pmf(&(x as u32)) < f64::EPSILON)
            .unwrap_or(99)
            + 1;

        let ps: Vec<f64> = (0..k).map(|x| cmp.pmf(&(x as u32))).collect();

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; k];
            let xs: Vec<u32> = cmp.sample(1000, &mut rng);
            xs.iter().for_each(|&x| f_obs[x as usize] += 1);
            let (_, p) = x2_test(&f_obs, &ps);
            if p > X2_PVAL {
                acc + 1
            } else {
                acc
            }
        });
        assert!(passes > 0);
    }

    #[test]
    fn draw_test_large_nu() {
        let mut rng = rand::thread_rng();
        let cmp = Cmp::new_unchecked(34.5, 2.2);

        // How many bins do we need?
        let k: usize = (0..100)
            .position(|x| cmp.pmf(&(x as u32)) < f64::EPSILON)
            .unwrap_or(99)
            + 1;

        let ps: Vec<f64> = (0..k).map(|x| cmp.pmf(&(x as u32))).collect();

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let mut f_obs: Vec<u32> = vec![0; k];
            let xs: Vec<u32> = cmp.sample(1000, &mut rng);
            xs.iter().for_each(|&x| f_obs[x as usize] += 1);
            let (_, p) = x2_test(&f_obs, &ps);
            if p > X2_PVAL {
                acc + 1
            } else {
                acc
            }
        });
        assert!(passes > 0);
    }

    #[test]
    fn rejection_b_poisson() {
        let cache = CmpCache::default();
        assert::close(cache.b_pois(2.0, 3.0), 1.5874010519681996, TOL);
    }

    #[test]
    fn rejection_b_geom() {
        let cache = CmpCache::default();
        assert::close(cache.p_geom(2.0, 0.5), 0.18181818181818182, TOL);
        assert::close(cache.b_geom(2.0, 0.5), 43.820055510797694, TOL);
    }
}
