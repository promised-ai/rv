#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::consts::LN_2PI;
use crate::data::VonMisesSuffStat;
use crate::impl_display;
use crate::misc::bessel;
use crate::traits::*;
use num::Zero;
use rand::Rng;
use rand_distr::Normal;
use std::f64::consts::PI;
use std::fmt;

/// [VonMises distribution](https://en.wikipedia.org/wiki/Von_Mises_distribution)
/// on the circular interval [0, 2π)
///
/// # Example
///
/// ```
/// use rv::prelude::*;
///
/// let vm = VonMises::new(1.0, 2.0).unwrap();
///
/// // x is in (0, 2π]
/// assert!(!vm.supports(&-0.001_f64));
/// assert!(!vm.supports(&6.3_f64));
///
/// // 103 VonMises draws
/// let mut rng = rand::thread_rng();
/// let xs: Vec<f64> = vm.sample(103, &mut rng);
/// assert_eq!(xs.len(), 103);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct VonMises {
    /// Mean
    mu: f64,
    /// Sort of like precision. Higher k implies lower variance.
    k: f64,
    // bessel:i0(k), save some cycles
    log_i0_k: f64,
    sin_mu: f64,
    cos_mu: f64,
}

impl PartialEq for VonMises {
    fn eq(&self, other: &Self) -> bool {
        self.mu == other.mu && self.k == other.k
    }
}

pub struct VonMisesParameters {
    pub mu: f64,
    pub k: f64,
}

impl Parameterized for VonMises {
    type Parameters = VonMisesParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            mu: self.mu(),
            k: self.k(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.mu, params.k)
    }

    fn map_params(
        &self,
        f: impl Fn(Self::Parameters) -> Self::Parameters,
    ) -> VonMises {
        let params0 = self.emit_params();
        let (mu0, k0) = (params0.mu, params0.k);
        let (mu, k) = {
            let params = f(params0);
            (params.mu, params.k)
        };
        let log_i0_k = if k == k0 {
            self.log_i0_k()
        } else {
            bessel::log_i0(k)
        };
        let (sin_mu, cos_mu) = if mu == mu0 {
            (self.sin_mu(), self.cos_mu())
        } else {
            mu.sin_cos()
        };
        VonMises::from_parts_unchecked(mu, k, log_i0_k, sin_mu, cos_mu)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum VonMisesError {
    /// The mu parameter is infinite or NaN
    MuNotFinite { mu: f64 },
    /// The k parameter is less than zero
    KTooLow { k: f64 },
    /// The k parameter is infinite or NaN
    KNotFinite { k: f64 },
}

impl VonMises {
    /// Create a new VonMises distribution with mean mu, and precision, k.
    pub fn new(mu: f64, k: f64) -> Result<Self, VonMisesError> {
        if !mu.is_finite() {
            Err(VonMisesError::MuNotFinite { mu })
        } else if k < 0.0 {
            Err(VonMisesError::KTooLow { k })
        } else if !k.is_finite() {
            Err(VonMisesError::KNotFinite { k })
        } else {
            let mu = mu.rem_euclid(2.0 * PI);
            let (sin_mu, cos_mu) = mu.sin_cos();
            let log_i0_k = bessel::log_i0(k);
            Ok(VonMises {
                mu,
                k,
                log_i0_k,
                sin_mu,
                cos_mu,
            })
        }
    }

    /// Creates a new VonMises without checking whether the parameters are
    /// valid.
    #[inline]
    pub fn new_unchecked(mu: f64, k: f64) -> Self {
        let (sin_mu, cos_mu) = mu.sin_cos();
        let log_i0_k = bessel::log_i0(k);
        VonMises {
            mu,
            k,
            log_i0_k,
            sin_mu,
            cos_mu,
        }
    }

    #[inline]
    pub fn from_parts_unchecked(
        mu: f64,
        k: f64,
        log_i0_k: f64,
        sin_mu: f64,
        cos_mu: f64,
    ) -> Self {
        VonMises {
            mu,
            k,
            log_i0_k,
            sin_mu,
            cos_mu,
        }
    }

    /// Get the mean parameter, mu
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::VonMises;
    /// let vm = VonMises::new(0.0, 1.0).unwrap();
    /// assert_eq!(vm.mu(), 0.0);
    /// ```
    #[inline]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn sin_mu(&self) -> f64 {
        self.sin_mu
    }

    pub fn cos_mu(&self) -> f64 {
        self.cos_mu
    }

    #[inline]
    fn log_i0_k(&self) -> f64 {
        self.log_i0_k
    }

    /// Set the value of mu
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::VonMises;
    /// let mut vm = VonMises::new(2.0, 1.5).unwrap();
    /// assert_eq!(vm.mu(), 2.0);
    ///
    /// vm.set_mu(1.3).unwrap();
    /// assert_eq!(vm.mu(), 1.3);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::VonMises;
    /// # let mut vm = VonMises::new(2.0, 1.5).unwrap();
    /// assert!(vm.set_mu(1.3).is_ok());
    /// assert!(vm.set_mu(0.0).is_ok());
    /// assert!(vm.set_mu(2.0 * std::f64::consts::PI).is_ok());
    ///
    /// assert!(vm.set_mu(f64::NEG_INFINITY).is_err());
    /// assert!(vm.set_mu(f64::INFINITY).is_err());
    /// assert!(vm.set_mu(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_mu(&mut self, mu: f64) -> Result<(), VonMisesError> {
        if !mu.is_finite() {
            Err(VonMisesError::MuNotFinite { mu })
        } else {
            self.set_mu_unchecked(mu.rem_euclid(2.0 * PI));
            Ok(())
        }
    }

    /// Set the value of mu without input validation
    #[inline]
    pub fn set_mu_unchecked(&mut self, mu: f64) {
        self.mu = mu;
        let (sin_mu, cos_mu) = mu.sin_cos();
        self.sin_mu = sin_mu;
        self.cos_mu = cos_mu;
    }

    /// Get the precision parameter, k
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::VonMises;
    /// let vm = VonMises::new(0.0, 1.0).unwrap();
    /// assert_eq!(vm.k(), 1.0);
    /// ```
    #[inline]
    pub fn k(&self) -> f64 {
        self.k
    }

    /// Set the value of k
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::prelude::*;
    /// let mut vm = VonMises::new(0.0, 1.0).unwrap();
    /// let v1: f64 = vm.variance().unwrap();
    /// assert::close(v1, 0.5536100341034653, 1E-10);
    ///
    /// vm.set_mu(0.2);
    /// vm.set_k(2.0);
    ///
    /// let v2: f64 = vm.variance().unwrap();
    /// assert::close(v2, 0.3022253420359917, 1E-10);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::VonMises;
    /// # let mut vm = VonMises::new(0.0, 1.0).unwrap();
    /// assert!(vm.set_k(0.1).is_ok());
    ///
    /// // Must be greater than zero
    /// assert!(vm.set_k(-1.0).is_err());
    ///
    /// assert!(vm.set_k(f64::INFINITY).is_err());
    /// assert!(vm.set_k(f64::NEG_INFINITY).is_err());
    /// assert!(vm.set_k(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_k(&mut self, k: f64) -> Result<(), VonMisesError> {
        if k < 0.0 {
            Err(VonMisesError::KTooLow { k })
        } else if !k.is_finite() {
            Err(VonMisesError::KNotFinite { k })
        } else {
            self.set_k_unchecked(k);
            Ok(())
        }
    }

    /// Set the value of k without input validation
    #[inline]
    pub fn set_k_unchecked(&mut self, k: f64) {
        self.k = k;
        self.log_i0_k = bessel::log_i0(k);
    }

    /// Perform a slice sampling step for the VonMises distribution
    ///
    /// # Arguments
    ///
    /// * `x` - The current value of x
    /// * `mu` - The mean of the distribution
    /// * `k` - The concentration parameter
    /// * `rng` - The random number generator
    ///
    /// # Returns
    ///
    /// The new value of x
    #[inline]
    pub fn slice_step<R: Rng>(x: f64, mu: f64, k: f64, rng: &mut R) -> f64 {
        // y ~ Uniform(0, exp(k * cos(x - μ)))
        let logy = k.mul_add((x - mu).cos(), rng.gen::<f64>().ln());
        // Need to solve for x in k cos(x) = logy
        // If logy < -k, then we're below the cos curve
        // In that case, sample uniformly on the circle
        let xmax = if logy < -k { PI } else { (logy / k).acos() };
        // Sample uniformly on [-xmax, xmax] and add μ
        let x = xmax.mul_add(rng.gen_range(-1.0..=1.0), mu);
        // Ensure result is in [0, 2π)
        x.rem_euclid(2.0 * PI)
    }
}

impl Default for VonMises {
    fn default() -> Self {
        VonMises::new(0.0, 0.0).unwrap()
    }
}

impl From<&VonMises> for String {
    fn from(vm: &VonMises) -> String {
        format!("VonMises(μ: {}, k: {})", vm.mu, vm.k)
    }
}

impl_display!(VonMises);
crate::impl_scalable!(VonMises);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for VonMises {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xf = f64::from(*x);
                self.k.mul_add((xf - self.mu).cos(), -LN_2PI) - self.log_i0_k()
            }
        }

        impl Sampleable<$kind> for VonMises {
            // Best, D. J., & Fisher, N. I. (1979). Efficient simulation of the
            //     von Mises distribution. Applied Statistics, 152-157.
            // https://www.researchgate.net/publication/246035131_Efficient_Simulation_of_the_von_Mises_Distribution
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                if self.k.is_zero() {
                    rng.gen_range(0.0..=2.0 * PI) as $kind
                } else if self.k > 700.0 {
                    let normal = Normal::new(self.mu, 1.0 / self.k).unwrap();
                    rng.sample(normal).rem_euclid(2.0 * PI) as $kind
                } else {
                    let tau =
                        1.0 + 4.0_f64.mul_add(self.k * self.k, 1.0).sqrt();
                    let rho = (tau - (2.0 * tau).sqrt()) / (2.0 * self.k);
                    let r = rho.mul_add(rho, 1.0) / (2.0 * rho);
                    let mut f;
                    loop {
                        let t;
                        let u;
                        loop {
                            let d: f64 = (rng.gen::<f64>() - 0.5).powi(2);
                            let e: f64 = (rng.gen::<f64>() - 0.5).powi(2);
                            let s: f64 = d + e;
                            if s <= 0.25 {
                                t = d / e;
                                u = 4.0 * s;
                                break;
                            }
                        }
                        let z = (1.0 - t) / (1.0 + t);
                        f = r.mul_add(z, 1.0) / (r + z);
                        let c = self.k * (r - f);
                        if (c.mul_add(2.0 - c, -u) > 0.0)
                            || ((c / u).ln() + 1.0 - c >= 0.0)
                        {
                            break;
                        }
                    }

                    let acf = f.acos();
                    let x = if rng.gen_bool(0.5) {
                        self.mu + acf
                    } else {
                        self.mu - acf
                    };
                    x.rem_euclid(2.0 * PI) as $kind
                }
            }
        }

        // TODO: XXX:This is going to be SLOW, because it uses quadrature.
        impl Cdf<$kind> for VonMises {
            fn cdf(&self, x: &$kind) -> f64 {
                use crate::misc::{
                    gauss_legendre_quadrature_cached, gauss_legendre_table,
                };

                let func = |y: f64| self.f(&y);

                let (weights, roots) = gauss_legendre_table(16);
                gauss_legendre_quadrature_cached(
                    func,
                    (0.0, *x as f64),
                    &weights,
                    &roots,
                )
            }
        }

        impl ContinuousDistr<$kind> for VonMises {}

        impl Support<$kind> for VonMises {
            fn supports(&self, x: &$kind) -> bool {
                let xf = f64::from(*x);
                (0.0..=2.0 * PI).contains(&xf)
            }
        }

        impl Mean<$kind> for VonMises {
            fn mean(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl Median<$kind> for VonMises {
            fn median(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        impl Mode<$kind> for VonMises {
            fn mode(&self) -> Option<$kind> {
                Some(self.mu as $kind)
            }
        }

        // This is the circular variance
        impl Variance<$kind> for VonMises {
            fn variance(&self) -> Option<$kind> {
                let v: f64 = 1.0 - bessel::i1(self.k) / self.log_i0_k().exp();
                Some(v as $kind)
            }
        }
    };
}

impl HasSuffStat<f64> for VonMises {
    type Stat = VonMisesSuffStat;

    fn empty_suffstat(&self) -> Self::Stat {
        VonMisesSuffStat::new()
    }

    fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
        self.k.mul_add(
            stat.sum_cos()
                .mul_add(self.cos_mu(), stat.sum_sin() * self.sin_mu()),
            -(stat.n() as f64 * (self.log_i0_k() + LN_2PI)),
        )
    }
}

impl Entropy for VonMises {
    fn entropy(&self) -> f64 {
        -self.k * bessel::i1(self.k) / self.log_i0_k().exp()
            + LN_2PI
            + self.log_i0_k()
    }
}

impl_traits!(f32);
impl_traits!(f64);

impl std::error::Error for VonMisesError {}

impl fmt::Display for VonMisesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MuNotFinite { mu } => write!(f, "non-finite mu: {}", mu),
            Self::KNotFinite { k } => write!(f, "non-finite k: {}", k),
            Self::KTooLow { k } => {
                write!(f, "k ({}) must be greater than zero", k)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;
    use crate::test::density_histogram_test;
    use crate::test_basic_impls;

    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!(f64, VonMises);

    #[test]
    fn new_should_allow_mu_in_0_2pi() {
        assert!(VonMises::new(0.0, 1.0).is_ok());
        assert!(VonMises::new(PI, 1.0).is_ok());
        assert!(VonMises::new(2.0 * PI, 1.0).is_ok());
    }

    #[test]
    fn mean() {
        let m1: f64 = VonMises::new(0.0, 1.0).unwrap().mean().unwrap();
        assert::close(m1, 0.0, TOL);

        let m2: f64 = VonMises::new(0.2, 1.0).unwrap().mean().unwrap();
        assert::close(m2, 0.2, TOL);
    }

    #[test]
    fn median() {
        let m1: f64 = VonMises::new(0.0, 1.0).unwrap().median().unwrap();
        assert::close(m1, 0.0, TOL);

        let m2: f64 = VonMises::new(0.2, 1.0).unwrap().median().unwrap();
        assert::close(m2, 0.2, TOL);
    }

    #[test]
    fn mode() {
        let m1: f64 = VonMises::new(0.0, 1.0).unwrap().mode().unwrap();
        assert::close(m1, 0.0, TOL);

        let m2: f64 = VonMises::new(0.2, 1.0).unwrap().mode().unwrap();
        assert::close(m2, 0.2, TOL);
    }

    #[test]
    fn variance() {
        let v1: f64 = VonMises::new(0.0, 1.0).unwrap().variance().unwrap();
        assert::close(v1, 0.553_610_034_103_465_3, TOL);

        let v2: f64 = VonMises::new(0.2, 2.0).unwrap().variance().unwrap();
        assert::close(v2, 0.302_225_342_035_991_7, TOL);
    }

    #[test]
    fn ln_pdf() {
        let xs: Vec<f64> = vec![
            0.261_799_387_799_149_4,
            0.523_598_775_598_298_8,
            std::f64::consts::FRAC_PI_4,
            1.047_197_551_196_597_6,
            1.308_996_938_995_747,
            std::f64::consts::FRAC_PI_2,
            1.832_595_714_594_046,
            2.094_395_102_393_195_3,
            2.356_194_490_192_345,
            2.617_993_877_991_494,
            2.879_793_265_790_643_5,
            std::f64::consts::PI,
            3.403_392_041_388_942_2,
            3.665_191_429_188_092,
            3.926_990_816_987_241,
            4.188_790_204_786_390_5,
            4.450_589_592_585_54,
            4.712_388_980_384_69,
            4.974_188_368_183_839,
            5.235_987_755_982_988,
            5.497_787_143_782_138,
            5.759_586_531_581_287,
            6.021_385_919_380_436,
        ];
        let target: Vec<f64> = vec![
            -4.754_467_571_353_559,
            -3.963_299_742_859_813,
            -3.143_188_567_981_266_3,
            -2.350_023_267_988_095_5,
            -1.637_856_747_307_201,
            -1.055_221_977_412_164_7,
            -0.641_824_555_021_850_2,
            -0.425_836_831_300_616_06,
            -0.421_978_012_683_474_73,
            -0.630_511_071_282_183_9,
            -1.037_224_823_770_415_8,
            -1.614_402_400_042_353_1,
            -2.322_710_102_106_105_6,
            -3.113_877_930_599_852,
            -3.933_989_105_478_397_5,
            -4.727_154_405_471_569,
            -5.439_320_926_152_463,
            -6.021_955_696_047_501,
            -6.435_353_118_437_814,
            -6.651_340_842_159_048_5,
            -6.655_199_660_776_191,
            -6.446_666_602_177_482,
            -6.039_952_849_689_25,
        ];
        let vm = VonMises::new(2.23, PI).unwrap();
        let ln_pdfs: Vec<f64> = xs.iter().map(|x| vm.ln_pdf(x)).collect();
        assert::close(ln_pdfs, target, TOL);
    }

    #[test]
    fn all_samples_should_be_supported() {
        let mut rng = rand::thread_rng();
        // kappa should be low so we get samples at the tails
        let vm = VonMises::new(1.5 * PI, 0.25).unwrap();
        let xs: Vec<f64> = vm.sample(1000, &mut rng);
        assert!(xs.iter().all(|x| vm.supports(x)));
    }

    #[test]
    fn vm_draw_test() {
        let mut rng = rand::thread_rng();
        let vm = VonMises::new(1.0, 1.2).unwrap();
        let cdf = |x: f64| vm.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = vm.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }

    #[test]
    fn slice_step_vs_draw_test() {
        let n_samples = 1_000_000;
        let mut rng = rand::thread_rng();
        let mu = 1.5;
        let k = 2.0;
        let vm = VonMises::new(mu, k).unwrap();

        // Generate samples using draw method
        let draw_samples: Vec<f64> = vm.sample(n_samples, &mut rng);

        // Generate samples using slice_step method
        let mut slice_samples = Vec::with_capacity(n_samples);
        let mut x = PI; // Start at a reasonable value
        for _ in 0..n_samples {
            x = VonMises::slice_step(x, mu, k, &mut rng);
            slice_samples.push(x);
        }

        // Use the existing two-sample KS test
        use crate::misc::{ks_two_sample, KsAlternative, KsMode};
        let (_, p_value) = ks_two_sample(
            &draw_samples,
            &slice_samples,
            KsMode::Auto,
            KsAlternative::TwoSided,
        )
        .unwrap();

        dbg!(p_value);
        assert!(
            p_value > 0.01,
            "Slice step sampling failed KS test with p-value {}",
            p_value
        );
    }

    #[test]
    fn vm_density_test() {
        let mut rng = rand::thread_rng();
        let mu = 1.0;
        let k = 100.2;
        let vm = VonMises::new(mu, k).unwrap();
        let xs: Vec<f64> = vm.sample(10_000, &mut rng);
        let density_fn = |x: f64| (k * (x - mu).cos()).exp();
        let normalized = false;
        let num_bins = 100;
        let p_value =
            density_histogram_test(&xs, num_bins, density_fn, normalized)
                .unwrap();
        dbg!(p_value);
        assert!(p_value > 0.01);
    }

    #[test]
    fn ln_f_vs_ln_f_stat_test() {
        // Create a VonMises distribution
        let mut rng = rand::thread_rng();
        let mu = 1.5;
        let k = 2.0;
        let vm = VonMises::new(mu, k).unwrap();

        // 1. Generate a sample
        let sample_size = 100;
        let xs: Vec<f64> = vm.sample(sample_size, &mut rng);

        // 2. Find the ln_f of the sample (direct log-likelihood)
        let ln_f_sum: f64 = xs.iter().map(|x| vm.ln_f(x)).sum();

        // 3. Aggregate the sample into a suffstat
        let mut stat = vm.empty_suffstat();
        for x in &xs {
            stat.observe(x);
        }

        // 4. Compute ln_f_stat
        let ln_f_stat_result = vm.ln_f_stat(&stat);

        // 5. Assert that they are close
        assert::close(ln_f_sum, ln_f_stat_result, 1e-10);
    }

    proptest! {
        #[test]
        fn vonmises_draw_produces_valid_range(
            mu in -10.0..10.0,
            k in 0.0..10000.0,
            seed: u64
        ) {
            let vm = VonMises::new(mu, k).unwrap();
            let mut rng = Xoshiro256Plus::seed_from_u64(seed);

            let sample: f64 = vm.draw(&mut rng);

            prop_assert!(
                (0.0..2.0 * std::f64::consts::PI).contains(&sample),
                "Sample {} not in range [0, 2π) for VonMises({}, {})",
                sample, mu, k
            );
        }
    }
}
