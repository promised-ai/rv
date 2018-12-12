use std::f64::consts::PI;

use rand::Rng;

use crate::consts::LN_2PI;
use crate::misc::{bessel, mod_euc, quad};
use crate::result;
use crate::traits::*;

/// [VonMises distirbution](https://en.wikipedia.org/wiki/Von_Mises_distribution)
/// on the circular interval (0, 2π]
///
/// # Example
///
/// ```
/// extern crate rand;
/// extern crate rv;
///
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
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct VonMises {
    /// Mean
    pub mu: f64,
    /// Sort of like precision. Higher k implies lower variance.
    pub k: f64,
    // bessel:i0(k), save some cycles
    i0_k: f64,
}

impl VonMises {
    pub fn new(mu: f64, k: f64) -> result::Result<Self> {
        let mu_ok = 0.0 <= mu && mu <= 2.0 * PI && mu.is_finite();
        let k_ok = k > 0.0 && k.is_finite();
        if !mu_ok {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let err = result::Error::new(err_kind, "mu must be in [0, 2π]");
            Err(err)
        } else if !k_ok {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "k must be finite and greater than zero";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        } else {
            let i0_k = bessel::i0(k);
            Ok(VonMises { mu, k, i0_k })
        }
    }
}

impl Default for VonMises {
    fn default() -> Self {
        VonMises::new(PI, 1.0).unwrap()
    }
}

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for VonMises {
            fn ln_f(&self, x: &$kind) -> f64 {
                let xf = f64::from(*x);
                self.k * (xf - self.mu).cos() - LN_2PI - self.i0_k.ln()
            }

            // Best, D. J., & Fisher, N. I. (1979). Efficient simulation of the
            //     von Mises distribution. Applied Statistics, 152-157.
            // https://www.researchgate.net/publication/246035131_Efficient_Simulation_of_the_von_Mises_Distribution
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let u = rand::distributions::Open01;
                let tau = 1.0 + (1.0 + 4.0 * self.k.powi(2)).sqrt();
                let rho = (tau * (2.0 * tau).sqrt()) / (2.0 * self.k);
                let r = (1.0 + rho.powi(2)) / (2.0 * rho);

                loop {
                    let u1: f64 = rng.sample(u);
                    let u2: f64 = rng.sample(u);

                    let z: f64 = (PI * u1).cos();
                    let f = (1.0 + r * z) / (r + z);
                    let c = self.k * (r - f);

                    if (c * (2.0 - c) - u2 >= 0.0)
                        || ((c / u2).ln() + 1.0 - c >= 0.0)
                    {
                        let u3: f64 = rng.sample(u);
                        let y = (u3 - 0.5).signum() * f.acos() + self.mu;
                        let x = mod_euc(y, 2.0 * PI) as $kind;
                        if !self.supports(&x) {
                            panic!(format!("VonMises does not support {}", x));
                        } else {
                            return x;
                        }
                    }
                }
            }
        }

        // TODO: XXX:This is going to be SLOW, because it uses quadrature.
        impl Cdf<$kind> for VonMises {
            fn cdf(&self, x: &$kind) -> f64 {
                let func = |y: f64| self.f(&y);
                quad(func, 0.0, f64::from(*x))
            }
        }

        impl ContinuousDistr<$kind> for VonMises {}

        impl Support<$kind> for VonMises {
            fn supports(&self, x: &$kind) -> bool {
                let xf = f64::from(*x);
                0.0 <= xf && xf <= 2.0 * PI
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
                let v: f64 = 1.0 - bessel::i1(self.k) / self.i0_k;
                Some(v as $kind)
            }
        }
    };
}

impl Entropy for VonMises {
    fn entropy(&self) -> f64 {
        -self.k * bessel::i1(self.k) / self.i0_k + LN_2PI + self.i0_k.ln()
    }
}

impl_traits!(f32);
impl_traits!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;
    use std::f64::EPSILON;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    #[test]
    fn new_should_allow_mu_in_0_2pi() {
        assert!(VonMises::new(0.0, 1.0).is_ok());
        assert!(VonMises::new(PI, 1.0).is_ok());
        assert!(VonMises::new(2.0 * PI, 1.0).is_ok());
    }

    #[test]
    fn new_should_not_allow_mu_outside_0_2pi() {
        assert!(VonMises::new(-PI, 1.0).is_err());
        assert!(VonMises::new(-EPSILON, 1.0).is_err());
        assert!(VonMises::new(2.0 * PI + 0.001, 1.0).is_err());
        assert!(VonMises::new(100.0, 1.0).is_err());
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
        assert::close(v1, 0.5536100341034653, TOL);

        let v2: f64 = VonMises::new(0.2, 2.0).unwrap().variance().unwrap();
        assert::close(v2, 0.3022253420359917, TOL);
    }

    #[test]
    fn ln_pdf() {
        let xs: Vec<f64> = vec![
            0.2617993877991494,
            0.5235987755982988,
            0.7853981633974483,
            1.0471975511965976,
            1.308996938995747,
            1.5707963267948966,
            1.832595714594046,
            2.0943951023931953,
            2.356194490192345,
            2.617993877991494,
            2.8797932657906435,
            3.141592653589793,
            3.4033920413889422,
            3.665191429188092,
            3.926990816987241,
            4.1887902047863905,
            4.45058959258554,
            4.71238898038469,
            4.974188368183839,
            5.235987755982988,
            5.497787143782138,
            5.759586531581287,
            6.021385919380436,
        ];
        let target: Vec<f64> = vec![
            -4.754467571353559,
            -3.963299742859813,
            -3.1431885679812663,
            -2.3500232679880955,
            -1.637856747307201,
            -1.0552219774121647,
            -0.6418245550218502,
            -0.42583683130061606,
            -0.42197801268347473,
            -0.6305110712821839,
            -1.0372248237704158,
            -1.6144024000423531,
            -2.3227101021061056,
            -3.113877930599852,
            -3.9339891054783975,
            -4.727154405471569,
            -5.439320926152463,
            -6.021955696047501,
            -6.435353118437814,
            -6.6513408421590485,
            -6.655199660776191,
            -6.446666602177482,
            -6.03995284968925,
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
}
