//! Kolmogorow-Smirnov two-sided test for large values of N.
//! Heavily inspired by SciPy's implementation which can be found here:
//! https://github.com/scipy/scipy/blob/a767030252ba3f7c8e2924847dffa7024171657b/scipy/special/cephes/kolmogorov.c#L153

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::traits::*;
use rand::Rng;
use std::f64::consts::{PI, SQRT_2};

#[inline]
fn within_tol(x: f64, y: f64, atol: f64, rtol: f64) -> bool {
    let diff = (x - y).abs();
    diff <= (atol + rtol * y.abs())
}

/// Kolmogorov-Smirnov distribution where the number of samples, $N$, is assumed to be large
/// This is the distribution of $\sqrt{N} D_n$ where $D_n = \sup_x |F_n(x) - F(x)|$ where $F$
/// is the true CDF and $F_n$ the emperical CDF.
///
/// # Example
///
/// Calculate the Survival Function for a particular KS stat.
///
/// ```rust
/// use rv::traits::*;
/// use rv::dist::KsTwoAsymptotic;
///
/// let ks = KsTwoAsymptotic::default();
/// let sf = ks.sf(&1.0);
/// const EXPECTED: f64 = 0.26999967167735456;
/// assert!((sf - EXPECTED).abs() < 1E-15);
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct KsTwoAsymptotic {}

struct CdfPdf {
    cdf: f64,
    pdf: f64,
}

const MIN_EXP: f64 = -746.0;
const MIN_THRESHOLD: f64 = PI / (8.0 * -MIN_EXP);
const KOLMOGO_CUTOVER: f64 = 0.82;
const MAX_ITERS: usize = 2000;

impl KsTwoAsymptotic {
    /// Create a new KsTwoAsymptotic distribution
    #[inline]
    pub fn new() -> Self {
        Self {}
    }

    fn compute(x: f64) -> CdfPdf {
        if x <= MIN_THRESHOLD {
            CdfPdf { cdf: 0.0, pdf: 0.0 }
        } else if x <= KOLMOGO_CUTOVER {
            /*
             * u = e^(-pi^2 / (8x^2))
             * w = sqrt(2pi) / x
             * P = w * u * (1 + u^8 + u^24 + u^48 + ...)
             */
            let mut p: f64 = 1.0;
            let mut d: f64 = 0.0;

            let w = (2.0 * PI).sqrt() / x;
            let logu8 = -PI * PI / (x * x);
            let u = (logu8 / 8.0).exp();

            if u == 0.0 {
                let log_p = logu8 / 8.0 + w.ln();
                p = log_p.exp();
            } else {
                let u_8 = logu8.exp();
                let u_8cub = u_8.powi(3);

                p = 1.0 + u_8cub * p;
                d = 25.0 + u_8cub * d;

                p = 1.0 + u_8cub * p;
                d = 9.0 + u_8cub * d;

                p = 1.0 + u_8cub * p;
                d = 1.0 + u_8cub * d;

                d = PI * PI / (4.0 * x * x) * d - p;
                d *= w * u / x;
                p *= w * u;
            }

            CdfPdf {
                cdf: p.max(0.0).min(1.0),
                pdf: d.max(0.0),
            }
        } else {
            let mut p: f64 = 1.0;
            let mut d: f64 = 0.0;
            /*
             * v = e^(-2x^2)
             * P = 2 (v - v^4 + v^9 - v^16 + ...)
             *   = 2v(1 - v^3*(1 - v^5*(1 - v^7*(1 - ...)))
             */
            let logv = -2.0 * x * x;
            let v = logv.exp();
            /*
             * Want q^((2k-1)^2)(1-q^(4k-1)) / q(1-q^3) < epsilon to break out of loop.
             * With KOLMOG_CUTOVER ~ 0.82, k <= 4.  Just unroll the loop, 4 iterations
             */
            let vsq = v * v;
            let v3 = v.powi(3);
            let mut vpwr;

            vpwr = v3 * v3 * v;
            p = 1.0 - vpwr * p;
            d = 3.0 * 3.0 - vpwr * d;

            vpwr = v3 * vsq;
            p = 1.0 - vpwr * p;
            d = 2.0 * 2.0 - vpwr * d;

            vpwr = v3;
            p = 1.0 - vpwr * p;
            d = 1.0 * 1.0 - vpwr * d;

            p *= 2.0 * v;
            d *= 8.0 * v * x;
            p = p.max(0.0);
            let cdf = (1.0 - p).max(0.0).min(1.0);
            let pdf = d.max(0.0);
            CdfPdf { cdf, pdf }
        }
    }

    /// Determine the value s.t.
    /// sf(x) = sf
    /// cdf(x) = cdf
    fn inverse(sf: f64, cdf: f64) -> f64 {
        if !(sf >= 0.0 && cdf >= 0.0 && sf <= 1.0 && cdf <= 1.0)
            || (1.0 - cdf - sf).abs() > 4.0 * std::f64::EPSILON
        {
            std::f64::NAN
        } else if cdf == 0.0 {
            0.0
        } else if sf == 0.0 {
            std::f64::INFINITY
        } else {
            let mut x: f64;
            let mut a: f64;
            let mut b: f64;

            if cdf <= 0.5 {
                let logcdf = cdf.ln();
                let log_sqrt_2pi: f64 = (2.0 * PI).sqrt().ln();

                a = PI
                    / (2.0
                        * SQRT_2
                        * (-(logcdf + logcdf / 2.0 - log_sqrt_2pi)).sqrt());
                b = PI
                    / (2.0 * SQRT_2 * (-(logcdf + 0.0 - log_sqrt_2pi)).sqrt());
                a = PI
                    / (2.0
                        * SQRT_2
                        * (-(logcdf + a.ln() - log_sqrt_2pi)).sqrt());
                b = PI
                    / (2.0
                        * SQRT_2
                        * (-(logcdf + b.ln() - log_sqrt_2pi)).sqrt());
                x = (a + b) / 2.0;
            } else {
                const JITTERB: f64 = std::f64::EPSILON * 256.0;
                let pba = sf / (2.0 * (1.0 - (-4.0_f64).exp()));
                let pbb = sf * (1.0 - JITTERB) / 2.0;

                a = (-0.5 * pba.ln()).sqrt();
                b = (-0.5 * pbb.ln()).sqrt();

                let q = sf / 2.0;
                let q2 = q * q;
                let q3 = q2 * q;

                let q0 = 1.0
                    + q3 * (1.0
                        + q3 * (4.0
                            + q2 * (-1.0
                                + q * (22.0 + q2 * (-13.0 + 140.0 * q)))));
                let q0 = q0 * q;

                x = (-(q0).ln() / 2.0).sqrt();
                if x < a || x > b {
                    x = (a + b) / 2.0;
                }
            }
            assert!(a <= b, format!("{} > {}", a, b));

            for _ in 0..MAX_ITERS {
                let x0 = x;
                let c = Self::compute(x0);
                let df = if cdf < 0.5 {
                    cdf - c.cdf
                } else {
                    (1.0 - c.cdf) - sf
                };

                if df == 0.0 {
                    break;
                }

                if df > 0.0 && x > a {
                    a = x;
                } else if df < 0.0 && x < b {
                    b = x;
                }

                let dfdx = -c.pdf;
                if dfdx.abs() <= 0.0 {
                    x = (a + b) / 2.0;
                } else {
                    let t = df / dfdx;
                    x = x0 - t;
                }

                if x >= a && x <= b {
                    if within_tol(
                        x,
                        x0,
                        std::f64::EPSILON,
                        std::f64::EPSILON * 2.0,
                    ) {
                        break;
                    } else if x == a || x == b {
                        x = (a + b) / 2.0;
                        if x == a || x == b {
                            break;
                        }
                    }
                } else {
                    x = (a + b) / 2.0;
                    if within_tol(
                        x,
                        x0,
                        std::f64::EPSILON,
                        std::f64::EPSILON * 2.0,
                    ) {
                        break;
                    }
                }
            }

            x
        }
    }
}

impl From<&KsTwoAsymptotic> for String {
    fn from(_kstwobign: &KsTwoAsymptotic) -> String {
        "KsTwoAsymptotic()".to_string()
    }
}

impl_display!(KsTwoAsymptotic);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for KsTwoAsymptotic {
            fn ln_f(&self, x: &$kind) -> f64 {
                Self::compute(*x as f64).pdf.ln()
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let p: f64 = rng.gen();
                self.invcdf(p)
            }
        }

        impl Support<$kind> for KsTwoAsymptotic {
            fn supports(&self, x: &$kind) -> bool {
                *x >= 0.0 && *x <= 1.0
            }
        }

        impl ContinuousDistr<$kind> for KsTwoAsymptotic {}

        impl Cdf<$kind> for KsTwoAsymptotic {
            fn cdf(&self, x: &$kind) -> f64 {
                Self::compute(*x as f64).cdf
            }
        }

        impl InverseCdf<$kind> for KsTwoAsymptotic {
            fn invcdf(&self, p: f64) -> $kind {
                Self::inverse(1.0 - p, p) as $kind
            }
        }
    };
}

impl_traits!(f32);
impl_traits!(f64);

#[cfg(test)]
mod test {
    use super::*;
    use crate::misc::ks_test;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;
    const TOL: f64 = 1E-5;

    #[test]
    fn ln_f() {
        let ks = KsTwoAsymptotic::new();
        let xs: [f64; 10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let ys: [f64; 10] = [
            -112.34167178024646,
            -22.599002391501188,
            -7.106946223524299,
            -2.2904051512937476,
            -0.44693911054492863,
            0.2807505379636071,
            0.5096655107352636,
            0.48675279163852686,
            0.3226102117905903,
            0.069478074891398,
        ];

        xs.iter().zip(ys.iter()).for_each(|(x, &y)| {
            let y_est: f64 = ks.ln_f(x);
            assert::close(y_est, y, TOL);
        });
    }

    #[test]
    fn cdf() {
        let ks = KsTwoAsymptotic::new();
        let xs: [f64; 10] = [
            0.1,
            0.3111111111111111,
            0.5222222222222223,
            0.7333333333333333,
            0.9444444444444444,
            1.1555555555555557,
            1.3666666666666667,
            1.577777777777778,
            1.788888888888889,
            2.0,
        ];
        let ys: [f64; 10] = [
            6.609305242245699e-53,
            2.347446802363517e-05,
            0.05207062833501679,
            0.3447355082583501,
            0.6656454869612993,
            0.861626906810242,
            0.9522808244357278,
            0.9862348958973179,
            0.9966777058897003,
            0.9993290747442203,
        ];
        xs.iter().zip(ys.iter()).for_each(|(x, &y)| {
            let y_est: f64 = ks.cdf(x);
            assert::close(y_est, y, TOL);
        });
    }

    #[test]
    fn invcdf() {
        let ks = KsTwoAsymptotic::new();
        let xs: [f64; 10] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let ys: [f64; 10] = [
            0.0,
            0.5711732651063401,
            0.6448126061663567,
            0.7067326523068981,
            0.7661855555617682,
            0.8275735551899059,
            0.8947644549851196,
            0.9730633753323726,
            1.072749174939648,
            1.2238478702170825,
        ];

        xs.iter().zip(ys.iter()).rev().for_each(|(&x, &y)| {
            let y_est: f64 = ks.invcdf(x);
            assert::close(y_est, y, TOL);
        });
    }

    #[test]
    fn draw() {
        let ks = KsTwoAsymptotic::new();
        let mut rng = Xoshiro256Plus::seed_from_u64(0x1234);
        let sample: Vec<f64> = ks.sample(1000, &mut rng);
        let (_, alpha) = ks_test(&sample, |x| ks.cdf(&x));
        assert!(alpha >= 0.05);
    }
}
