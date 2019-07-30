#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::consts;
use crate::dist;
use crate::impl_display;
use crate::result;
use crate::traits::*;
use getset::Setters;
use rand::Rng;
use special::Gamma;
use std::f32;
use std::f64;
use std::f64::consts::{LN_2, PI};

/// [Generalized Extreme Value Distribution](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution)
/// Gev(μ, σ, ξ) where the parameters are
/// μ is location
/// σ is the scale
/// ξ is the shape
///
/// ```math
/// f(x|μ, σ, ξ) = \frac{1}{σ} t(x)^{ξ + 1} e^{-t(x)}
///
/// t(x) = ⎰ (1 + ξ ((x - μ) / σ))^(-1/ξ) if ξ ≠ 0
///        ⎱ e^{(μ - x) / σ}              if ξ = 0
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Setters)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Gev {
    #[set = "pub"]
    loc: f64,
    #[set = "pub"]
    scale: f64,
    #[set = "pub"]
    shape: f64,
}

impl Gev {
    /// Create a new `Gev` distribution with location, scale, and shape.
    pub fn new(loc: f64, scale: f64, shape: f64) -> result::Result<Self> {
        let scale_ok = scale > 0.0 && scale.is_finite();
        let loc_ok = loc.is_finite();
        let shape_ok = shape.is_finite();

        if scale_ok && loc_ok && shape_ok {
            Ok(Gev { loc, scale, shape })
        } else {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let msg = "location, shape, and scale must all be finite and scale must be greater than zero.";
            let err = result::Error::new(err_kind, msg);
            Err(err)
        }
    }

    /// Creates a new Gev without checking whether the parameters are valid.
    pub fn new_unchecked(loc: f64, scale: f64, shape: f64) -> Self {
        Gev { loc, scale, shape }
    }

    /// Get the location parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gev;
    /// let gev = Gev::new(1.2, 2.3, 3.4).unwrap();
    ///
    /// assert_eq!(gev.loc(), 1.2);
    /// ```
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// Get the shape parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gev;
    /// let gev = Gev::new(1.2, 2.3, 3.4).unwrap();
    ///
    /// assert_eq!(gev.shape(), 3.4);
    /// ```
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Get the scale parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gev;
    /// let gev = Gev::new(1.2, 2.3, 3.4).unwrap();
    ///
    /// assert_eq!(gev.scale(), 2.3);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

fn t(loc: f64, shape: f64, scale: f64, x: f64) -> f64 {
    if shape == 0.0 {
        ((loc - x) / scale).exp()
    } else {
        (1.0 + shape * (x - loc) / scale).powf(-1.0 / shape)
    }
}

impl From<&Gev> for String {
    fn from(gev: &Gev) -> String {
        format!("GEV(μ: {}, α: {}, ξ: {})", gev.loc, gev.scale, gev.shape)
    }
}

impl_display!(Gev);

macro_rules! impl_traits {
    ($kind: ty) => {
        impl Rv<$kind> for Gev {
            fn ln_f(&self, x: &$kind) -> f64 {
                let tv = t(self.loc, self.shape, self.scale, f64::from(*x));
                -self.scale.ln() + (self.shape + 1.0) * tv.ln() - tv
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let uni = dist::Uniform::new(0.0, 1.0).unwrap();
                let u: f64 = uni.draw(rng);
                let lnu = -u.ln();
                if self.shape == 0.0 {
                    (self.loc - self.scale * lnu.ln()) as $kind
                } else {
                    (self.loc
                        + self.scale * (lnu.powf(-self.shape) - 1.0)
                            / self.shape) as $kind
                }
            }
        }

        impl ContinuousDistr<$kind> for Gev {}

        impl Support<$kind> for Gev {
            fn supports(&self, x: &$kind) -> bool {
                if self.shape > 0.0 {
                    x.is_finite()
                        && f64::from(*x) >= self.loc - self.scale / self.shape
                } else if self.shape == 0.0 {
                    x.is_finite()
                } else {
                    x.is_finite()
                        && f64::from(*x) <= self.loc - self.scale / self.shape
                }
            }
        }

        impl Cdf<$kind> for Gev {
            fn cdf(&self, x: &$kind) -> f64 {
                (-t(self.loc, self.shape, self.scale, f64::from(*x))).exp()
            }
        }

        impl Mean<$kind> for Gev {
            fn mean(&self) -> Option<$kind> {
                if self.shape == 0.0 {
                    Some(
                        (self.loc + self.scale * consts::EULER_MASCERONI)
                            as $kind,
                    )
                } else if self.shape >= 1.0 {
                    Some(f64::INFINITY as $kind)
                } else {
                    let g1 = (1.0 - self.shape).gamma();
                    Some(
                        (self.loc + self.scale * (g1 - 1.0) / self.shape)
                            as $kind,
                    )
                }
            }
        }

        impl Mode<$kind> for Gev {
            fn mode(&self) -> Option<$kind> {
                if self.shape == 0.0 {
                    Some(self.loc as $kind)
                } else {
                    Some(
                        (self.loc
                            + (self.scale
                                * (1.0 + self.shape).powf(-self.shape)
                                - 1.0)
                                / self.shape) as $kind,
                    )
                }
            }
        }

        impl Median<$kind> for Gev {
            fn median(&self) -> Option<$kind> {
                if self.shape == 0.0 {
                    Some((self.loc - self.scale * consts::LN_LN_2) as $kind)
                } else {
                    Some(
                        (self.loc
                            + self.scale * (LN_2.powf(-self.shape) - 1.0)
                                / self.shape) as $kind,
                    )
                }
            }
        }
    };
}

impl Variance<f64> for Gev {
    fn variance(&self) -> Option<f64> {
        if self.shape == 0.0 {
            Some(self.scale * self.scale * PI * PI / 6.0)
        } else if self.shape >= 0.5 {
            Some(f64::INFINITY)
        } else {
            let g1 = (1.0 - self.shape).gamma();
            let g2 = (1.0 - 2.0 * self.shape).gamma();
            Some(
                self.scale * self.scale * (g2 - g1 * g1)
                    / (self.shape * self.shape),
            )
        }
    }
}

impl Entropy for Gev {
    fn entropy(&self) -> f64 {
        self.scale.ln()
            + consts::EULER_MASCERONI * self.shape
            + consts::EULER_MASCERONI
            + 1.0
    }
}

impl_traits!(f32);
impl_traits!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;
    use crate::misc::linspace;
    use std::f64;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    #[test]
    fn new() {
        let gev = Gev::new(0.0, 1.0, 0.0).unwrap();
        assert::close(gev.loc, 0.0, TOL);
        assert::close(gev.scale, 1.0, TOL);
        assert::close(gev.shape, 0.0, TOL);
    }

    #[test]
    fn ln_pdf_0() {
        let gev = Gev::new(0.0, 1.0, 0.0).unwrap();
        let xs: Vec<f64> = linspace(-10.0, 10.0, 50);

        let this_ln_pdf: Vec<f64> = xs.iter().map(|x| gev.ln_f(x)).collect();
        let known_ln_pdf: Vec<f64> = vec![
            -22016.465794806718,
            -14635.151518216397,
            -9727.6715229257752,
            -6464.9705171385704,
            -4295.8342439455982,
            -2853.7767041299953,
            -1895.1322342252688,
            -1257.8947666614729,
            -834.35127549932372,
            -552.88656674623974,
            -365.88582347668228,
            -241.69136713870239,
            -159.2549468723667,
            -104.58220539921258,
            -68.36870992145117,
            -44.42821923014872,
            -28.647685154952825,
            -18.292464043881665,
            -11.544372497764506,
            -7.1945543387135125,
            -4.4392769732597488,
            -2.7441624550266681,
            -1.7539187479639238,
            -1.2323227224743045,
            -1.0223166308609293,
            -1.0194774382005241,
            -1.1543773678791485,
            -1.3808559518631276,
            -1.6682224650132045,
            -1.9960715550940842,
            -2.3508363090414059,
            -2.7234964890286628,
            -3.1080548066483384,
            -3.5005238428395669,
            -3.8982524810165562,
            -4.2994780724473367,
            -4.7030286843059548,
            -5.1081251332397493,
            -5.5142493633639287,
            -5.9210569346967432,
            -6.3283188393174106,
            -6.7358828166564262,
            -7.1436476331802625,
            -7.55154598171712,
            -7.9595331117261701,
            -8.3675792699010145,
            -8.775664674151324,
            -9.1837761719523758,
            -9.5919050185808512,
            -10.000045399929762,
        ];

        assert::close(known_ln_pdf, this_ln_pdf, TOL);
    }

    #[test]
    fn ln_pdf_one_half() {
        let gev = Gev::new(0.0, 1.0, 0.5).unwrap();
        let xs: Vec<f64> = linspace(-1.9, 10.0, 50);

        let this_ln_pdf: Vec<f64> = xs.iter().map(|x| gev.ln_f(x)).collect();
        let known_ln_pdf: Vec<f64> = vec![
            -391.01280317933725,
            -28.737012000993683,
            -7.9755152856461002,
            -3.1827989100658023,
            -1.6119815172254617,
            -1.0561284444156163,
            -0.89880916566197588,
            -0.91848635426108904,
            -1.0220887003150052,
            -1.1662191778735678,
            -1.3291403664858692,
            -1.499425187893265,
            -1.6708888163001137,
            -1.8401487079858558,
            -2.0053779760511659,
            -2.1656373896589791,
            -2.3205034040066055,
            -2.4698545283071605,
            -2.6137455891174968,
            -2.7523323300807534,
            -2.8858256018567796,
            -3.014463329165149,
            -3.138493349819397,
            -3.258162997160523,
            -3.3737129089187619,
            -3.4853735023913863,
            -3.5933631353703781,
            -3.6978873294780152,
            -3.7991386561627816,
            -3.897297027438662,
            -3.992530224449923,
            -4.0849945558882279,
            -4.1748355767638445,
            -4.2621888232889233,
            -4.3471805362681231,
            -4.4299283563624767,
            -4.5105419818120751,
            -4.5891237839265617,
            -4.6657693787083963,
            -4.7405681549141123,
            -4.8136037600524562,
            -4.884954546513983,
            -4.9546939803922845,
            -5.0228910157061009,
            -5.0896104367412303,
            -5.1549131711535008,
            -5.2188565763442245,
            -5.2814947014612086,
            -5.3428785272069916,
            -5.4030561854619421,
        ];

        assert::close(known_ln_pdf, this_ln_pdf, TOL);
    }

    #[test]
    fn ln_pdf_minus_one_half() {
        let gev = Gev::new(0.0, 1.0, -0.5).unwrap();
        let xs: Vec<f64> = linspace(-10.0, 1.9, 50);

        let this_ln_pdf: Vec<f64> = xs.iter().map(|x| gev.ln_f(x)).collect();
        let known_ln_pdf: Vec<f64> = vec![
            -34.208240530771945,
            -32.786288262748577,
            -31.39425255765369,
            -30.03215161196751,
            -28.700004811360039,
            -27.397832836625589,
            -26.1256577816823,
            -24.883503285324355,
            -23.67139467869616,
            -22.489359150795124,
            -21.33742593471408,
            -20.215626517822667,
            -19.12399487967734,
            -18.062567762169611,
            -17.03138497730048,
            -16.030489759050919,
            -15.059929167153456,
            -14.119754552231905,
            -13.21002209385319,
            -12.330793425651896,
            -11.482136365003441,
            -10.664125768956731,
            -9.8768445435857011,
            -9.1203848409894999,
            -8.3948494874264163,
            -7.7003536982969738,
            -7.0370271520187426,
            -6.4050165168703579,
            -5.804488554972564,
            -5.2356339691930582,
            -4.6986722171283697,
            -4.1938575994160905,
            -3.7214870499178971,
            -3.2819102326246732,
            -2.8755428168073927,
            -2.5028842120645773,
            -2.1645416916140485,
            -1.8612638809699724,
            -1.5939883451786301,
            -1.3639110573824136,
            -1.1725910563550697,
            -1.0221141188800091,
            -0.91536051565782661,
            -0.85646800976791571,
            -0.85169058025414157,
            -0.91114410499136123,
            -1.0528320651241088,
            -1.3138356620274461,
            -1.7929763473633977,
            -2.9982322735539904,
        ];

        assert::close(known_ln_pdf, this_ln_pdf, TOL);
    }

    #[test]
    fn cdf() {
        let gev_a = Gev::new(0.0, 1.0, 0.0).unwrap();
        let gev_b = Gev::new(0.0, 1.0, 0.5).unwrap();
        let gev_c = Gev::new(0.0, 1.0, -0.5).unwrap();

        assert::close(gev_a.cdf(&0.0), 0.36787944117144233, TOL);
        assert::close(gev_a.cdf(&2.0), 0.87342301849311665, TOL);
        assert::close(gev_a.cdf(&-2.0), 0.00061797898933109343, TOL);

        assert::close(gev_b.cdf(&0.0), 0.36787944117144233, TOL);
        assert::close(gev_b.cdf(&2.0), 0.77880078307140488, TOL);
        assert::close(gev_b.cdf(&-2.0), 0.0, TOL);

        assert::close(gev_c.cdf(&0.0), 0.36787944117144233, TOL);
        assert::close(gev_c.cdf(&2.0), 1.0, TOL);
        assert::close(gev_c.cdf(&-2.0), 0.018315638888734179, TOL);
    }

    #[test]
    fn entropy() {
        let gev_a = Gev::new(0.0, 1.0, 0.0).unwrap();
        let gev_b = Gev::new(0.0, 1.0, 0.5).unwrap();
        let gev_c = Gev::new(0.0, 1.0, -0.5).unwrap();

        assert::close(gev_a.entropy(), 1.5772156649015328, TOL);
        assert::close(gev_b.entropy(), 1.8658234973522994, TOL);
        assert::close(gev_c.entropy(), 1.2886078324507664, TOL);
    }

    #[test]
    fn draw_0() {
        let mut rng = rand::thread_rng();
        let gev = Gev::new(0.0, 1.0, 0.0).unwrap();
        let cdf = |x: f64| gev.cdf(&x);

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = gev.sample(1000, &mut rng);
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
    fn draw_one_half() {
        let mut rng = rand::thread_rng();
        let gev = Gev::new(0.0, 1.0, 0.5).unwrap();
        let cdf = |x: f64| gev.cdf(&x);

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = gev.sample(1000, &mut rng);
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
    fn draw_negative_one_half() {
        let mut rng = rand::thread_rng();
        let gev = Gev::new(0.0, 1.0, -0.5).unwrap();
        let cdf = |x: f64| gev.cdf(&x);

        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = gev.sample(1000, &mut rng);
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
    fn mode() {
        let m1: f64 = Gev::new(0.0, 1.0, 0.0).unwrap().mode().unwrap();
        let m2: f64 = Gev::new(0.0, 1.0, 0.5).unwrap().mode().unwrap();
        let m3: f64 = Gev::new(0.0, 1.0, -0.5).unwrap().mode().unwrap();

        assert::close(m1, 0.0, TOL);
        assert::close(
            m2,
            -0.367006838144547934535143950196072405356035012895553,
            TOL,
        );
        assert::close(
            m3,
            0.5857864376269049511983112757903019214303281246230519,
            TOL,
        );
    }

    #[test]
    fn median() {
        let m1: f64 = Gev::new(0.0, 1.0, 0.0).unwrap().median().unwrap();
        let m2: f64 = Gev::new(0.0, 1.0, 0.5).unwrap().median().unwrap();
        let m3: f64 = Gev::new(0.0, 1.0, -0.5).unwrap().median().unwrap();

        assert::close(
            m1,
            0.3665129205816643270124391582326694694542634478371052,
            TOL,
        );
        assert::close(
            m2,
            0.4022448175728995897156065721904434451335280561373988,
            TOL,
        );
        assert::close(
            m3,
            0.3348907776846044872936707102095979047388222954711185,
            TOL,
        );
    }

    #[test]
    fn variance() {
        let m1: f64 = Gev::new(0.0, 1.0, 0.0).unwrap().variance().unwrap();
        let m2: f64 = Gev::new(0.0, 1.0, 0.5).unwrap().variance().unwrap();
        let m3: f64 = Gev::new(0.0, 1.0, -0.5).unwrap().variance().unwrap();

        assert::close(
            m1,
            1.644934066848226436472415166646025189218949901206798,
            TOL,
        );
        assert::close(m2, f64::INFINITY, TOL);
        assert::close(
            m3,
            0.858407346410206761537356616720497115802830600624894,
            TOL,
        );
    }
}
