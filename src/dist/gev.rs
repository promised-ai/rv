#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::consts;
use crate::impl_display;
use crate::misc::gammafn;
use crate::traits::*;
use rand::Rng;
use std::f32;
use std::f64;
use std::f64::consts::{LN_2, PI};
use std::fmt;

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
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Gev {
    loc: f64,
    scale: f64,
    shape: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum GevError {
    /// The location parameter is infinite or NaN
    LocNotFinite { loc: f64 },
    /// The shape parameter is infinite or NaN
    ShapeNotFinite { shape: f64 },
    /// The scale parameter is infinite or NaN
    ScaleNotFinite { scale: f64 },
    /// The scale parameter is less than or equal to zero
    ScaleTooLow { scale: f64 },
}

impl Gev {
    /// Create a new `Gev` distribution with location, scale, and shape.
    pub fn new(loc: f64, scale: f64, shape: f64) -> Result<Self, GevError> {
        if scale <= 0.0 {
            Err(GevError::ScaleTooLow { scale })
        } else if !scale.is_finite() {
            Err(GevError::ScaleNotFinite { scale })
        } else if !shape.is_finite() {
            Err(GevError::ShapeNotFinite { shape })
        } else if !loc.is_finite() {
            Err(GevError::LocNotFinite { loc })
        } else {
            Ok(Gev { loc, scale, shape })
        }
    }

    /// Creates a new Gev without checking whether the parameters are valid.
    #[inline]
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
    #[inline]
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// Set the loc parameter without input validation
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gev;
    /// let mut gev = Gev::new(1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(gev.loc(), 1.2);
    ///
    /// gev.set_loc(2.8).unwrap();
    /// assert_eq!(gev.loc(), 2.8);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Gev;
    /// # let mut gev = Gev::new(1.2, 2.3, 3.4).unwrap();
    /// assert!(gev.set_loc(2.8).is_ok());
    /// assert!(gev.set_loc(std::f64::INFINITY).is_err());
    /// assert!(gev.set_loc(f64::NEG_INFINITY).is_err());
    /// assert!(gev.set_loc(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_loc(&mut self, loc: f64) -> Result<(), GevError> {
        if !loc.is_finite() {
            Err(GevError::LocNotFinite { loc })
        } else {
            self.set_loc_unchecked(loc);
            Ok(())
        }
    }

    /// Set the loc parameter without input validation
    #[inline]
    pub fn set_loc_unchecked(&mut self, loc: f64) {
        self.loc = loc
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
    #[inline]
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Set the shape parameter without input validation
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gev;
    /// let mut gev = Gev::new(1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(gev.shape(), 3.4);
    ///
    /// gev.set_shape(2.8).unwrap();
    /// assert_eq!(gev.shape(), 2.8);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Gev;
    /// # let mut gev = Gev::new(1.2, 2.3, 3.4).unwrap();
    /// assert!(gev.set_shape(2.8).is_ok());
    /// assert!(gev.set_shape(std::f64::INFINITY).is_err());
    /// assert!(gev.set_shape(f64::NEG_INFINITY).is_err());
    /// assert!(gev.set_shape(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_shape(&mut self, shape: f64) -> Result<(), GevError> {
        if !shape.is_finite() {
            Err(GevError::ShapeNotFinite { shape })
        } else {
            self.set_shape_unchecked(shape);
            Ok(())
        }
    }

    /// Set the shape parameter without input validation
    #[inline]
    pub fn set_shape_unchecked(&mut self, shape: f64) {
        self.shape = shape
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
    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Set the scale parameter without input validation
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rv::dist::Gev;
    /// let mut gev = Gev::new(1.2, 2.3, 3.4).unwrap();
    /// assert_eq!(gev.scale(), 2.3);
    ///
    /// gev.set_scale(2.8).unwrap();
    /// assert_eq!(gev.scale(), 2.8);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Gev;
    /// # let mut gev = Gev::new(1.2, 2.3, 3.4).unwrap();
    /// assert!(gev.set_scale(2.8).is_ok());
    /// assert!(gev.set_scale(0.0).is_err());
    /// assert!(gev.set_scale(-1.0).is_err());
    /// assert!(gev.set_scale(std::f64::INFINITY).is_err());
    /// assert!(gev.set_scale(f64::NEG_INFINITY).is_err());
    /// assert!(gev.set_scale(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_scale(&mut self, scale: f64) -> Result<(), GevError> {
        if !scale.is_finite() {
            Err(GevError::ScaleNotFinite { scale })
        } else if scale <= 0.0 {
            Err(GevError::ScaleTooLow { scale })
        } else {
            self.set_scale_unchecked(scale);
            Ok(())
        }
    }

    /// Set the scale parameter without input validation
    #[inline]
    pub fn set_scale_unchecked(&mut self, scale: f64) {
        self.scale = scale
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
        impl HasDensity<$kind> for Gev {
            fn ln_f(&self, x: &$kind) -> f64 {
                // TODO: could cache ln(scale)
                let tv = t(self.loc, self.shape, self.scale, f64::from(*x));
                (self.shape + 1.0).mul_add(tv.ln(), -self.scale.ln()) - tv
            }
        }

        impl Sampleable<$kind> for Gev {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let uni = rand_distr::Open01;
                let u: f64 = rng.sample(uni);
                let lnu = -u.ln();
                if self.shape == 0.0 {
                    self.scale.mul_add(-lnu.ln(), self.loc) as $kind
                } else {
                    (self.loc
                        + self.scale * (lnu.powf(-self.shape) - 1.0)
                            / self.shape) as $kind
                }
            }
        }

        impl ContinuousDistr<$kind> for Gev {}

        #[allow(clippy::cmp_owned)]
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
                    Some(self.scale.mul_add(consts::EULER_MASCERONI, self.loc)
                        as $kind)
                } else if self.shape >= 1.0 {
                    Some(f64::INFINITY as $kind)
                } else {
                    let g1 = gammafn(1.0 - self.shape);
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
                            + self.scale.mul_add(
                                (1.0 + self.shape).powf(-self.shape),
                                -1.0,
                            ) / self.shape) as $kind,
                    )
                }
            }
        }

        impl Median<$kind> for Gev {
            fn median(&self) -> Option<$kind> {
                if self.shape == 0.0 {
                    Some(self.scale.mul_add(-consts::LN_LN_2, self.loc) as $kind)
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
            let g1 = gammafn(1.0 - self.shape);
            let g2 = gammafn(2.0_f64.mul_add(-self.shape, 1.0));
            Some(
                self.scale * self.scale * g1.mul_add(-g1, g2)
                    / (self.shape * self.shape),
            )
        }
    }
}

impl Entropy for Gev {
    fn entropy(&self) -> f64 {
        consts::EULER_MASCERONI.mul_add(self.shape, self.scale.ln())
            + consts::EULER_MASCERONI
            + 1.0
    }
}

impl_traits!(f32);
impl_traits!(f64);

impl std::error::Error for GevError {}

impl fmt::Display for GevError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocNotFinite { loc } => write!(f, "non-finite loc: {}", loc),
            Self::ShapeNotFinite { shape } => {
                write!(f, "non-finite shape: {}", shape)
            }
            Self::ScaleNotFinite { scale } => {
                write!(f, "non-finite scale: {}", scale)
            }
            Self::ScaleTooLow { scale } => {
                write!(f, "scale ({}) must be greater than zero", scale)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::ks_test;
    use crate::misc::linspace;
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!([continuous] Gev::new(0.0, 1.0, 2.0).unwrap());

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
            -22_016.465_794_806_718,
            -14_635.151_518_216_397,
            -9_727.671_522_925_775,
            -6_464.970_517_138_57,
            -4_295.834_243_945_598,
            -2_853.776_704_129_995_3,
            -1_895.132_234_225_268_8,
            -1_257.894_766_661_472_9,
            -834.351_275_499_323_7,
            -552.886_566_746_239_7,
            -365.885_823_476_682_3,
            -241.691_367_138_702_4,
            -159.254_946_872_366_7,
            -104.582_205_399_212_58,
            -68.368_709_921_451_17,
            -44.428_219_230_148_72,
            -28.647_685_154_952_825,
            -18.292_464_043_881_665,
            -11.544_372_497_764_506,
            -7.194_554_338_713_512_5,
            -4.439_276_973_259_749,
            -2.744_162_455_026_668,
            -1.753_918_747_963_923_8,
            -1.232_322_722_474_304_5,
            -1.022_316_630_860_929_3,
            -1.019_477_438_200_524,
            -1.154_377_367_879_148_5,
            -1.380_855_951_863_127_6,
            -1.668_222_465_013_204_5,
            -1.996_071_555_094_084_2,
            -2.350_836_309_041_406,
            -2.723_496_489_028_663,
            -3.108_054_806_648_338_4,
            -3.500_523_842_839_567,
            -3.898_252_481_016_556,
            -4.299_478_072_447_337,
            -4.703_028_684_305_955,
            -5.108_125_133_239_749,
            -5.514_249_363_363_929,
            -5.921_056_934_696_743,
            -6.328_318_839_317_411,
            -6.735_882_816_656_426,
            -7.143_647_633_180_262_5,
            -7.551_545_981_717_12,
            -7.959_533_111_726_17,
            -8.367_579_269_901_015,
            -8.775_664_674_151_324,
            -9.183_776_171_952_376,
            -9.591_905_018_580_851,
            -10.000_045_399_929_762,
        ];

        assert::close(known_ln_pdf, this_ln_pdf, TOL);
    }

    #[test]
    fn ln_pdf_one_half() {
        let gev = Gev::new(0.0, 1.0, 0.5).unwrap();
        let xs: Vec<f64> = linspace(-1.9, 10.0, 50);

        let this_ln_pdf: Vec<f64> = xs.iter().map(|x| gev.ln_f(x)).collect();
        let known_ln_pdf: Vec<f64> = vec![
            -391.012_803_179_337_25,
            -28.737_012_000_993_683,
            -7.975_515_285_646_1,
            -3.182_798_910_065_802_3,
            -1.611_981_517_225_461_7,
            -1.056_128_444_415_616_3,
            -0.898_809_165_661_975_9,
            -0.918_486_354_261_089,
            -1.022_088_700_315_005_2,
            -1.166_219_177_873_567_8,
            -1.329_140_366_485_869_2,
            -1.499_425_187_893_265,
            -1.670_888_816_300_113_7,
            -1.840_148_707_985_855_8,
            -2.005_377_976_051_166,
            -2.165_637_389_658_979,
            -2.320_503_404_006_605_5,
            -2.469_854_528_307_160_5,
            -2.613_745_589_117_496_8,
            -2.752_332_330_080_753_4,
            -2.885_825_601_856_779_6,
            -3.014_463_329_165_149,
            -3.138_493_349_819_397,
            -3.258_162_997_160_523,
            -3.373_712_908_918_762,
            -3.485_373_502_391_386_3,
            -3.593_363_135_370_378,
            -3.697_887_329_478_015_2,
            -3.799_138_656_162_781_6,
            -3.897_297_027_438_662,
            -3.992_530_224_449_923,
            -4.084_994_555_888_228,
            -4.174_835_576_763_844_5,
            -4.262_188_823_288_923,
            -4.347_180_536_268_123,
            -4.429_928_356_362_477,
            -4.510_541_981_812_075,
            -4.589_123_783_926_562,
            -4.665_769_378_708_396,
            -4.740_568_154_914_112,
            -4.813_603_760_052_456,
            -4.884_954_546_513_983,
            -4.954_693_980_392_284_5,
            -5.022_891_015_706_101,
            -5.089_610_436_741_23,
            -5.154_913_171_153_501,
            -5.218_856_576_344_224_5,
            -5.281_494_701_461_209,
            -5.342_878_527_206_992,
            -5.403_056_185_461_942,
        ];

        assert::close(known_ln_pdf, this_ln_pdf, TOL);
    }

    #[test]
    fn ln_pdf_minus_one_half() {
        let gev = Gev::new(0.0, 1.0, -0.5).unwrap();
        let xs: Vec<f64> = linspace(-10.0, 1.9, 50);

        let this_ln_pdf: Vec<f64> = xs.iter().map(|x| gev.ln_f(x)).collect();
        let known_ln_pdf: Vec<f64> = vec![
            -34.208_240_530_771_945,
            -32.786_288_262_748_58,
            -31.394_252_557_653_69,
            -30.032_151_611_967_51,
            -28.700_004_811_360_04,
            -27.397_832_836_625_59,
            -26.125_657_781_682_3,
            -24.883_503_285_324_355,
            -23.671_394_678_696_16,
            -22.489_359_150_795_124,
            -21.337_425_934_714_08,
            -20.215_626_517_822_667,
            -19.123_994_879_677_34,
            -18.062_567_762_169_61,
            -17.031_384_977_300_48,
            -16.030_489_759_050_92,
            -15.059_929_167_153_456,
            -14.119_754_552_231_905,
            -13.210_022_093_853_19,
            -12.330_793_425_651_896,
            -11.482_136_365_003_441,
            -10.664_125_768_956_731,
            -9.876_844_543_585_701,
            -9.120_384_840_989_5,
            -8.394_849_487_426_416,
            -7.700_353_698_296_974,
            -7.037_027_152_018_743,
            -6.405_016_516_870_358,
            -5.804_488_554_972_564,
            -5.235_633_969_193_058,
            -4.698_672_217_128_37,
            -4.193_857_599_416_090_5,
            -3.721_487_049_917_897,
            -3.281_910_232_624_673,
            -2.875_542_816_807_392_7,
            -2.502_884_212_064_577_3,
            -2.164_541_691_614_048_5,
            -1.861_263_880_969_972_4,
            -1.593_988_345_178_63,
            -1.363_911_057_382_413_6,
            -1.172_591_056_355_069_7,
            -1.022_114_118_880_009_1,
            -0.915_360_515_657_826_6,
            -0.856_468_009_767_915_7,
            -0.851_690_580_254_141_6,
            -0.911_144_104_991_361_2,
            -1.052_832_065_124_108_8,
            -1.313_835_662_027_446,
            -1.792_976_347_363_397_7,
            -2.998_232_273_553_990_4,
        ];

        assert::close(known_ln_pdf, this_ln_pdf, TOL);
    }

    #[test]
    fn cdf() {
        let gev_a = Gev::new(0.0, 1.0, 0.0).unwrap();
        let gev_b = Gev::new(0.0, 1.0, 0.5).unwrap();
        let gev_c = Gev::new(0.0, 1.0, -0.5).unwrap();

        assert::close(gev_a.cdf(&0.0), 0.367_879_441_171_442_33, TOL);
        assert::close(gev_a.cdf(&2.0), 0.873_423_018_493_116_7, TOL);
        assert::close(gev_a.cdf(&-2.0), 0.000_617_978_989_331_093_4, TOL);

        assert::close(gev_b.cdf(&0.0), 0.367_879_441_171_442_33, TOL);
        assert::close(gev_b.cdf(&2.0), 0.778_800_783_071_404_9, TOL);
        assert::close(gev_b.cdf(&-2.0), 0.0, TOL);

        assert::close(gev_c.cdf(&0.0), 0.367_879_441_171_442_33, TOL);
        assert::close(gev_c.cdf(&2.0), 1.0, TOL);
        assert::close(gev_c.cdf(&-2.0), 0.018_315_638_888_734_18, TOL);
    }

    #[test]
    fn entropy() {
        let gev_a = Gev::new(0.0, 1.0, 0.0).unwrap();
        let gev_b = Gev::new(0.0, 1.0, 0.5).unwrap();
        let gev_c = Gev::new(0.0, 1.0, -0.5).unwrap();

        assert::close(gev_a.entropy(), 1.577_215_664_901_532_8, TOL);
        assert::close(gev_b.entropy(), 1.865_823_497_352_299_4, TOL);
        assert::close(gev_c.entropy(), 1.288_607_832_450_766_4, TOL);
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
        assert::close(m2, -0.367_006_838_144_547_93, TOL);
        assert::close(m3, 0.585_786_437_626_905, TOL);
    }

    #[test]
    fn median() {
        let m1: f64 = Gev::new(0.0, 1.0, 0.0).unwrap().median().unwrap();
        let m2: f64 = Gev::new(0.0, 1.0, 0.5).unwrap().median().unwrap();
        let m3: f64 = Gev::new(0.0, 1.0, -0.5).unwrap().median().unwrap();

        assert::close(m1, 0.366_512_920_581_664_35, TOL);
        assert::close(m2, 0.402_244_817_572_899_6, TOL);
        assert::close(m3, 0.334_890_777_684_604_5, TOL);
    }

    #[test]
    fn variance() {
        let m1: f64 = Gev::new(0.0, 1.0, 0.0).unwrap().variance().unwrap();
        let m2: f64 = Gev::new(0.0, 1.0, 0.5).unwrap().variance().unwrap();
        let m3: f64 = Gev::new(0.0, 1.0, -0.5).unwrap().variance().unwrap();

        assert::close(m1, 1.644_934_066_848_226_4, TOL);
        assert::close(m2, f64::INFINITY, TOL);
        assert::close(m3, 0.858_407_346_410_206_8, TOL);
    }
}
