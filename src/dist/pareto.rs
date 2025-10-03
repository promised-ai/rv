//! Pareto distribution over x in [shape, ∞)
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::impl_display;
use crate::traits::{
    Cdf, ContinuousDistr, Entropy, HasDensity, Kurtosis, Mean, Mode,
    Parameterized, Sampleable, Scalable, Shiftable, Skewness, Support,
    Variance,
};
use rand::Rng;
use std::f64;
use std::fmt;

/// [Pareto distribution](https://en.wikipedia.org/wiki/Pareto_distribution) `Pareto(x_m`, α)
/// over x in (`x_m`, ∞).
///
/// **NOTE**: The Pareto distribution is parameterized in terms of shape, α, and
/// scale, `x_m`.
///
/// ```math
///                α x_m^α
/// f(x|α, x_m) = ---------
///               x^(α + 1)
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Pareto {
    shape: f64,
    scale: f64,
}

impl Scalable for Pareto {
    type Output = Pareto;
    type Error = ParetoError;

    fn scaled(self, scale: f64) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
    {
        Pareto::new(self.shape(), self.scale() * scale)
    }

    fn scaled_unchecked(self, scale: f64) -> Self::Output
    where
        Self: Sized,
    {
        Pareto::new_unchecked(self.shape(), self.scale() * scale)
    }
}

pub struct ParetoParameters {
    pub shape: f64,
    pub scale: f64,
}

impl Parameterized for Pareto {
    type Parameters = ParetoParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            shape: self.shape(),
            scale: self.scale(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new_unchecked(params.shape, params.scale)
    }
}

crate::impl_shiftable!(Pareto);

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub enum ParetoError {
    /// Shape parameter is less than or equal to zero
    ShapeTooLow { shape: f64 },
    /// Shape parameter is infinite or NaN
    ShapeNotFinite { shape: f64 },
    /// Scale parameter is less than or equal to zero
    ScaleTooLow { scale: f64 },
    /// Scale parameter is infinite or NaN
    ScaleNotFinite { scale: f64 },
}

impl Pareto {
    /// Create a new `Pareto` distribution with shape (α) and scale (`x_m`).
    pub fn new(shape: f64, scale: f64) -> Result<Self, ParetoError> {
        if shape <= 0.0 {
            Err(ParetoError::ShapeTooLow { shape })
        } else if !shape.is_finite() {
            Err(ParetoError::ShapeNotFinite { shape })
        } else if scale <= 0.0 {
            Err(ParetoError::ScaleTooLow { scale })
        } else if !scale.is_finite() {
            Err(ParetoError::ScaleNotFinite { scale })
        } else {
            Ok(Pareto { shape, scale })
        }
    }

    /// Creates a new Pareto without checking whether the parameters are valid.
    #[inline]
    #[must_use]
    pub fn new_unchecked(shape: f64, scale: f64) -> Self {
        Pareto { shape, scale }
    }

    /// Get shape parameter
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::Pareto;
    /// let pareto = Pareto::new(1.0, 2.0).unwrap();
    /// assert_eq!(pareto.shape(), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Set the shape parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::Pareto;
    /// let mut pareto = Pareto::new(2.0, 1.0).unwrap();
    /// assert_eq!(pareto.shape(), 2.0);
    ///
    /// pareto.set_shape(1.1).unwrap();
    /// assert_eq!(pareto.shape(), 1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Pareto;
    /// # let mut pareto = Pareto::new(2.0, 1.0).unwrap();
    /// assert!(pareto.set_shape(1.1).is_ok());
    /// assert!(pareto.set_shape(0.0).is_err());
    /// assert!(pareto.set_shape(-1.0).is_err());
    /// assert!(pareto.set_shape(f64::INFINITY).is_err());
    /// assert!(pareto.set_shape(f64::NEG_INFINITY).is_err());
    /// assert!(pareto.set_shape(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_shape(&mut self, shape: f64) -> Result<(), ParetoError> {
        if shape <= 0.0 {
            Err(ParetoError::ShapeTooLow { shape })
        } else if !shape.is_finite() {
            Err(ParetoError::ShapeNotFinite { shape })
        } else {
            self.set_shape_unchecked(shape);
            Ok(())
        }
    }

    /// Set the shape parameter without input validation
    #[inline]
    pub fn set_shape_unchecked(&mut self, shape: f64) {
        self.shape = shape;
    }

    /// Get scale parameter
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::Pareto;
    /// let pareto = Pareto::new(1.0, 2.0).unwrap();
    /// assert_eq!(pareto.scale(), 2.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Set the scale parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::Pareto;
    /// let mut pareto = Pareto::new(2.0, 1.0).unwrap();
    /// assert_eq!(pareto.scale(), 1.0);
    ///
    /// pareto.set_scale(1.1).unwrap();
    /// assert_eq!(pareto.scale(), 1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Pareto;
    /// # let mut pareto = Pareto::new(2.0, 1.0).unwrap();
    /// assert!(pareto.set_scale(1.1).is_ok());
    /// assert!(pareto.set_scale(0.0).is_err());
    /// assert!(pareto.set_scale(-1.0).is_err());
    /// assert!(pareto.set_scale(f64::INFINITY).is_err());
    /// assert!(pareto.set_scale(f64::NEG_INFINITY).is_err());
    /// assert!(pareto.set_scale(f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_scale(&mut self, scale: f64) -> Result<(), ParetoError> {
        if scale <= 0.0 {
            Err(ParetoError::ScaleTooLow { scale })
        } else if !scale.is_finite() {
            Err(ParetoError::ScaleNotFinite { scale })
        } else {
            self.set_scale_unchecked(scale);
            Ok(())
        }
    }

    /// Set the scale parameter without input validation
    #[inline]
    pub fn set_scale_unchecked(&mut self, scale: f64) {
        self.scale = scale;
    }
}

impl From<&Pareto> for String {
    fn from(pareto: &Pareto) -> String {
        format!("Pareto(xₘ: {}, α: {})", pareto.scale, pareto.shape)
    }
}

impl_display!(Pareto);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl HasDensity<$kind> for Pareto {
            fn ln_f(&self, x: &$kind) -> f64 {
                // TODO: cache ln(shape) and ln(scale)
                (self.shape + 1.0).mul_add(
                    -f64::from(*x).ln(),
                    self.shape.mul_add(self.scale.ln(), self.shape.ln()),
                )
            }
        }

        impl Sampleable<$kind> for Pareto {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let p =
                    rand_distr::Pareto::new(self.scale, self.shape).unwrap();
                rng.sample(p) as $kind
            }
        }

        impl ContinuousDistr<$kind> for Pareto {}

        impl Support<$kind> for Pareto {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite() && *x >= self.scale as $kind
            }
        }

        impl Cdf<$kind> for Pareto {
            fn cdf(&self, x: &$kind) -> f64 {
                let xk = f64::from(*x);
                1.0 - (self.scale / xk).powf(self.shape)
            }
        }

        impl Mean<$kind> for Pareto {
            fn mean(&self) -> Option<$kind> {
                if self.shape <= 1.0 {
                    Some(f64::INFINITY as $kind)
                } else {
                    Some(
                        ((self.shape * self.scale) / (self.shape - 1.0))
                            as $kind,
                    )
                }
            }
        }

        impl Mode<$kind> for Pareto {
            fn mode(&self) -> Option<$kind> {
                Some(self.scale as $kind)
            }
        }
    };
}

impl Variance<f64> for Pareto {
    fn variance(&self) -> Option<f64> {
        if self.shape <= 2.0 {
            Some(f64::INFINITY)
        } else {
            Some(
                (self.scale * self.scale * self.shape)
                    / ((self.shape - 1.0)
                        * (self.shape - 1.0)
                        * (self.shape - 2.0)),
            )
        }
    }
}

impl Entropy for Pareto {
    fn entropy(&self) -> f64 {
        (self.scale / self.shape).ln() + 1.0 + self.shape.recip()
    }
}

impl Skewness for Pareto {
    fn skewness(&self) -> Option<f64> {
        if self.shape <= 3.0 {
            None
        } else {
            Some(
                ((2.0 * (1.0 + self.shape)) / (self.shape - 3.0))
                    * ((self.shape - 2.0) / self.shape).sqrt(),
            )
        }
    }
}

impl Kurtosis for Pareto {
    fn kurtosis(&self) -> Option<f64> {
        let s = self.shape;
        if s <= 4.0 {
            None
        } else {
            let s2 = s * s;
            Some(
                6.0 * (6.0_f64.mul_add(-s, s2.mul_add(s, s2)) - 2.0)
                    / (s * (s - 3.0) * (s - 4.0)),
            )
        }
    }
}

impl_traits!(f32);
impl_traits!(f64);

impl std::error::Error for ParetoError {}

impl fmt::Display for ParetoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeTooLow { shape } => {
                write!(f, "rate ({shape}) must be greater than zero")
            }
            Self::ShapeNotFinite { shape } => {
                write!(f, "non-finite rate: {shape}")
            }
            Self::ScaleTooLow { scale } => {
                write!(f, "scale ({scale}) must be greater than zero")
            }
            Self::ScaleNotFinite { scale } => {
                write!(f, "non-finite scale: {scale}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::{ks_test, linspace};
    use crate::test_basic_impls;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!(f64, Pareto, Pareto::new(1.0, 0.2).unwrap());

    #[test]
    fn new() {
        let par = Pareto::new(1.0, 2.0).unwrap();
        assert::close(par.shape, 1.0, TOL);
        assert::close(par.scale, 2.0, TOL);
    }

    #[test]
    fn ln_pdf() {
        let par = Pareto::new(1.0, 1.0).unwrap();
        let xs: Vec<f64> = linspace(1.0, 10.0, 50);
        let known_values = vec![
            0.0,
            -0.337_245_424_871_59,
            -0.625_744_642_560_68,
            -0.877_826_084_351_41,
            -1.101_661_916_759_38,
            -1.302_948_968_318_75,
            -1.485_817_380_238_02,
            -1.653_357_146_368_94,
            -1.807_940_494_972_23,
            -1.951_428_304_689_91,
            -2.085_307_270_040_13,
            -2.210_783_951_306_98,
            -2.328_851_014_475_36,
            -2.440_334_980_491_83,
            -2.545_931_351_625_78,
            -2.646_230_918_996_72,
            -2.741_739_781_588_52,
            -2.832_894_798_581_16,
            -2.920_075_670_730_88,
            -3.003_614_496_483_47,
            -3.083_803_410_887_23,
            -3.160_900_751_121_7,
            -3.235_136_077_034_7,
            -3.306_714_292_737_87,
            -3.375_819_055_751_19,
            -3.442_615_616_554_89,
            -3.507_253_199_065_22,
            -3.569_867_008_315_31,
            -3.630_579_933_276_5,
            -3.689_503_998_737_13,
            -3.746_741_609_348_44,
            -3.802_386_620_547_03,
            -3.856_525_264_483_47,
            -3.909_236_953_894_2,
            -3.960_594_982_729_58,
            -4.010_667_139_052_23,
            -4.059_516_243_066_38,
            -4.107_200_620_991_9,
            -4.153_774_523_749_48,
            -4.199_288_497_994_71,
            -4.243_789_715_864_35,
            -4.287_322_268_828_26,
            -4.329_927_430_236,
            -4.371_643_890_476_82,
            -4.412_507_968_111_23,
            -4.452_553_799_861_44,
            -4.491_813_511_951_22,
            -4.530_317_374_949_86,
            -4.568_093_943_990_27,
            -4.605_170_185_988_09,
        ];

        let generated_values: Vec<f64> =
            xs.iter().map(|x| par.ln_f(x)).collect();
        assert::close(known_values, generated_values, TOL);
    }

    #[test]
    fn cdf_shape_2() {
        let par = Pareto::new(2.0, 1.0).unwrap();
        let xs: Vec<f64> = linspace(1.0, 10.0, 50);
        let known_values = vec![
            0.,
            0.286_266_349_583_83,
            0.465_137_001_559_37,
            0.584_314_404_432_13,
            0.667_681_660_899_65,
            0.728_270_710_728_84,
            0.773_682_722_216_99,
            0.808_593_75,
            0.836_008_469_366_85,
            0.857_928_994_082_84,
            0.875_731_069_820_4,
            0.890_385_317_750_18,
            0.902_592_397_257_5,
            0.912_868_340_833_21,
            0.9216,
            0.929_081_994_328_92,
            0.935_541_893_742_11,
            0.941_157_729_634_35,
            0.946_070_393_746_77,
            0.950_392_561_983_47,
            0.954_215_213_287_31,
            0.957_612_456_747_4,
            0.960_645_150_715_47,
            0.963_363_647_460_94,
            0.965_809_896_760_41,
            0.968_019_074_005_01,
            0.970_020_851_802_37,
            0.971_840_401_576_28,
            0.973_499_188_750_68,
            0.975_015_608_740_89,
            0.976_405_499_159_8,
            0.977_682_555_026_77,
            0.978_858_667_418_05,
            0.979_944_201_276_35,
            0.980_948_224_558_62,
            0.981_878_698_224_85,
            0.982_742_634_533_42,
            0.983_546_229_544_15,
            0.984_294_974_522_67,
            0.984_993_75,
            0.985_646_905_506_3,
            0.986_258_327_419_24,
            0.986_831_496_909_43,
            0.987_369_539_601_04,
            0.987_875_268_274_21,
            0.988_351_219_701_53,
            0.988_799_686_521_84,
            0.989_222_744_900_89,
            0.989_622_278_603_57,
            0.99,
        ];
        let generated_values: Vec<f64> =
            xs.iter().map(|x| par.cdf(x)).collect();
        assert::close(known_values, generated_values, TOL);
    }

    #[test]
    fn cdf_scale_2() {
        let par = Pareto::new(1.0, 2.0).unwrap();
        let xs: Vec<f64> = linspace(2.0, 10.0, 50);
        let known_values = vec![
            0.,
            0.075_471_698_113_21,
            0.140_350_877_192_98,
            0.196_721_311_475_41,
            0.246_153_846_153_85,
            0.289_855_072_463_77,
            0.328_767_123_287_67,
            0.363_636_363_636_36,
            0.395_061_728_395_06,
            0.423_529_411_764_71,
            0.449_438_202_247_19,
            0.473_118_279_569_89,
            0.494_845_360_824_74,
            0.514_851_485_148_51,
            0.533_333_333_333_33,
            0.550_458_715_596_33,
            0.566_371_681_415_93,
            0.581_196_581_196_58,
            0.595_041_322_314_05,
            0.608,
            0.620_155_038_759_69,
            0.631_578_947_368_42,
            0.642_335_766_423_36,
            0.652_482_269_503_55,
            0.662_068_965_517_24,
            0.671_140_939_597_32,
            0.679_738_562_091_5,
            0.687_898_089_171_97,
            0.695_652_173_913_04,
            0.703_030_303_030_3,
            0.710_059_171_597_63,
            0.716_763_005_780_35,
            0.723_163_841_807_91,
            0.729_281_767_955_8,
            0.735_135_135_135_14,
            0.740_740_740_740_74,
            0.746_113_989_637_31,
            0.751_269_035_532_99,
            0.756_218_905_472_64,
            0.760_975_609_756_1,
            0.765_550_239_234_45,
            0.769_953_051_643_19,
            0.774_193_548_387_1,
            0.778_280_542_986_43,
            0.782_222_222_222_22,
            0.786_026_200_873_36,
            0.789_699_570_815_45,
            0.793_248_945_147_68,
            0.796_680_497_925_31,
            0.8,
        ];
        let generated_values: Vec<f64> =
            xs.iter().map(|x| par.cdf(x)).collect();
        assert::close(known_values, generated_values, TOL);
    }

    #[test]
    fn mean() {
        let m1: f64 = Pareto::new(0.5, 1.0).unwrap().mean().unwrap();
        let m2: f64 = Pareto::new(1.0, 1.0).unwrap().mean().unwrap();
        let m3: f64 = Pareto::new(3.0, 1.0).unwrap().mean().unwrap();
        let m4: f64 = Pareto::new(2.0, 0.1).unwrap().mean().unwrap();
        assert::close(m1, f64::INFINITY, TOL);
        assert::close(m2, f64::INFINITY, TOL);
        assert::close(m3, 1.5, TOL);
        assert::close(m4, 0.2, TOL);
    }

    #[test]
    fn mode() {
        let m1: f64 = Pareto::new(2.0, 2.0).unwrap().mode().unwrap();
        let m2: f64 = Pareto::new(1.0, 2.0).unwrap().mode().unwrap();
        let m3: f64 = Pareto::new(5.0, 1.0).unwrap().mode().unwrap();
        assert::close(m1, 2.0, TOL);
        assert::close(m2, 2.0, TOL);
        assert::close(m3, 1.0, TOL);
    }

    #[test]
    fn variance() {
        let v1: f64 = Pareto::new(1.0, 1.0).unwrap().variance().unwrap();
        let v2: f64 = Pareto::new(2.0, 1.0).unwrap().variance().unwrap();
        let v3: f64 = Pareto::new(3.0, 1.0).unwrap().variance().unwrap();
        let v4: f64 = Pareto::new(3.0, 2.0).unwrap().variance().unwrap();

        assert::close(v1, f64::INFINITY, TOL);
        assert::close(v2, f64::INFINITY, TOL);
        assert::close(v3, 0.75, TOL);
        assert::close(v4, 3.0, TOL);
    }

    #[test]
    fn skewness() {
        let s1 = Pareto::new(1.0, 1.0).unwrap().skewness();
        let s2 = Pareto::new(3.0, 1.0).unwrap().skewness();
        let s3 = Pareto::new(4.0, 1.0).unwrap().skewness();
        let s4 = Pareto::new(4.0, 2.0).unwrap().skewness();

        assert!(s1.is_none());
        assert!(s2.is_none());
        assert::close(s3.unwrap(), 7.071_067_811_865_475_5, TOL);
        assert::close(s4.unwrap(), 7.071_067_811_865_475_5, TOL);
    }

    #[test]
    fn kurtosis() {
        let k1 = Pareto::new(1.0, 1.0).unwrap().kurtosis();
        let k2 = Pareto::new(4.0, 1.0).unwrap().kurtosis();
        let k3 = Pareto::new(5.0, 1.0).unwrap().kurtosis();
        let k4 = Pareto::new(5.0, 2.0).unwrap().kurtosis();

        assert!(k1.is_none());
        assert!(k2.is_none());
        assert::close(k3.unwrap(), 70.8, TOL);
        assert::close(k4.unwrap(), 70.8, TOL);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::rng();
        let par = Pareto::new(1.2, 3.4).unwrap();
        let cdf = |x: f64| par.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = par.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL { acc + 1 } else { acc }
        });

        assert!(passes > 0);
    }

    use crate::test_scalable_cdf;
    use crate::test_scalable_density;
    use crate::test_scalable_entropy;
    use crate::test_scalable_method;

    test_scalable_method!(Pareto::new(2.0, 4.0).unwrap(), mean);
    test_scalable_method!(Pareto::new(2.0, 4.0).unwrap(), variance);
    test_scalable_method!(Pareto::new(2.0, 4.0).unwrap(), skewness);
    test_scalable_method!(Pareto::new(2.0, 4.0).unwrap(), kurtosis);
    test_scalable_density!(Pareto::new(2.0, 4.0).unwrap());
    test_scalable_entropy!(Pareto::new(2.0, 4.0).unwrap());
    test_scalable_cdf!(Pareto::new(2.0, 4.0).unwrap());
}
