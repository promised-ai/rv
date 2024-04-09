const MAX_ITER: usize = 500;

const BESSI0_COEFFS_A: [f64; 30] = [
    -4.415_341_646_479_339_5E-18,
    3.330_794_518_822_238_4E-17,
    -2.431_279_846_547_955E-16,
    1.715_391_285_555_133E-15,
    -1.168_533_287_799_345_1E-14,
    7.676_185_498_604_936E-14,
    -4.856_446_783_111_929E-13,
    2.955_052_663_129_64E-12,
    -1.726_826_291_441_556E-11,
    9.675_809_035_373_237E-11,
    -5.189_795_601_635_263E-10,
    2.659_823_724_682_386_6E-9,
    -1.300_025_009_986_248E-8,
    6.046_995_022_541_919E-8,
    -2.670_793_853_940_612E-7,
    1.117_387_539_120_103_7E-6,
    -4.416_738_358_458_750_5E-6,
    1.644_844_807_072_889_6E-5,
    -5.754_195_010_082_104E-5,
    1.885_028_850_958_416_5E-4,
    -5.763_755_745_385_824E-4,
    1.639_475_616_941_335_7E-3,
    -4.324_309_995_050_576E-3,
    1.054_646_039_459_499_8E-2,
    -2.373_741_480_589_947E-2,
    4.930_528_423_967_071E-2,
    -9.490_109_704_804_764E-2,
    1.716_209_015_222_087_7E-1,
    -3.046_826_723_431_984E-1,
    6.767_952_744_094_761E-1,
];

const BESSI0_COEFFS_B: [f64; 25] = [
    -7.233_180_487_874_754E-18,
    -4.830_504_485_944_182E-18,
    4.465_621_420_296_76E-17,
    3.461_222_867_697_461E-17,
    -2.827_623_980_516_583_6E-16,
    -3.425_485_619_677_219E-16,
    1.772_560_133_056_526_3E-15,
    3.811_680_669_352_622_4E-15,
    -9.554_846_698_828_307E-15,
    -4.150_569_347_287_222E-14,
    1.540_086_217_521_41E-14,
    3.852_778_382_742_142_6E-13,
    7.180_124_451_383_666E-13,
    -1.794_178_531_506_806_2E-12,
    -1.321_581_184_044_771_3E-11,
    -3.149_916_527_963_241_6E-11,
    1.188_914_710_784_643_9E-11,
    4.940_602_388_224_97E-10,
    3.396_232_025_708_386_5E-9,
    2.266_668_990_498_178E-8,
    2.048_918_589_469_063_8E-7,
    2.891_370_520_834_756_7E-6,
    6.889_758_346_916_825E-5,
    3.369_116_478_255_694_3E-3,
    8.044_904_110_141_088E-1,
];

const BESSI1_COEFFS_A: [f64; 29] = [
    2.777_914_112_761_046_4E-18,
    -2.111_421_214_358_166E-17,
    1.553_631_957_736_200_5E-16,
    -1.105_596_947_735_386_2E-15,
    7.600_684_294_735_408E-15,
    -5.042_185_504_727_912E-14,
    3.223_793_365_945_575E-13,
    -1.983_974_397_764_943_6E-12,
    1.173_618_629_889_090_1E-11,
    -6.663_489_723_502_027E-11,
    3.625_590_281_552_117E-10,
    -1.887_249_751_722_829_4E-9,
    9.381_537_386_495_773E-9,
    -4.445_059_128_796_328E-8,
    2.003_294_753_552_135_3E-7,
    -8.568_720_264_695_455E-7,
    3.470_251_308_137_678_5E-6,
    -1.327_316_365_603_943_6E-5,
    4.781_565_107_550_054E-5,
    -1.617_608_158_258_967_4E-4,
    5.122_859_561_685_758E-4,
    -1.513_572_450_631_253_2E-3,
    4.156_422_944_312_888E-3,
    -1.056_408_489_462_619_7E-2,
    2.472_644_903_062_651_6E-2,
    -5.294_598_120_809_499E-2,
    1.026_436_586_898_471E-1,
    -1.764_165_183_578_340_6E-1,
    2.525_871_864_436_336_5E-1,
];

#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
const BESSI1_COEFFS_B: [f64; 25] = [
    7.51729631084210481353E-18,
    4.41434832307170791151E-18,
    -4.65030536848935832153E-17,
    -3.20952592199342395980E-17,
    2.96262899764595013876E-16,
    3.30820231092092828324E-16,
    -1.88035477551078244854E-15,
    -3.81440307243700780478E-15,
    1.04202769841288027642E-14,
    4.27244001671195135429E-14,
    -2.10154184277266431302E-14,
    -4.08355111109219731823E-13,
    -7.19855177624590851209E-13,
    2.03562854414708950722E-12,
    1.41258074366137813316E-11,
    3.25260358301548823856E-11,
    -1.89749581235054123450E-11,
    -5.58974346219658380687E-10,
    -3.83538038596423702205E-9,
    -2.63146884688951950684E-8,
    -2.51223623787020892529E-7,
    -3.88256480887769039346E-6,
    -1.10588938762623716291E-4,
    -9.76109749136146840777E-3,
    7.78576235018280120474E-1,
];

fn chbevl(x: f64, coeffs: &[f64]) -> f64 {
    let mut b0 = coeffs[0];
    let mut b1 = 0.0;
    let mut b2 = 0.0;

    coeffs.iter().skip(1).for_each(|c| {
        b2 = b1;
        b1 = b0;
        b0 = x.mul_add(b1, *c) - b2;
    });

    0.5 * (b0 - b2)
}

/// Modified Bessel function, I<sub>0</sub>(x)
pub fn i0(x: f64) -> f64 {
    let ax = x.abs();

    if ax <= 8.0 {
        let y = ax.mul_add(0.5, -2.0);
        ax.exp() * chbevl(y, &BESSI0_COEFFS_A)
    } else {
        ax.exp() * chbevl(32.0_f64.mul_add(ax.recip(), -2.0), &BESSI0_COEFFS_B)
            / ax.sqrt()
    }
}

/// Modified Bessel function, I<sub>1</sub>(x)
pub fn i1(x: f64) -> f64 {
    let z = x.abs();
    let res = if z <= 8.0 {
        let y = z.mul_add(0.5, -2.0);
        chbevl(y, &BESSI1_COEFFS_A) * z * z.exp()
    } else {
        z.exp() * chbevl(32.0_f64.mul_add(x.recip(), -2.0), &BESSI1_COEFFS_B)
            / z.sqrt()
    };

    res * x.signum()
}

/// An encounterable error when computing Bessel's I function
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BesselIvError {
    /// The order, v, must be an integer if z is negative.
    OrderNotIntegerForNegativeZ,
    /// Arguments would lead to an overflow
    Overflow,
    /// Failed to converge
    FailedToConverge,
    /// Precision Error
    PrecisionLoss,
    /// Input parameters are outside the domain
    Domain,
}

impl std::error::Error for BesselIvError {}

impl std::fmt::Display for BesselIvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OrderNotIntegerForNegativeZ => {
                write!(f, "If z is negative, the order, v, must be an integer.")
            }
            Self::Overflow => {
                write!(f, "Arguments would cause an overflow.")
            }
            Self::FailedToConverge => {
                write!(f, "Failed to converge.")
            }
            Self::PrecisionLoss => {
                write!(f, "Precision loss.")
            }
            Self::Domain => {
                write!(f, "Parameters are outside domain.")
            }
        }
    }
}

/// Modified Bessel function of the first kind of real order
pub fn bessel_iv(v: f64, z: f64) -> Result<f64, BesselIvError> {
    if v.is_nan() || z.is_nan() {
        return Ok(f64::NAN);
    }
    let (v, t) = {
        let t = v.floor();
        if v < 0.0 && (t - v).abs() < f64::EPSILON {
            (-v, -t)
        } else {
            (v, t)
        }
    };

    let sign: f64 = if z < 0.0 {
        // Return error if v is not an integer if x < 0
        if (t - v).abs() > f64::EPSILON {
            return Err(BesselIvError::OrderNotIntegerForNegativeZ);
        }

        if 2.0_f64.mul_add(-(v / 2.0).floor(), v) > f64::EPSILON {
            -1.0
        } else {
            1.0
        }
    } else {
        1.0
    };

    if z == 0.0 {
        if v == 0.0 {
            return Ok(1.0);
        } else if v < 0.0 {
            return Err(BesselIvError::Overflow);
        } else {
            return Ok(0.0);
        }
    }

    let az = z.abs();
    let res: f64 = if v.abs() > 50.0 {
        // Use asymptotic expansion for large orders
        bessel_ikv_asymptotic_uniform(v, az)?.0
    } else {
        // Use Temme's method for small orders
        bessel_ikv_temme(v, az)?.0
    };

    Ok(res * sign)
}

const N_UFACTORS: usize = 11;
const N_UFACTOR_TERMS: usize = 31;
const ASYMPTOTIC_UFACTORS: [[f64; N_UFACTOR_TERMS]; N_UFACTORS] = [
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.208_333_333_333_333_34,
        0.0,
        0.125,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.334_201_388_888_888_9,
        0.0,
        -0.401_041_666_666_666_7,
        0.0,
        0.070_312_5,
        0.0,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.025_812_596_450_617_3,
        0.0,
        1.846_462_673_611_111_2,
        0.0,
        -0.891_210_937_5,
        0.0,
        0.073_242_187_5,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.669_584_423_426_247,
        0.0,
        -11.207_002_616_222_995,
        0.0,
        8.789_123_535_156_25,
        0.0,
        -2.364_086_914_062_5,
        0.0,
        0.112_152_099_609_375,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -28.212_072_558_200_244,
        0.0,
        84.636_217_674_600_74,
        0.0,
        -91.818_241_543_240_03,
        0.0,
        42.534_998_745_388_46,
        0.0,
        -7.368_794_359_479_631,
        0.0,
        0.227_108_001_708_984_38,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        212.570_130_039_217_1,
        0.0,
        -765.252_468_141_181_6,
        0.0,
        1_059.990_452_527_999_9,
        0.0,
        -699.579_627_376_132_7,
        0.0,
        218.190_511_744_211_6,
        0.0,
        -26.491_430_486_951_554,
        0.0,
        0.572_501_420_974_731_4,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1_919.457_662_318_406_8,
        0.0,
        8_061.722_181_737_308,
        0.0,
        -13_586.550_006_434_136,
        0.0,
        11_655.393_336_864_536,
        0.0,
        -5_305.646_978_613_405,
        0.0,
        1_200.902_913_216_352_5,
        0.0,
        -108.090_919_788_394_64,
        0.0,
        1.727_727_502_584_457_4,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        20_204.291_330_966_15,
        0.0,
        -96_980.598_388_637_5,
        0.0,
        192_547.001_232_531_5,
        0.0,
        -203_400.177_280_415_55,
        0.0,
        122_200.464_983_017_47,
        0.0,
        -41_192.654_968_897_56,
        0.0,
        7_109.514_302_489_364,
        0.0,
        -493.915_304_773_088,
        0.0,
        6.074_042_001_273_483,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        -242_919.187_900_551_33,
        0.0,
        1_311_763.614_662_977,
        0.0,
        -2_998_015.918_538_106,
        0.0,
        3_763_271.297_656_404,
        0.0,
        -2_813_563.226_586_534,
        0.0,
        1_268_365.273_321_624_8,
        0.0,
        -331_645.172_484_563_6,
        0.0,
        45_218.768_981_362_74,
        0.0,
        -2_499.830_481_811_209,
        0.0,
        24.380_529_699_556_064,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        3_284_469.853_072_037_5,
        0.0,
        -19_706_819.118_432_22,
        0.0,
        50_952_602.492_664_63,
        0.0,
        -74_105_148.211_532_64,
        0.0,
        66_344_512.274_729_03,
        0.0,
        -37_567_176.660_763_35,
        0.0,
        13_288_767.166_421_82,
        0.0,
        -2_785_618.128_086_455,
        0.0,
        308_186.404_612_662_45,
        0.0,
        -13_886.089_753_717_039,
        0.0,
        110.017_140_269_246_74,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
];

///Compute Iv, Kv from (AMS5 9.7.7 + 9.7.8), asymptotic expansion for large v
///
/// Heavily inspired by
/// https://github.com/scipy/scipy/blob/1984f97749a355a6767cea55cad5d1dc6977ad5f/scipy/special/cephes/scipy_iv.c#L249
fn bessel_ikv_asymptotic_uniform(
    v: f64,
    x: f64,
) -> Result<(f64, f64), BesselIvError> {
    use std::f64::consts::PI;
    let (v, sign) = (v.abs(), v.signum());

    let z = x / v;
    let t = z.mul_add(z, 1.0).sqrt().recip();
    let t2 = t * t;
    let eta = z.mul_add(z, 1.0).sqrt() + (z / (1.0 + t.recip())).ln();

    let i_prefactor = (t / (2.0 * PI * v)).sqrt() * (v * eta).exp();
    let mut i_sum = 1.0;

    let k_prefactor = (PI * t / (2.0 * v)).sqrt() * (-v * eta).exp();
    let mut k_sum = 1.0;

    let mut divisor = v;
    let mut term = 0.0;

    for (n, item) in ASYMPTOTIC_UFACTORS
        .iter()
        .enumerate()
        .take(N_UFACTORS)
        .skip(1)
    {
        term = 0.0;
        for k in
            ((N_UFACTOR_TERMS - 1 - 3 * n)..(N_UFACTOR_TERMS - n)).step_by(2)
        {
            term *= t2;
            term += item[k];
        }
        for _k in (1..n).step_by(2) {
            term *= t2;
        }
        if n % 2 == 1 {
            term *= t;
        }

        term /= divisor;
        i_sum += term;
        k_sum += if n % 2 == 0 { term } else { -term };

        if term.abs() < f64::EPSILON {
            break;
        }
        divisor *= v;
    }

    // check convergence
    if term.abs() > 1E-3 * i_sum.abs() {
        Err(BesselIvError::FailedToConverge)
    } else if term.abs() > f64::EPSILON * i_sum.abs() {
        Err(BesselIvError::PrecisionLoss)
    } else {
        let k_value = k_prefactor * k_sum;
        let i_value = if sign > 0.0 {
            i_prefactor * i_sum
        } else {
            i_prefactor.mul_add(
                i_sum,
                (2.0 / PI) * (PI * v).sin() * k_prefactor * k_sum,
            )
        };
        Ok((i_value, k_value))
    }
}

/// Compute I(v, x) and K(v, x) simultaneously by Temme's method, see
/// Temme, Journal of Computational Physics, vol 19, 324 (1975)
/// Heavily inspired by
/// https://github.com/scipy/scipy/blob/1984f97749a355a6767cea55cad5d1dc6977ad5f/scipy/special/cephes/scipy_iv.c#L532
#[allow(clippy::many_single_char_names)]
pub(crate) fn bessel_ikv_temme(
    v: f64,
    x: f64,
) -> Result<(f64, f64), BesselIvError> {
    use std::f64::consts::PI;
    let (v, reflect) = if v < 0.0 { (-v, true) } else { (v, false) };

    let n = v.round();
    let u = v - n;
    let n = n as isize;

    if x < 0.0 {
        return Err(BesselIvError::Domain);
    } else if x == 0.0 {
        return Err(BesselIvError::Overflow);
    }

    let w = x.recip();
    let (ku, ku_1) = if x <= 2.0 {
        temme_ik_series(u, x)?
    } else {
        cf2_ik(u, x)?
    };

    let mut prev = ku;
    let mut current = ku_1;
    for k in 1..=n {
        let kf = k as f64;
        let next = 2.0 * (u + kf) * current / x + prev;
        prev = current;
        current = next;
    }

    let kv = prev;
    let kv1 = current;

    let lim = (4.0_f64.mul_add(v * v, 10.0) / (8.0 * x)).powi(3) / 24.0;

    let iv = if lim < 10.0 * f64::EPSILON && x > 100.0 {
        bessel_iv_asymptotic(v, x)?
    } else {
        let fv = cf1_ik(v, x)?;
        w / kv.mul_add(fv, kv1)
    };

    if reflect {
        let z = u + ((n % 2) as f64);
        Ok(((2.0 / PI).mul_add((PI * z).sin() * kv, iv), kv))
    } else {
        Ok((iv, kv))
    }
}

/// Modified Bessel functions of the first and second kind of fractional order
///
/// Calculate K(v, x) and K(v+1, x) by method analogous to
/// Temme, Journal of Computational Physics, vol 21, 343 (1976)
#[allow(clippy::many_single_char_names)]
fn temme_ik_series(v: f64, x: f64) -> Result<(f64, f64), BesselIvError> {
    use crate::consts::EULER_MASCERONI;
    use crate::misc::gammafn;
    use std::f64::consts::PI;
    /*
     * |x| <= 2, Temme series converge rapidly
     * |x| > 2, the larger the |x|, the slower the convergence
     */
    debug_assert!(x.abs() <= 2.0);
    debug_assert!(v.abs() <= 0.5);

    let gp = gammafn(v + 1.0) - 1.0;
    let gm = gammafn(1.0 - v) - 1.0;

    let a = (x / 2.0).ln();
    let b = (v * a).exp();
    let sigma = -a * v;
    let c = if v.abs() < 2.0 * f64::EPSILON {
        1.0
    } else {
        (PI * v).sin() / (PI * v)
    };
    let d = if sigma.abs() < f64::EPSILON {
        1.0
    } else {
        sigma.sinh() / sigma
    };
    let gamma1 = if v.abs() < f64::EPSILON {
        -EULER_MASCERONI
    } else {
        (0.5 / v) * (gp - gm) * c
    };
    let gamma2 = (2.0 + gp + gm) * c / 2.0;

    let mut p = (gp + 1.0) / (2.0 * b);
    let mut q = (gm + 1.0) * b / 2.0;
    let mut f = d.mul_add(-a * gamma2, sigma.cosh() * gamma1) / c;
    let mut h = p;
    let mut coef = 1.0;
    let mut sum = coef * f;
    let mut sum1 = coef * h;

    for k in 1..MAX_ITER {
        let kf = k as f64;
        f = kf.mul_add(f, p + q) / kf.mul_add(kf, -v * v);
        p /= kf - v;
        q /= kf + v;
        h = kf.mul_add(-f, p);
        coef *= x * x / (4.0 * kf);
        sum += coef * f;
        sum1 += coef * h;

        if (coef * f).abs() < sum.abs() * f64::EPSILON {
            return Ok((sum, 2.0 * sum1 / x));
        }
    }

    Err(BesselIvError::FailedToConverge)
}

/// Calculate K(v, x) and K(v+1, x) by evaluating continued fraction
/// z1 / z0 = U(v+1.5, 2v+1, 2x) / U(v+0.5, 2v+1, 2x), see
/// Thompson and Barnett, Computer Physics Communications, vol 47, 245 (1987)
#[allow(clippy::many_single_char_names)]
fn cf2_ik(v: f64, x: f64) -> Result<(f64, f64), BesselIvError> {
    use std::f64::consts::PI;
    /*
     * Steed's algorithm, see Thompson and Barnett,
     * Journal of Computational Physics, vol 64, 490 (1986)
     */
    debug_assert!(x.abs() > 1.0);

    let mut a = v.mul_add(v, -0.25);
    let mut b = 2.0 * (x + 1.0);
    let mut d = b.recip();

    let mut delta = d;
    let mut f = d;
    let mut prev = 0.0;
    let mut cur = 1.0;
    let mut q = -a;
    let mut c = -a;
    let mut s = q.mul_add(delta, 1.0);

    for k in 2..MAX_ITER {
        let kf = k as f64;
        a -= 2.0 * (kf - 1.0);
        b += 2.0;
        d = a.mul_add(d, b).recip();
        delta *= b.mul_add(d, -1.0);
        f += delta;

        let t = (b - 2.0).mul_add(-cur, prev) / a;
        prev = cur;
        cur = t;
        c *= -a / kf;
        q += c * t;
        s += q * delta;

        if (q * delta).abs() < s.abs() * f64::EPSILON / 2.0 {
            let kv = (PI / (2.0 * x)).sqrt() * (-x).exp() / s;
            let kv1 = kv * v.mul_add(v, -0.25).mul_add(f, 0.5 + v + x) / x;
            return Ok((kv, kv1));
        }
    }
    Err(BesselIvError::FailedToConverge)
}

/// Evaluate continued fraction fv = I_(v+1) / I_v, derived from
/// Abramowitz and Stegun, Handbook of Mathematical Functions, 1972, 9.1.73 */
#[allow(clippy::many_single_char_names)]
fn cf1_ik(v: f64, x: f64) -> Result<f64, BesselIvError> {
    /*
     * |x| <= |v|, CF1_ik converges rapidly
     * |x| > |v|, CF1_ik needs O(|x|) iterations to converge
     */

    /*
     * modified Lentz's method, see
     * Lentz, Applied Optics, vol 15, 668 (1976)
     */

    const TOL: f64 = f64::EPSILON;
    let tiny: f64 = f64::MAX.sqrt().recip();
    let mut c = tiny;
    let mut f = tiny;
    let mut d = 0.0;

    for k in 1..MAX_ITER {
        let kf = k as f64;
        let a = 1.0;
        let b = 2.0 * (v + kf) / x;
        c = b + a / c;
        d = a.mul_add(d, b);
        if c == 0.0 {
            c = tiny;
        }
        if d == 0.0 {
            d = tiny;
        }
        d = d.recip();
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() <= TOL {
            return Ok(f);
        }
    }

    Err(BesselIvError::FailedToConverge)
}

/// Compute I_v from (AMS5 9.7.1), asymptotic expansion for large |z|
///  I_v ~ exp(x)/sqrt(2 pi x) ( 1 + (4*v*v-1)/8x + (4*v*v-1)(4*v*v-9)/8x/2! + ...)
///  Heavily inspired by
///  https://github.com/scipy/scipy/blob/1984f97749a355a6767cea55cad5d1dc6977ad5f/scipy/special/cephes/scipy_iv.c#L145
fn bessel_iv_asymptotic(v: f64, x: f64) -> Result<f64, BesselIvError> {
    let prefactor = x.exp() / (2.0 * std::f64::consts::PI * x).sqrt();

    if prefactor.is_infinite() {
        Ok(x)
    } else {
        let mu = 4.0 * v * v;
        let mut sum: f64 = 1.0;
        let mut term: f64 = 1.0;
        let mut k: usize = 1;

        while term.abs() > f64::EPSILON * sum.abs() {
            let kf = k as f64;
            let factor = 2.0_f64
                .mul_add(kf, -1.0)
                .mul_add(-(2.0_f64.mul_add(kf, -1.0)), mu)
                / (8.0 * x)
                / kf;
            if k > 100 {
                return Err(BesselIvError::FailedToConverge);
            }
            term *= -factor;
            sum += term;
            k += 1;
        }
        Ok(sum * prefactor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn chbevl_val() {
        assert::close(
            chbevl(2.23525, &BESSI0_COEFFS_A),
            0.139_251_866_282_289,
            TOL,
        );
    }

    #[test]
    fn bessi0_small() {
        assert::close(i0(3.74), 9.041_496_849_012_773, TOL);
        assert::close(i0(-3.74), 9.041_496_849_012_773, TOL);
        assert::close(i0(8.0), 427.564_115_721_804_74, TOL);
    }

    #[test]
    fn bessi0_large() {
        assert::close(i0(8.1), 469.500_606_710_121_4, TOL);
        assert::close(i0(10.0), 2_815.716_628_466_254, TOL);
    }

    #[test]
    fn bessi1_small() {
        assert::close(i1(3.74), 7.709_894_215_253_694, TOL);
        assert::close(i1(-3.74), -7.709_894_215_253_694, TOL);
        assert::close(i1(0.0024), 0.001_200_000_864_000_207_2, TOL);
        assert::close(i1(8.0), 399.873_136_782_559_9, TOL);
    }

    #[test]
    fn bessi1_large() {
        assert::close(i1(8.1), 439.484_308_910_358_44, TOL);
        assert::close(i1(10.0), 2_670.988_303_701_255, TOL);
    }

    #[test]
    fn bessel_iv_basic_limits() {
        assert::close(bessel_iv(0.0, 0.0).unwrap(), 1.0, TOL);
        assert::close(bessel_iv(1.0, 0.0).unwrap(), 0.0, TOL);
    }

    #[test]
    fn bessel_iv_high_order() {
        assert::close(
            bessel_iv(60.0, 40.0).unwrap(),
            0.071_856_419_684_526_32,
            TOL,
        );
    }

    #[test]
    fn bessel_iv_low_order() {
        assert::close(
            bessel_iv(0.0, 1.0).unwrap(),
            1.266_065_877_752_008_4,
            TOL,
        );
        assert::close(
            bessel_iv(0.0, 10.0).unwrap(),
            2_815.716_628_466_254_4,
            TOL,
        );

        assert::close(
            bessel_iv(1.0, 10.0).unwrap(),
            2_670.988_303_701_254,
            TOL,
        );
        assert::close(
            bessel_iv(20.0, 10.0).unwrap(),
            0.000_125_079_973_564_494_78,
            TOL,
        );
    }

    #[test]
    fn cf1_ik_checks() {
        assert::close(cf1_ik(0.0, 10.0).unwrap(), 0.948_599_825_954_845_8, TOL);
        assert::close(
            cf1_ik(10.0, 10.0).unwrap(),
            0.389_913_883_928_382_94,
            TOL,
        );
        assert::close(
            cf1_ik(60.0, 5.0).unwrap(),
            0.040_916_097_908_833_04,
            TOL,
        );
    }

    #[test]
    fn cf2_ik_checks() {
        let (k1, k2) = cf2_ik(0.0, 2.0).unwrap();
        assert::close(k1, 0.113_893_872_749_533_53, TOL);
        assert::close(k2, 0.139_865_881_816_522_54, TOL);

        let (k1, k2) = cf2_ik(5.0, 5.0).unwrap();
        assert::close(k1, 3.270_627_371_203_162e-2, TOL);
        assert::close(k2, 8.067_161_323_456_37e-2, TOL);
    }

    #[test]
    fn temme_ik_series_checks() {
        let res = temme_ik_series(0.0, 0.0);
        assert!(res.is_err());

        let (k1, k2) = temme_ik_series(0.0, 1.0).unwrap();
        assert::close(k1, 4.210_244_382_407_083_4e-1, TOL);
        assert::close(k2, 6.019_072_301_972_346e-1, TOL);

        let (k1, k2) = temme_ik_series(0.5, 2.0).unwrap();
        assert::close(k1, 1.199_377_719_680_612_3e-1, TOL);
        assert::close(k2, 1.799_066_579_520_924_3e-1, TOL);
    }

    #[test]
    fn bessel_ikv_temme_checks() {
        let (i, k) = bessel_ikv_temme(0.0, 1.0).unwrap();
        assert::close(i, 1.266_065_877_752_008_4, TOL);
        assert::close(k, 0.421_024_438_240_708_34, TOL);

        let (i, k) = bessel_ikv_temme(5.0, 2.0).unwrap();
        assert::close(i, 0.009_825_679_323_131_702, TOL);
        assert::close(k, 9.431_049_100_596_468, TOL);

        let (i, k) = bessel_ikv_temme(20.0, 2.0).unwrap();
        assert::close(i, 4.310_560_576_109_548E-19, TOL);
        assert::close(k, 5.770_856_852_700_242_4E16, TOL);

        let (i, k) = bessel_ikv_temme(20.0, 2.0).unwrap();
        assert::close(i, 4.310_560_576_109_548E-19, TOL);
        assert::close(k, 5.770_856_852_700_242_4E16, TOL);

        let (i, k) = bessel_ikv_temme(1.0, 10.0).unwrap();
        assert::close(i, 2_670.988_303_701_254, TOL);
        assert::close(k, 1.864_877_345_382_558_5E-5, TOL);
    }

    #[test]
    fn bessel_ikv_asymptotic_uniform_checks() {
        let (i, k) = bessel_ikv_asymptotic_uniform(60.0, 40.0).unwrap();
        assert::close(i, 7.185_641_968_452_632e-2, TOL);
        assert::close(k, 9.649_278_749_222_319e-2, TOL);

        let (i, k) = bessel_ikv_asymptotic_uniform(100.0, 60.0).unwrap();
        assert::close(i, 2.883_277_090_649_164e-7, TOL);
        assert::close(k, 1.487_001_275_494_647_4e4, TOL);
    }
}
