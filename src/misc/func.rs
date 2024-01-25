use crate::consts::{LN_2PI, LN_PI};
use rand::distributions::Open01;
use rand::Rng;
use special::Gamma;
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::ops::AddAssign;

/// Convert a Vector to a printable string
///
/// # Example
///
/// ```rust
/// # use rv::misc::vec_to_string;
/// let xs: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
///
/// assert_eq!(vec_to_string(&xs, 6).as_str(), "[0, 1, 2, 3, 4, 5]");
/// assert_eq!(vec_to_string(&xs, 5).as_str(), "[0, 1, 2, 3, ... , 5]");
///
/// ```
pub fn vec_to_string<T: Debug>(xs: &[T], max_entries: usize) -> String {
    let mut out = String::new();
    out += "[";
    let n = xs.len();
    xs.iter().enumerate().for_each(|(i, x)| {
        let to_push = if i < max_entries - 1 {
            format!("{:?}, ", x)
        } else if i == (max_entries - 1) && n > max_entries {
            String::from("... , ")
        } else {
            format!("{:?}]", x)
        };

        out.push_str(to_push.as_str());
    });

    out
}

/// Natural logarithm of binomial coefficent, ln nCk
///
/// # Example
///
/// ```rust
/// use rv::misc::ln_binom;
///
/// assert!((ln_binom(4.0, 2.0) - 6.0_f64.ln()) < 1E-12);
/// ```
pub fn ln_binom(n: f64, k: f64) -> f64 {
    (n + 1.0).ln_gamma().0 - (k + 1.0).ln_gamma().0 - (n - k + 1.0).ln_gamma().0
}

/// Safely compute `log(sum(exp(xs))`
/// Streaming `logexp` implementation as described in [Sebastian Nowozin's blog](https://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html)
pub fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        panic!("Empty container");
    } else if xs.len() == 1 {
        xs[0]
    } else {
        let (alpha, r) =
            xs.iter()
                .fold((std::f64::NEG_INFINITY, 0.0), |(alpha, r), &x| {
                    if x == std::f64::NEG_INFINITY {
                        (alpha, r)
                    } else if x <= alpha {
                        (alpha, r + (x - alpha).exp())
                    } else {
                        (x, r.mul_add((alpha - x).exp(), 1.0))
                    }
                });

        r.ln() + alpha
    }
}

/// Cumulative sum of `xs`
///
/// # Example
///
/// ```rust
/// # use rv::misc::cumsum;
/// let xs: Vec<i32> = vec![1, 1, 2, 1];
/// assert_eq!(cumsum(&xs), vec![1, 2, 4, 5]);
/// ```
pub fn cumsum<T>(xs: &[T]) -> Vec<T>
where
    T: AddAssign + Copy + Default,
{
    xs.iter()
        .scan(T::default(), |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect()
}

#[inline]
fn binary_search(cws: &[f64], r: f64) -> usize {
    let mut left: usize = 0;
    let mut right: usize = cws.len();
    while left < right {
        let mid = (left + right) / 2;
        if cws[mid] < r {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

#[inline]
fn catflip_bisection(cws: &[f64], r: f64) -> Option<usize> {
    let ix = binary_search(cws, r);
    if ix < cws.len() {
        Some(ix)
    } else {
        None
    }
}

#[inline]
fn catflip_standard(cws: &[f64], r: f64) -> Option<usize> {
    cws.iter().position(|&w| w > r)
}

fn catflip(cws: &[f64], r: f64) -> Option<usize> {
    if cws.len() > 9 {
        catflip_bisection(cws, r)
    } else {
        catflip_standard(cws, r)
    }
}

/// Draw `n` indices in proportion to their `weights`
pub fn pflip(weights: &[f64], n: usize, rng: &mut impl Rng) -> Vec<usize> {
    assert!(!weights.is_empty(), "Empty container");

    let cws: Vec<f64> = cumsum(weights);
    let scale: f64 = *cws.last().unwrap();
    let u = rand::distributions::Uniform::new(0.0, 1.0);

    (0..n)
        .map(|_| {
            let r = rng.sample(u) * scale;
            match catflip(&cws, r) {
                Some(ix) => ix,
                None => {
                    let wsvec = weights.to_vec();
                    panic!("Could not draw from {:?}", wsvec)
                }
            }
        })
        .collect()
}

/// Draw an index according to log-domain weights
///
/// Draw a `usize` from the categorical distribution defined by `ln_weights`.
/// If `normed` is `true` then exp(`ln_weights`) is assumed to sum to 1.
///
/// # Examples
///
/// ```rust
/// use rv::misc::ln_pflip;
///
/// let weights: Vec<f64> = vec![0.4, 0.2, 0.3, 0.1];
/// let ln_weights: Vec<f64> = weights.iter().map(|&w| w.ln()).collect();
///
/// let xs = ln_pflip(&ln_weights, 100, true, &mut rand::thread_rng());
///
/// assert_eq!(xs.len(), 100);
/// assert!(xs.iter().all(|&x| x <= 3));
/// assert!(!xs.iter().any(|&x| x > 3));
/// ```
///
/// Can handle -Inf ln weights
///
/// ```rust
/// # use rv::misc::ln_pflip;
/// use std::f64::NEG_INFINITY;
/// use std::f64::consts::LN_2;
///
/// let ln_weights: Vec<f64> = vec![-LN_2, NEG_INFINITY, -LN_2];
///
/// let xs = ln_pflip(&ln_weights, 100, true, &mut rand::thread_rng());
///
/// let zero_count = xs.iter().filter(|&&x| x == 0).count();
/// let one_count = xs.iter().filter(|&&x| x == 1).count();
/// let two_count = xs.iter().filter(|&&x| x == 2).count();
///
/// assert!(zero_count > 30);
/// assert_eq!(one_count, 0);
/// assert!(two_count > 30);
/// ```
pub fn ln_pflip<R: Rng>(
    ln_weights: &[f64],
    n: usize,
    normed: bool,
    rng: &mut R,
) -> Vec<usize> {
    let z = if normed { 0.0 } else { logsumexp(ln_weights) };

    // doing this instead of calling pflip shaves about 30% off the runtime.
    let cws: Vec<f64> = ln_weights
        .iter()
        .scan(0.0, |state, w| {
            *state += (w - z).exp();
            Some(*state)
        })
        .collect();

    (0..n)
        .map(|_| {
            let r = rng.sample(Open01);
            match catflip(&cws, r) {
                Some(ix) => ix,
                None => {
                    let wsvec = ln_weights.to_vec();
                    panic!("Could not draw from {:?}", wsvec)
                }
            }
        })
        .collect()
}

/// Indices of the largest element(s) in xs.
///
/// If there is more than one largest element, `argmax` returns the indices of
/// all replicates.
///
/// # Examples
///
/// ```rust
/// use rv::misc::argmax;
///
/// let xs: Vec<u8> = vec![1, 2, 3, 4, 5, 4, 5];
/// let ys: Vec<u8> = vec![1, 2, 3, 4, 5, 4, 0];
///
/// assert_eq!(argmax(&xs), vec![4, 6]);
/// assert_eq!(argmax(&ys), vec![4]);
/// ```
pub fn argmax<T: PartialOrd>(xs: &[T]) -> Vec<usize> {
    if xs.is_empty() {
        vec![]
    } else if xs.len() == 1 {
        vec![0]
    } else {
        let mut maxval = &xs[0];
        let mut max_ixs: Vec<usize> = vec![0];
        for (i, x) in xs.iter().enumerate().skip(1) {
            match x.partial_cmp(maxval) {
                Some(Ordering::Greater) => {
                    maxval = x;
                    max_ixs = vec![i];
                }
                Some(Ordering::Equal) => max_ixs.push(i),
                _ => (),
            }
        }
        max_ixs
    }
}

/// Natural logarithm of the multivariate gamma function, *ln Γ<sub>p</sub>(a)*.
///
/// # Arguments
///
/// * `p` - Positive integer degrees of freedom
/// * `a` - The number for which to compute the multivariate gamma
pub fn lnmv_gamma(p: usize, a: f64) -> f64 {
    let pf = p as f64;
    let a0 = pf * (pf - 1.0) / 4.0 * LN_PI;
    (1..=p).fold(a0, |acc, j| acc + (a + (1.0 - j as f64) / 2.0).ln_gamma().0)
}

/// Multivariate gamma function, *Γ<sub>p</sub>(a)*.
///
/// # Arguments
///
/// * `p` - Positive integer degrees of freedom
/// * `a` - The number for which to compute the multivariate gamma
pub fn mvgamma(p: usize, a: f64) -> f64 {
    lnmv_gamma(p, a).exp()
}

/// ln factorial
///
/// # Notes
///
/// n < 254 are computed via lookup table. n > 254 are computed via Sterling's
/// approximation. Code based on [C code from John
/// Cook](https://www.johndcook.com/blog/csharp_log_factorial/)
///
///
pub fn ln_fact(n: usize) -> f64 {
    if n < 254 {
        LN_FACT[n]
    } else {
        let y: f64 = (n as f64) + 1.0;
        (y - 0.5).mul_add(y.ln(), -y)
            + 0.5_f64.mul_add(LN_2PI, (12.0 * y).recip())
    }
}

const LN_FACT: [f64; 255] = [
    0.000_000_000_000_000,
    0.000_000_000_000_000,
    std::f64::consts::LN_2,
    1.791_759_469_228_055,
    3.178_053_830_347_946,
    4.787_491_742_782_046,
    6.579_251_212_010_101,
    8.525_161_361_065_415,
    10.604_602_902_745_25,
    12.801_827_480_081_469,
    15.104_412_573_075_516,
    17.502_307_845_873_887,
    19.987_214_495_661_885,
    22.552_163_853_123_42,
    25.191_221_182_738_683,
    27.899_271_383_840_894,
    30.671_860_106_080_675,
    33.505_073_450_136_89,
    36.395_445_208_033_05,
    39.339_884_187_199_495,
    42.335_616_460_753_485,
    45.380_138_898_476_91,
    48.471_181_351_835_23,
    51.606_675_567_764_38,
    54.784_729_398_112_32,
    58.003_605_222_980_52,
    61.261_701_761_002,
    64.557_538_627_006_32,
    67.889_743_137_181_53,
    71.257_038_967_168,
    74.658_236_348_830_16,
    78.092_223_553_315_3,
    81.557_959_456_115_03,
    85.054_467_017_581_52,
    88.580_827_542_197_68,
    92.136_175_603_687_08,
    95.719_694_542_143_2,
    99.330_612_454_787_43,
    102.968_198_614_513_81,
    106.631_760_260_643_45,
    110.320_639_714_757_39,
    114.034_211_781_461_69,
    117.771_881_399_745_06,
    121.533_081_515_438_64,
    125.317_271_149_356_88,
    129.123_933_639_127_24,
    132.952_575_035_616_3,
    136.802_722_637_326_35,
    140.673_923_648_234_25,
    144.565_743_946_344_9,
    148.477_766_951_773_02,
    152.409_592_584_497_35,
    156.360_836_303_078_8,
    160.331_128_216_630_93,
    164.320_112_263_195_17,
    168.327_445_448_427_65,
    172.352_797_139_162_82,
    176.395_848_406_997_37,
    180.456_291_417_543_78,
    184.533_828_861_449_5,
    188.628_173_423_671_6,
    192.739_047_287_844_9,
    196.866_181_672_889_98,
    201.009_316_399_281_57,
    205.168_199_482_641_2,
    209.342_586_752_536_82,
    213.532_241_494_563_27,
    217.736_934_113_954_25,
    221.956_441_819_130_36,
    226.190_548_323_727_57,
    230.439_043_565_776_93,
    234.701_723_442_818_26,
    238.978_389_561_834_35,
    243.268_849_002_982_73,
    247.572_914_096_186_9,
    251.890_402_209_723_2,
    256.221_135_550_009_5,
    260.564_940_971_863_2,
    264.921_649_798_552_8,
    269.291_097_651_019_8,
    273.673_124_285_693_7,
    278.067_573_440_366_1,
    282.474_292_687_630_4,
    286.893_133_295_427,
    291.323_950_094_270_3,
    295.766_601_350_760_6,
    300.220_948_647_014_1,
    304.686_856_765_668_7,
    309.164_193_580_146_9,
    313.652_829_949_879,
    318.152_639_620_209_3,
    322.663_499_126_726_2,
    327.185_287_703_775_2,
    331.717_887_196_928_5,
    336.261_181_979_198_45,
    340.815_058_870_798_96,
    345.379_407_062_266_86,
    349.954_118_040_770_25,
    354.539_085_519_440_8,
    359.134_205_369_575_34,
    363.739_375_555_563_47,
    368.354_496_072_404_7,
    372.979_468_885_689,
    377.614_197_873_918_67,
    382.258_588_773_06,
    386.912_549_123_217_56,
    391.575_988_217_329_6,
    396.248_817_051_791_5,
    400.930_948_278_915_76,
    405.622_296_161_144_9,
    410.322_776_526_937_3,
    415.032_306_728_249_6,
    419.750_805_599_544_8,
    424.478_193_418_257_1,
    429.214_391_866_651_57,
    433.959_323_995_014_87,
    438.712_914_186_121_17,
    443.475_088_120_918_94,
    448.245_772_745_384_6,
    453.024_896_238_496_1,
    457.812_387_981_278_1,
    462.608_178_526_874_9,
    467.412_199_571_608_1,
    472.224_383_926_980_5,
    477.044_665_492_585_6,
    481.872_979_229_887_9,
    486.709_261_136_839_36,
    491.553_448_223_298,
    496.405_478_487_217_6,
    501.265_290_891_579_24,
    506.132_825_342_034_83,
    511.008_022_665_236_07,
    515.890_824_587_822_5,
    520.781_173_716_044_2,
    525.679_013_515_995,
    530.584_288_294_433_6,
    535.496_943_180_169_5,
    540.416_924_105_997_7,
    545.344_177_791_155,
    550.278_651_724_285_6,
    555.220_294_146_895,
    560.169_054_037_273_1,
    565.124_881_094_874_4,
    570.087_725_725_134_2,
    575.057_539_024_710_2,
    580.034_272_767_130_8,
    585.017_879_388_839_2,
    590.008_311_975_617_9,
    595.005_524_249_382,
    600.009_470_555_327_4,
    605.020_105_849_423_8,
    610.037_385_686_238_7,
    615.061_266_207_084_9,
    620.091_704_128_477_4,
    625.128_656_730_891_1,
    630.172_081_847_810_2,
    635.221_937_855_059_8,
    640.278_183_660_408_1,
    645.340_778_693_435,
    650.409_682_895_655_2,
    655.484_856_710_889_1,
    660.566_261_075_873_5,
    665.653_857_411_106,
    670.747_607_611_912_7,
    675.847_474_039_736_9,
    680.953_419_513_637_5,
    686.065_407_301_994,
    691.183_401_114_410_8,
    696.307_365_093_814,
    701.437_263_808_737_2,
    706.573_062_245_787_5,
    711.714_725_802_29,
    716.862_220_279_103_4,
    722.015_511_873_601_3,
    727.174_567_172_815_8,
    732.339_353_146_739_3,
    737.509_837_141_777_4,
    742.685_986_874_351_2,
    747.867_770_424_643_4,
    753.055_156_230_484_2,
    758.248_113_081_374_3,
    763.446_610_112_640_2,
    768.650_616_799_717,
    773.860_102_952_558_5,
    779.075_038_710_167_4,
    784.295_394_535_245_7,
    789.521_141_208_959,
    794.752_249_825_813_5,
    799.988_691_788_643_5,
    805.230_438_803_703_1,
    810.477_462_875_863_6,
    815.729_736_303_910_2,
    820.987_231_675_937_9,
    826.249_921_864_842_8,
    831.517_780_023_906_3,
    836.790_779_582_469_9,
    842.068_894_241_700_5,
    847.352_097_970_438_4,
    852.640_365_001_133_1,
    857.933_669_825_857_5,
    863.231_987_192_405_4,
    868.535_292_100_464_6,
    873.843_559_797_865_7,
    879.156_765_776_907_6,
    884.474_885_770_751_8,
    889.797_895_749_890_2,
    895.125_771_918_679_9,
    900.458_490_711_945_3,
    905.796_028_791_646_3,
    911.138_363_043_611_2,
    916.485_470_574_328_8,
    921.837_328_707_804_9,
    927.193_914_982_476_7,
    932.555_207_148_186_2,
    937.921_183_163_208_1,
    943.291_821_191_335_7,
    948.667_099_599_019_8,
    954.046_996_952_560_4,
    959.431_492_015_349_5,
    964.820_563_745_165_9,
    970.214_191_291_518_3,
    975.612_353_993_036_2,
    981.015_031_374_908_4,
    986.422_203_146_368_6,
    991.833_849_198_223_4,
    997.249_949_600_427_8,
    1_002.670_484_599_700_3,
    1_008.095_434_617_181_7,
    1_013.524_780_246_136_2,
    1_018.958_502_249_690_2,
    1_024.396_581_558_613_4,
    1_029.838_999_269_135_5,
    1_035.285_736_640_801_6,
    1_040.736_775_094_367_4,
    1_046.192_096_209_725,
    1_051.651_681_723_869_2,
    1_057.115_513_528_895,
    1_062.583_573_670_03,
    1_068.055_844_343_701_4,
    1_073.532_307_895_632_8,
    1_079.012_946_818_975,
    1_084.497_743_752_465_6,
    1_089.986_681_478_622_4,
    1_095.479_742_921_962_7,
    1_100.976_911_147_256,
    1_106.478_169_357_800_9,
    1_111.983_500_893_733,
    1_117.492_889_230_361,
    1_123.006_317_976_526_1,
    1_128.523_770_872_990_8,
    1_134.045_231_790_853,
    1_139.570_684_729_984_8,
    1_145.100_113_817_496,
    1_150.633_503_306_223_7,
    1_156.170_837_573_242_4,
];

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn argmax_empty_is_empty() {
        let xs: Vec<f64> = vec![];
        assert_eq!(argmax(&xs), Vec::<usize>::new());
    }

    #[test]
    fn argmax_single_elem_is_0() {
        let xs: Vec<f64> = vec![1.0];
        assert_eq!(argmax(&xs), vec![0]);
    }

    #[test]
    fn argmax_unique_max() {
        let xs: Vec<u8> = vec![1, 2, 3, 4, 5, 4, 3];
        assert_eq!(argmax(&xs), vec![4]);
    }

    #[test]
    fn argmax_repeated_max() {
        let xs: Vec<u8> = vec![1, 2, 3, 4, 5, 4, 5];
        assert_eq!(argmax(&xs), vec![4, 6]);
    }

    #[test]
    fn logsumexp_on_vector_of_zeros() {
        let xs: Vec<f64> = vec![0.0; 5];
        // should be about log(5)
        assert::close(logsumexp(&xs), 1.609_437_912_434_100_3, TOL);
    }

    #[test]
    fn logsumexp_on_random_values() {
        let xs: Vec<f64> = vec![
            0.304_153_86,
            -0.070_722_96,
            -1.042_870_19,
            0.278_554_07,
            -0.818_967_65,
        ];
        assert::close(logsumexp(&xs), 1.482_000_789_426_305_9, TOL);
    }

    #[test]
    fn logsumexp_returns_only_value_on_one_element_container() {
        let xs: Vec<f64> = vec![0.304_153_86];
        assert::close(logsumexp(&xs), 0.304_153_86, TOL);
    }

    #[test]
    #[should_panic]
    fn logsumexp_should_panic_on_empty() {
        let xs: Vec<f64> = Vec::new();
        logsumexp(&xs);
    }

    #[test]
    fn logsumexp_leading_neginf() {
        let inf = std::f64::INFINITY;
        let weights = vec![
            -inf,
            -210.148_738_791_973_16,
            -818.104_304_460_164_3,
            -1_269.048_018_522_644_5,
            -2_916.862_476_271_387,
            -inf,
        ];

        let lse = logsumexp(&weights);
        assert::close(lse, -210.148_738_791_973_16, TOL);
    }

    #[test]
    fn lnmv_gamma_values() {
        assert::close(lnmv_gamma(1, 1.0), 0.0, TOL);
        assert::close(lnmv_gamma(1, 12.0), 17.502_307_845_873_887, TOL);
        assert::close(lnmv_gamma(3, 12.0), 50.615_815_724_290_74, TOL);
        assert::close(lnmv_gamma(3, 8.23), 25.709_195_968_438_628, TOL);
    }

    #[test]
    fn bisection_and_stanard_catflip_equivalence() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let n: usize = rng.gen_range(10..100);
            let cws: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let u2 = rand::distributions::Uniform::new(0.0, n as f64);
            let r = rng.sample(u2);

            let ix1 = catflip_standard(&cws, r).unwrap();
            let ix2 = catflip_bisection(&cws, r).unwrap();

            assert_eq!(ix1, ix2);
        }
    }

    #[test]
    fn ln_fact_agrees_with_naive() {
        fn ln_fact_naive(x: usize) -> f64 {
            if x < 2 {
                0.0
            } else {
                (2..=x).map(|y| (y as f64).ln()).sum()
            }
        }

        for x in 0..300 {
            let f1 = ln_fact_naive(x);
            let f2 = ln_fact(x);
            assert::close(f1, f2, 1e-9);
        }
    }

    #[test]
    fn ln_pflip_works_with_zero_weights() {
        use std::f64::consts::LN_2;
        use std::f64::NEG_INFINITY;

        let ln_weights: Vec<f64> = vec![-LN_2, NEG_INFINITY, -LN_2];

        let xs = ln_pflip(&ln_weights, 100, true, &mut rand::thread_rng());

        let zero_count = xs.iter().filter(|&&x| x == 0).count();
        let one_count = xs.iter().filter(|&&x| x == 1).count();
        let two_count = xs.iter().filter(|&&x| x == 2).count();

        assert!(zero_count > 30);
        assert_eq!(one_count, 0);
        assert!(two_count > 30);
    }
}
