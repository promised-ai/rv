#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]

const MAX_ITER: usize = 500;

const BESSI0_COEFFS_A: [f64; 30] = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1,
];

#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
const BESSI0_COEFFS_B: [f64; 25] = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1,
];

#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
const BESSI1_COEFFS_A: [f64; 29] = [
    2.77791411276104639959E-18,
    -2.11142121435816608115E-17,
    1.55363195773620046921E-16,
    -1.10559694773538630805E-15,
    7.60068429473540693410E-15,
    -5.04218550472791168711E-14,
    3.22379336594557470981E-13,
    -1.98397439776494371520E-12,
    1.17361862988909016308E-11,
    -6.66348972350202774223E-11,
    3.62559028155211703701E-10,
    -1.88724975172282928790E-9,
    9.38153738649577178388E-9,
    -4.44505912879632808065E-8,
    2.00329475355213526229E-7,
    -8.56872026469545474066E-7,
    3.47025130813767847674E-6,
    -1.32731636560394358279E-5,
    4.78156510755005422638E-5,
    -1.61760815825896745588E-4,
    5.12285956168575772895E-4,
    -1.51357245063125314899E-3,
    4.15642294431288815669E-3,
    -1.05640848946261981558E-2,
    2.47264490306265168283E-2,
    -5.29459812080949914269E-2,
    1.02643658689847095384E-1,
    -1.76416518357834055153E-1,
    2.52587186443633654823E-1,
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
#[derive(Clone, Debug, PartialEq)]
pub enum BesselIvError {
    /// The order, v, must be an integer if z is negative.
    OrderMustIntegerError,
    /// Arguments would lead to an overflow
    OverflowError,
    /// Failed to converge
    FailedToConvergeError,
    /// Precision Error
    PrecisionLossError,
    /// Domain Error
    DomainError,
}

/// Modified Bessel function of the first kind of real order
pub fn bessel_iv(v: f64, z: f64) -> Result<f64, BesselIvError> {
    if v.is_nan() || z.is_nan() {
        return Ok(std::f64::NAN);
    }
    let (v, t) = {
        let t = v.floor();
        if v < 0.0 && t == v {
            (-v, -t)
        } else {
            (v, t)
        }
    };

    let sign: f64 = if z < 0.0 {
        // Return error if v is not an integer if x < 0
        if t != v {
            return Err(BesselIvError::OrderMustIntegerError);
        }

        if v != 2.0 * (v / 2.0).floor() {
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
            return Err(BesselIvError::OverflowError);
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
        -0.20833333333333334,
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
        0.3342013888888889,
        0.0,
        -0.40104166666666669,
        0.0,
        0.0703125,
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
        -1.0258125964506173,
        0.0,
        1.8464626736111112,
        0.0,
        -0.89121093750000002,
        0.0,
        0.0732421875,
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
        4.6695844234262474,
        0.0,
        -11.207002616222995,
        0.0,
        8.78912353515625,
        0.0,
        -2.3640869140624998,
        0.0,
        0.112152099609375,
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
        -28.212072558200244,
        0.0,
        84.636217674600744,
        0.0,
        -91.818241543240035,
        0.0,
        42.534998745388457,
        0.0,
        -7.3687943594796312,
        0.0,
        0.22710800170898438,
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
        212.5701300392171,
        0.0,
        -765.25246814118157,
        0.0,
        1059.9904525279999,
        0.0,
        -699.57962737613275,
        0.0,
        218.19051174421159,
        0.0,
        -26.491430486951554,
        0.0,
        0.57250142097473145,
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
        -1919.4576623184068,
        0.0,
        8061.7221817373083,
        0.0,
        -13586.550006434136,
        0.0,
        11655.393336864536,
        0.0,
        -5305.6469786134048,
        0.0,
        1200.9029132163525,
        0.0,
        -108.09091978839464,
        0.0,
        1.7277275025844574,
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
        20204.291330966149,
        0.0,
        -96980.598388637503,
        0.0,
        192547.0012325315,
        0.0,
        -203400.17728041555,
        0.0,
        122200.46498301747,
        0.0,
        -41192.654968897557,
        0.0,
        7109.5143024893641,
        0.0,
        -493.915304773088,
        0.0,
        6.074042001273483,
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
        -242919.18790055133,
        0.0,
        1311763.6146629769,
        0.0,
        -2998015.9185381061,
        0.0,
        3763271.2976564039,
        0.0,
        -2813563.2265865342,
        0.0,
        1268365.2733216248,
        0.0,
        -331645.17248456361,
        0.0,
        45218.768981362737,
        0.0,
        -2499.8304818112092,
        0.0,
        24.380529699556064,
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
        3284469.8530720375,
        0.0,
        -19706819.11843222,
        0.0,
        50952602.492664628,
        0.0,
        -74105148.211532637,
        0.0,
        66344512.274729028,
        0.0,
        -37567176.660763353,
        0.0,
        13288767.166421819,
        0.0,
        -2785618.1280864552,
        0.0,
        308186.40461266245,
        0.0,
        -13886.089753717039,
        0.0,
        110.01714026924674,
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
    use std::f64::EPSILON;
    let (v, sign) = (v.abs(), v.signum());

    let z = x / v;
    let t = (1.0 + z * z).sqrt().recip();
    let t2 = t * t;
    let eta = (1.0 + z * z).sqrt() + (z / (1.0 + t.recip())).ln();

    let i_prefactor = (t / (2.0 * PI * v)).sqrt() * (v * eta).exp();
    let mut i_sum = 1.0;

    let k_prefactor = (PI * t / (2.0 * v)).sqrt() * (-v * eta).exp();
    let mut k_sum = 1.0;

    let mut divisor = v;
    let mut term = 0.0;

    for n in 1..N_UFACTORS {
        term = 0.0;
        for k in
            ((N_UFACTOR_TERMS - 1 - 3 * n)..(N_UFACTOR_TERMS - n)).step_by(2)
        {
            term *= t2;
            term += ASYMPTOTIC_UFACTORS[n][k];
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

        if term.abs() < EPSILON {
            break;
        }
        divisor *= v;
    }

    // check convergence
    if term.abs() > 1E-3 * i_sum.abs() {
        Err(BesselIvError::FailedToConvergeError)
    } else if term.abs() > EPSILON * i_sum.abs() {
        Err(BesselIvError::PrecisionLossError)
    } else {
        let k_value = k_prefactor * k_sum;
        let i_value = if sign > 0.0 {
            i_prefactor * i_sum
        } else {
            i_prefactor * i_sum
                + (2.0 / PI) * (PI * v).sin() * k_prefactor * k_sum
        };
        Ok((i_value, k_value))
    }
}

/// Compute I(v, x) and K(v, x) simultaneously by Temme's method, see
/// Temme, Journal of Computational Physics, vol 19, 324 (1975)
/// Heavily inspired by
/// https://github.com/scipy/scipy/blob/1984f97749a355a6767cea55cad5d1dc6977ad5f/scipy/special/cephes/scipy_iv.c#L532
fn bessel_ikv_temme(v: f64, x: f64) -> Result<(f64, f64), BesselIvError> {
    use std::f64::consts::PI;
    use std::f64::EPSILON;
    let (v, reflect) = if v < 0.0 { (-v, true) } else { (v, false) };

    let n = v.round();
    let u = v - n;
    let n = n as isize;

    if x < 0.0 {
        return Err(BesselIvError::DomainError);
    } else if x == 0.0 {
        return Err(BesselIvError::OverflowError);
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

    let lim = ((4.0 * v * v + 10.0) / (8.0 * x)).powi(3) / 24.0;

    let iv = if lim < 10.0 * EPSILON && x > 100.0 {
        bessel_iv_asymptotic(v, x)?
    } else {
        let fv = cf1_ik(v, x)?;
        w / (kv * fv + kv1)
    };

    if reflect {
        let z = (u as f64) + ((n % 2) as f64);
        Ok((iv + (2.0 / PI) * (PI * z).sin() * kv, kv))
    } else {
        Ok((iv, kv))
    }
}

/// Modified Bessel functions of the first and second kind of fractional order
///
/// Calculate K(v, x) and K(v+1, x) by method analogous to
/// Temme, Journal of Computational Physics, vol 21, 343 (1976)
fn temme_ik_series(v: f64, x: f64) -> Result<(f64, f64), BesselIvError> {
    use crate::consts::EULER_MASCERONI;
    use special::Gamma;
    use std::f64::consts::PI;
    use std::f64::EPSILON;
    /*
     * |x| <= 2, Temme series converge rapidly
     * |x| > 2, the larger the |x|, the slower the convergence
     */
    debug_assert!(x.abs() <= 2.0);
    debug_assert!(v.abs() <= 0.5);

    let gp = (v + 1.0).gamma() - 1.0;
    let gm = (1.0 - v).gamma() - 1.0;

    let a = (x / 2.0).ln();
    let b = (v * a).exp();
    let sigma = -a * v;
    let c = if v.abs() < 2.0 * EPSILON {
        1.0
    } else {
        (PI * v).sin() / (PI * v)
    };
    let d = if sigma.abs() < EPSILON {
        1.0
    } else {
        sigma.sinh() / sigma
    };
    let gamma1 = if v.abs() < EPSILON {
        -EULER_MASCERONI
    } else {
        (0.5 / v) * (gp - gm) * c
    };
    let gamma2 = (2.0 + gp + gm) * c / 2.0;

    let mut p = (gp + 1.0) / (2.0 * b);
    let mut q = (gm + 1.0) * b / 2.0;
    let mut f = (sigma.cosh() * gamma1 + d * -a * gamma2) / c;
    let mut h = p;
    let mut coef = 1.0;
    let mut sum = coef * f;
    let mut sum1 = coef * h;

    for k in 1..MAX_ITER {
        let kf = k as f64;
        f = (kf * f + p + q) / (kf * kf - v * v);
        p /= kf - v;
        q /= kf + v;
        h = p - kf * f;
        coef *= x * x / (4.0 * kf);
        sum += coef * f;
        sum1 += coef * h;

        if (coef * f).abs() < sum.abs() * EPSILON {
            return Ok((sum, 2.0 * sum1 / x));
        }
    }

    Err(BesselIvError::FailedToConvergeError)
}

/// Calculate K(v, x) and K(v+1, x) by evaluating continued fraction
/// z1 / z0 = U(v+1.5, 2v+1, 2x) / U(v+0.5, 2v+1, 2x), see
/// Thompson and Barnett, Computer Physics Communications, vol 47, 245 (1987)
fn cf2_ik(v: f64, x: f64) -> Result<(f64, f64), BesselIvError> {
    use std::f64::consts::PI;
    use std::f64::EPSILON;
    /*
     * Steed's algorithm, see Thompson and Barnett,
     * Journal of Computational Physics, vol 64, 490 (1986)
     */
    debug_assert!(x.abs() > 1.0);

    let mut a = v * v - 0.25;
    let mut b = 2.0 * (x + 1.0);
    let mut d = b.recip();

    let mut delta = d;
    let mut f = d;
    let mut prev = 0.0;
    let mut cur = 1.0;
    let mut q = -a;
    let mut c = -a;
    let mut s = 1.0 + q * delta;

    for k in 2..MAX_ITER {
        let kf = k as f64;
        a -= 2.0 * (kf - 1.0);
        b += 2.0;
        d = (b + a * d).recip();
        delta *= b * d - 1.0;
        f += delta;

        let t = (prev - (b - 2.0) * cur) / a;
        prev = cur;
        cur = t;
        c *= -a / kf;
        q += c * t;
        s += q * delta;

        if (q * delta).abs() < s.abs() * EPSILON / 2.0 {
            let kv = (PI / (2.0 * x)).sqrt() * (-x).exp() / s;
            let kv1 = kv * (0.5 + v + x + (v * v - 0.25) * f) / x;
            return Ok((kv, kv1));
        }
    }
    Err(BesselIvError::FailedToConvergeError)
}

/// Evaluate continued fraction fv = I_(v+1) / I_v, derived from
/// Abramowitz and Stegun, Handbook of Mathematical Functions, 1972, 9.1.73 */
fn cf1_ik(v: f64, x: f64) -> Result<f64, BesselIvError> {
    use std::f64::EPSILON;

    /*
     * |x| <= |v|, CF1_ik converges rapidly
     * |x| > |v|, CF1_ik needs O(|x|) iterations to converge
     */

    /*
     * modified Lentz's method, see
     * Lentz, Applied Optics, vol 15, 668 (1976)
     */

    const TOL: f64 = EPSILON;
    let tiny: f64 = std::f64::MAX.sqrt().recip();
    let mut c = tiny;
    let mut f = tiny;
    let mut d = 0.0;

    for k in 1..MAX_ITER {
        let kf = k as f64;
        let a = 1.0;
        let b = 2.0 * (v + kf) / x;
        c = b + a / c;
        d = b + a * d;
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

    Err(BesselIvError::FailedToConvergeError)
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

        while term.abs() > std::f64::EPSILON * sum.abs() {
            let kf = k as f64;
            let factor =
                (mu - (2.0 * kf - 1.0) * (2.0 * kf - 1.0)) / (8.0 * x) / kf;
            if k > 100 {
                return Err(BesselIvError::FailedToConvergeError);
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
            0.1392518662822890,
            TOL,
        );
    }

    #[test]
    fn bessi0_small() {
        assert::close(i0(3.74), 9.0414968490127734, TOL);
        assert::close(i0(-3.74), 9.0414968490127734, TOL);
        assert::close(i0(8.0), 427.56411572180474, TOL);
    }

    #[test]
    fn bessi0_large() {
        assert::close(i0(8.1), 469.5006067101214, TOL);
        assert::close(i0(10.0), 2815.716628466254, TOL);
    }

    #[test]
    fn bessi1_small() {
        assert::close(i1(3.74), 7.709894215253694, TOL);
        assert::close(i1(-3.74), -7.709894215253694, TOL);
        assert::close(i1(0.0024), 0.0012000008640002072, TOL);
        assert::close(i1(8.0), 399.8731367825599, TOL);
    }

    #[test]
    fn bessi1_large() {
        assert::close(i1(8.1), 439.48430891035844, TOL);
        assert::close(i1(10.0), 2670.988303701255, TOL);
    }

    #[test]
    fn bessel_iv_basic_limits() {
        assert::close(bessel_iv(0.0, 0.0).unwrap(), 1.0, TOL);
        assert::close(bessel_iv(1.0, 0.0).unwrap(), 0.0, TOL);
    }

    #[test]
    fn bessel_iv_high_order() {
        assert::close(bessel_iv(60.0, 40.0).unwrap(), 0.07185641968452632, TOL);
    }

    #[test]
    fn bessel_iv_low_order() {
        assert::close(bessel_iv(0.0, 1.0).unwrap(), 1.2660658777520084, TOL);
        assert::close(bessel_iv(0.0, 10.0).unwrap(), 2815.7166284662544, TOL);

        assert::close(bessel_iv(1.0, 10.0).unwrap(), 2670.988303701254, TOL);
        assert::close(
            bessel_iv(20.0, 10.0).unwrap(),
            0.00012507997356449478,
            TOL,
        );
    }

    #[test]
    fn cf1_ik_checks() {
        assert::close(cf1_ik(0.0, 10.0).unwrap(), 0.9485998259548458, TOL);
        assert::close(cf1_ik(10.0, 10.0).unwrap(), 0.38991388392838294, TOL);
        assert::close(cf1_ik(60.0, 5.0).unwrap(), 0.04091609790883304, TOL);
    }

    #[test]
    fn cf2_ik_checks() {
        let (k1, k2) = cf2_ik(0.0, 2.0).unwrap();
        assert::close(k1, 0.1138938727495335256901, TOL);
        assert::close(k2, 0.1398658818165225414809, TOL);

        let (k1, k2) = cf2_ik(5.0, 5.0).unwrap();
        assert::close(k1, 3.270627371203162214730e-02, TOL);
        assert::close(k2, 8.067161323456370491947e-02, TOL);
    }

    #[test]
    fn temme_ik_series_checks() {
        let res = temme_ik_series(0.0, 0.0);
        assert!(res.is_err());

        let (k1, k2) = temme_ik_series(0.0, 1.0).unwrap();
        assert::close(k1, 4.210244382407083429953e-01, TOL);
        assert::close(k2, 6.019072301972345773180e-01, TOL);

        let (k1, k2) = temme_ik_series(0.5, 2.0).unwrap();
        assert::close(k1, 1.199377719680612269793e-01, TOL);
        assert::close(k2, 1.799066579520924302749e-01, TOL);
    }

    #[test]
    fn bessel_ikv_temme_checks() {
        let (i, k) = bessel_ikv_temme(0.0, 1.0).unwrap();
        assert::close(i, 1.2660658777520084, TOL);
        assert::close(k, 0.42102443824070834, TOL);

        let (i, k) = bessel_ikv_temme(5.0, 2.0).unwrap();
        assert::close(i, 0.009825679323131702, TOL);
        assert::close(k, 9.431049100596468, TOL);

        let (i, k) = bessel_ikv_temme(20.0, 2.0).unwrap();
        assert::close(i, 4.310560576109548E-19, TOL);
        assert::close(k, 5.7708568527002424E16, TOL);

        let (i, k) = bessel_ikv_temme(20.0, 2.0).unwrap();
        assert::close(i, 4.310560576109548E-19, TOL);
        assert::close(k, 5.7708568527002424E16, TOL);

        let (i, k) = bessel_ikv_temme(1.0, 10.0).unwrap();
        assert::close(i, 2670.988303701254, TOL);
        assert::close(k, 1.8648773453825585E-5, TOL);
    }

    #[test]
    fn bessel_ikv_asymptotic_uniform_checks() {
        let (i, k) = bessel_ikv_asymptotic_uniform(60.0, 40.0).unwrap();
        assert::close(i, 7.185641968452631533903e-2, TOL);
        assert::close(k, 9.649278749222318929402e-2, TOL);

        let (i, k) = bessel_ikv_asymptotic_uniform(100.0, 60.0).unwrap();
        assert::close(i, 2.883277090649163827086e-7, TOL);
        assert::close(k, 1.487001275494647416053e4, TOL);
    }
}
