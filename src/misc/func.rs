use crate::consts::{LN_2PI, LN_PI};
use crate::traits::Rv;
use rand::distributions::Open01;
use rand::Rng;
use special::Gamma;
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::ops::AddAssign;

/// Compute the entropy for count-type distributions by enumeration
///
/// # Notes
/// Enumeration begins at `x` and ends at the first `x` after `mid` for which
/// f(x) < 1E-16.
///
/// # Example
///
/// ```rust
/// # use rv::misc::count_entropy;
/// use rv::dist::Poisson;
///
/// let pois = Poisson::new(1.2).unwrap();
/// let h = count_entropy(&pois, 0, 2);
/// assert!((h - 1.4100058897431968).abs() < 1e-9) ;
/// ```
pub fn count_entropy<Fx: Rv<u32>>(fx: &Fx, mut x: u32, mid: u32) -> f64 {
    let mut h = 0.0;
    loop {
        let ln_f = fx.ln_f(&x);
        let f = ln_f.exp();
        h -= f * ln_f;
        if x > mid && f < 1E-16 {
            return h;
        }
        x += 1;
    }
}

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
pub fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        panic!("Empty container");
    } else if xs.len() == 1 {
        xs[0]
    } else {
        let maxval =
            *xs.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();

        xs.iter().fold(0.0, |acc, x| acc + (x - maxval).exp()).ln() + maxval
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
    let ix = binary_search(&cws, r);
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
        catflip_bisection(&cws, r)
    } else {
        catflip_standard(&cws, r)
    }
}

/// Draw `n` indices in proportion to their `weights`
pub fn pflip(weights: &[f64], n: usize, rng: &mut impl Rng) -> Vec<usize> {
    if weights.is_empty() {
        panic!("Empty container");
    }
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
pub fn ln_pflip<R: Rng>(
    ln_weights: &[f64],
    n: usize,
    normed: bool,
    rng: &mut R,
) -> Vec<usize> {
    let z = if normed { 0.0 } else { logsumexp(ln_weights) };

    let mut cws: Vec<f64> = ln_weights.iter().map(|w| (w - z).exp()).collect();

    // doing this instead of calling pflip shaves about 30% off the runtime.
    for i in 1..cws.len() {
        cws[i] += cws[i - 1];
    }

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
        (y - 0.5) * y.ln() - y + 0.5 * LN_2PI + (12.0 * y).recip()
    }
}

const LN_FACT: [f64; 255] = [
    0.000000000000000,
    0.000000000000000,
    0.693147180559945,
    1.791759469228055,
    3.178053830347946,
    4.787491742782046,
    6.579251212010101,
    8.525161361065415,
    10.604602902745251,
    12.801827480081469,
    15.104412573075516,
    17.502307845873887,
    19.987214495661885,
    22.552163853123421,
    25.191221182738683,
    27.899271383840894,
    30.671860106080675,
    33.505073450136891,
    36.395445208033053,
    39.339884187199495,
    42.335616460753485,
    45.380138898476908,
    48.471181351835227,
    51.606675567764377,
    54.784729398112319,
    58.003605222980518,
    61.261701761002001,
    64.557538627006323,
    67.889743137181526,
    71.257038967168000,
    74.658236348830158,
    78.092223553315307,
    81.557959456115029,
    85.054467017581516,
    88.580827542197682,
    92.136175603687079,
    95.719694542143202,
    99.330612454787428,
    102.968198614513810,
    106.631760260643450,
    110.320639714757390,
    114.034211781461690,
    117.771881399745060,
    121.533081515438640,
    125.317271149356880,
    129.123933639127240,
    132.952575035616290,
    136.802722637326350,
    140.673923648234250,
    144.565743946344900,
    148.477766951773020,
    152.409592584497350,
    156.360836303078800,
    160.331128216630930,
    164.320112263195170,
    168.327445448427650,
    172.352797139162820,
    176.395848406997370,
    180.456291417543780,
    184.533828861449510,
    188.628173423671600,
    192.739047287844900,
    196.866181672889980,
    201.009316399281570,
    205.168199482641200,
    209.342586752536820,
    213.532241494563270,
    217.736934113954250,
    221.956441819130360,
    226.190548323727570,
    230.439043565776930,
    234.701723442818260,
    238.978389561834350,
    243.268849002982730,
    247.572914096186910,
    251.890402209723190,
    256.221135550009480,
    260.564940971863220,
    264.921649798552780,
    269.291097651019810,
    273.673124285693690,
    278.067573440366120,
    282.474292687630400,
    286.893133295426990,
    291.323950094270290,
    295.766601350760600,
    300.220948647014100,
    304.686856765668720,
    309.164193580146900,
    313.652829949878990,
    318.152639620209300,
    322.663499126726210,
    327.185287703775200,
    331.717887196928470,
    336.261181979198450,
    340.815058870798960,
    345.379407062266860,
    349.954118040770250,
    354.539085519440790,
    359.134205369575340,
    363.739375555563470,
    368.354496072404690,
    372.979468885689020,
    377.614197873918670,
    382.258588773060010,
    386.912549123217560,
    391.575988217329610,
    396.248817051791490,
    400.930948278915760,
    405.622296161144900,
    410.322776526937280,
    415.032306728249580,
    419.750805599544780,
    424.478193418257090,
    429.214391866651570,
    433.959323995014870,
    438.712914186121170,
    443.475088120918940,
    448.245772745384610,
    453.024896238496130,
    457.812387981278110,
    462.608178526874890,
    467.412199571608080,
    472.224383926980520,
    477.044665492585580,
    481.872979229887900,
    486.709261136839360,
    491.553448223298010,
    496.405478487217580,
    501.265290891579240,
    506.132825342034830,
    511.008022665236070,
    515.890824587822520,
    520.781173716044240,
    525.679013515995050,
    530.584288294433580,
    535.496943180169520,
    540.416924105997740,
    545.344177791154950,
    550.278651724285620,
    555.220294146894960,
    560.169054037273100,
    565.124881094874350,
    570.087725725134190,
    575.057539024710200,
    580.034272767130800,
    585.017879388839220,
    590.008311975617860,
    595.005524249382010,
    600.009470555327430,
    605.020105849423770,
    610.037385686238740,
    615.061266207084940,
    620.091704128477430,
    625.128656730891070,
    630.172081847810200,
    635.221937855059760,
    640.278183660408100,
    645.340778693435030,
    650.409682895655240,
    655.484856710889060,
    660.566261075873510,
    665.653857411105950,
    670.747607611912710,
    675.847474039736880,
    680.953419513637530,
    686.065407301994010,
    691.183401114410800,
    696.307365093814040,
    701.437263808737160,
    706.573062245787470,
    711.714725802289990,
    716.862220279103440,
    722.015511873601330,
    727.174567172815840,
    732.339353146739310,
    737.509837141777440,
    742.685986874351220,
    747.867770424643370,
    753.055156230484160,
    758.248113081374300,
    763.446610112640200,
    768.650616799717000,
    773.860102952558460,
    779.075038710167410,
    784.295394535245690,
    789.521141208958970,
    794.752249825813460,
    799.988691788643450,
    805.230438803703120,
    810.477462875863580,
    815.729736303910160,
    820.987231675937890,
    826.249921864842800,
    831.517780023906310,
    836.790779582469900,
    842.068894241700490,
    847.352097970438420,
    852.640365001133090,
    857.933669825857460,
    863.231987192405430,
    868.535292100464630,
    873.843559797865740,
    879.156765776907600,
    884.474885770751830,
    889.797895749890240,
    895.125771918679900,
    900.458490711945270,
    905.796028791646340,
    911.138363043611210,
    916.485470574328820,
    921.837328707804890,
    927.193914982476710,
    932.555207148186240,
    937.921183163208070,
    943.291821191335660,
    948.667099599019820,
    954.046996952560450,
    959.431492015349480,
    964.820563745165940,
    970.214191291518320,
    975.612353993036210,
    981.015031374908400,
    986.422203146368590,
    991.833849198223450,
    997.249949600427840,
    1002.670484599700300,
    1008.095434617181700,
    1013.524780246136200,
    1018.958502249690200,
    1024.396581558613400,
    1029.838999269135500,
    1035.285736640801600,
    1040.736775094367400,
    1046.192096209724900,
    1051.651681723869200,
    1057.115513528895000,
    1062.583573670030100,
    1068.055844343701400,
    1073.532307895632800,
    1079.012946818975000,
    1084.497743752465600,
    1089.986681478622400,
    1095.479742921962700,
    1100.976911147256000,
    1106.478169357800900,
    1111.983500893733000,
    1117.492889230361000,
    1123.006317976526100,
    1128.523770872990800,
    1134.045231790853000,
    1139.570684729984800,
    1145.100113817496100,
    1150.633503306223700,
    1156.170837573242400,
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
        assert::close(logsumexp(&xs), 1.6094379124341003, TOL);
    }

    #[test]
    fn logsumexp_on_random_values() {
        let xs: Vec<f64> = vec![
            0.30415386,
            -0.07072296,
            -1.04287019,
            0.27855407,
            -0.81896765,
        ];
        assert::close(logsumexp(&xs), 1.4820007894263059, TOL);
    }

    #[test]
    fn logsumexp_returns_only_value_on_one_element_container() {
        let xs: Vec<f64> = vec![0.30415386];
        assert::close(logsumexp(&xs), 0.30415386, TOL);
    }

    #[test]
    #[should_panic]
    fn logsumexp_should_panic_on_empty() {
        let xs: Vec<f64> = Vec::new();
        logsumexp(&xs);
    }

    #[test]
    fn lnmv_gamma_values() {
        assert::close(lnmv_gamma(1, 1.0), 0.0, TOL);
        assert::close(lnmv_gamma(1, 12.0), 17.502307845873887, TOL);
        assert::close(lnmv_gamma(3, 12.0), 50.615815724290741, TOL);
        assert::close(lnmv_gamma(3, 8.23), 25.709195968438628, TOL);
    }

    #[test]
    fn bisection_and_stanard_catflip_equivalence() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let n: usize = rng.gen_range(10, 100);
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
}
