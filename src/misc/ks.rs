use crate::dist::KsTwoAsymptotic;
use crate::traits::Cdf;
use num::integer::binomial;
use num::Integer;

/// Univariate one-sample [Kolmogorov-Smirnov](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
/// test.
///
/// Given a set of samples, `xs`, and a distribution `F`, the KS test
/// determines whether `xs` were generated by `F`.
///
/// # Example
///
/// ```rust
/// use rv::prelude::*;
/// use rv::misc::ks_test;
///
/// let gauss = Gaussian::standard();
/// let laplace = Laplace::new(0.0, 1.0).unwrap();
///
/// let gauss_cdf = |x: f64| gauss.cdf(&x);
/// let laplace_cdf = |x: f64| laplace.cdf(&x);
///
/// // Generate some samples from the Gaussian
/// let mut rng = rand::thread_rng();
/// let xs = gauss.sample(1000, &mut rng);
///
/// // Check the the samples came from the one that generated them
/// let (_, p_gauss) = ks_test(&xs, gauss_cdf);
/// assert!(p_gauss > 0.05);
///
/// // They did not come from a Laplace
/// let (_, p_laplace) = ks_test(&xs, laplace_cdf);
/// assert!(p_laplace < 0.05);
/// ```
pub fn ks_test<X, F>(xs: &[X], cdf: F) -> (f64, f64)
where
    X: Copy + PartialOrd,
    F: Fn(X) -> f64,
{
    let mut xs_r: Vec<X> = xs.to_vec();
    xs_r.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n: f64 = xs_r.len() as f64;
    let d = xs_r.iter().enumerate().fold(0.0, |acc, (i, &x)| {
        let diff = ((i as f64) / n - cdf(x)).abs();
        if diff > acc {
            diff
        } else {
            acc
        }
    });

    let p = 1.0 - ks_cdf(xs.len(), d);
    (d, p)
}

const KS_AUTO_CUTOVER: usize = 10_000;

/// Mode in which to run the KS Test
#[derive(Debug, Clone, Copy, Default)]
pub enum KsMode {
    /// Compute the exact statistic
    Exact,
    /// Compute the statistic in the large N limit
    Asymptotic,
    /// Determine appropiate method automatically.
    #[default]
    Auto,
}

/// Hypothesis Alternative for ks_two_sample test
#[derive(Debug, Clone, Copy, Default)]
pub enum KsAlternative {
    /// Alternative where the emperical CDFs could lie on either side on one another.
    #[default]
    TwoSided,
    /// Alternative where the emperical CDF of the first sequence is strictly less than the
    /// emperical cdf of the second sequence.
    Less,
    /// Alternative where the emperical CDF of the first sequence is strictly greater than the
    /// emperical cdf of the second sequence.
    Greater,
}

/// Errors when calculating ks_two_sample
#[derive(Debug)]
pub enum KsError {
    /// Once of the slices given is empty
    EmptySlice,
    /// Requested Exact with too many elements
    TooLongForExact,
}

/// Two sample Kolmogorov-Smirnov statistic on two samples.
///
/// Heavily inspired by https://github.com/scipy/scipy/blob/v1.4.1/scipy/stats/stats.py#L6087
/// Exact computations are derived from:
///     Hodges, J.L. Jr.,  "The Significance Probability of the Smirnov
///         Two-Sample Test," Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.
///
/// # Example
/// ```rust
/// use rv::misc::{ks_two_sample, KsMode, KsAlternative};
///
/// let xs = [
///     0.95692026,  1.1348812 , -0.76579239, -0.58065653, -0.05122393,
///     0.71598754,  1.39873528,  0.42790527,  1.84580764,  0.64228521
/// ];
///
/// let ys = [
///     0.6948678 , -0.3741825 ,  0.36657279,  1.15834174, -0.32421706,
///     -0.38499295,  1.44976991,  0.2504608 , -0.53694774,  1.42221993
/// ];
///
/// let (stat, alpha) = ks_two_sample(&xs, &ys, KsMode::Auto, KsAlternative::TwoSided).unwrap();
///
/// assert::close(stat, 0.3, 1E-8);
/// assert::close(alpha, 0.7869297884777761, 1E-8);
/// ```
#[allow(clippy::many_single_char_names)]
pub fn ks_two_sample<X>(
    xs: &[X],
    ys: &[X],
    mode: KsMode,
    alternative: KsAlternative,
) -> Result<(f64, f64), KsError>
where
    X: Copy + PartialOrd,
{
    if xs.is_empty() || ys.is_empty() {
        return Err(KsError::EmptySlice);
    }

    let n_x = xs.len();
    let n_y = ys.len();
    let n_x_f = xs.len() as f64;
    let n_y_f = ys.len() as f64;

    let mut xs = xs.to_vec();
    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut ys = ys.to_vec();
    ys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut cdf_x = Vec::new();
    let mut cdf_y = Vec::new();

    for x in [&xs[..], &ys[..]].concat() {
        match xs.binary_search_by(|b| b.partial_cmp(&x).unwrap()) {
            Ok(z) => cdf_x.push((z as f64) / n_x_f),
            Err(z) => cdf_x.push((z as f64) / n_x_f),
        }
        match ys.binary_search_by(|b| b.partial_cmp(&x).unwrap()) {
            Ok(z) => cdf_y.push((z as f64) / n_y_f),
            Err(z) => cdf_y.push((z as f64) / n_y_f),
        }
    }

    let (min_s, max_s) = cdf_x
        .iter()
        .zip(cdf_y.iter())
        .map(|(cx, cy)| (cx - cy))
        .fold((std::f64::MAX, std::f64::MIN), |(min, max), z| {
            let new_min = min.min(z);
            let new_max = max.max(z);
            (new_min, new_max)
        });

    let min_s = -min_s;

    let stat = match alternative {
        KsAlternative::Less => min_s,
        KsAlternative::Greater => max_s,
        KsAlternative::TwoSided => max_s.max(min_s),
    };

    let g = n_x.gcd(&n_y);
    let g_f = g as f64;
    let n_x_g = n_x_f / (g as f64);
    let n_y_g = n_y_f / (g as f64);

    let use_method = match mode {
        KsMode::Asymptotic => KsMode::Asymptotic,
        KsMode::Auto => {
            if n_x.max(n_y) <= KS_AUTO_CUTOVER {
                KsMode::Exact
            } else {
                KsMode::Asymptotic
            }
        }
        KsMode::Exact => {
            if n_x_g > std::f64::MAX / n_y_g {
                return Err(KsError::TooLongForExact);
            }
            KsMode::Exact
        }
    };

    match use_method {
        KsMode::Exact => {
            let lcm = (n_x_f / g_f) * n_y_f;
            let h = (stat * lcm).round();
            let stat = h / lcm;
            if h == 0.0 {
                Ok((stat, 1.0))
            } else {
                match alternative {
                    KsAlternative::TwoSided => {
                        if n_x == n_y {
                            Ok((stat, paths_outside_proportion(n_x, h)))
                        } else {
                            Ok((
                                stat,
                                1.0 - paths_inside_proportion(n_x, n_y, g, h),
                            ))
                        }
                    }
                    _ => {
                        if n_x == n_y {
                            let p = (0..(h as usize)).fold(1.0, |p, j| {
                                ((n_x - j) as f64) * p
                                    / (n_x_f + (j as f64) + 1.0)
                            });
                            Ok((stat, p))
                        } else {
                            let paths = paths_outside(n_x, n_y, g, h);
                            let bin = binomial(n_x + n_y, n_x);
                            Ok((stat, (paths as f64) / (bin as f64)))
                        }
                    }
                }
            }
        }
        KsMode::Asymptotic => {
            let ks_dist = KsTwoAsymptotic::new();

            match alternative {
                KsAlternative::TwoSided => {
                    let en = (n_x_f * n_y_f / (n_y_f + n_x_f)).sqrt();
                    Ok((stat, 1.0 - ks_dist.cdf(&(en * stat))))
                }
                _ => {
                    let m = n_x.max(n_y) as f64;
                    let n = n_x.min(n_y) as f64;

                    let z = (m * n / (m + n)).sqrt() * stat;
                    let expt = (-2.0 * z).mul_add(
                        z,
                        -2.0 * z * 2.0_f64.mul_add(n, m)
                            / (m * n * (m + n)).sqrt()
                            / 3.0,
                    );
                    let p = expt.exp();
                    Ok((stat, p))
                }
            }
        }
        KsMode::Auto => unreachable!(),
    }
}

#[allow(clippy::many_single_char_names)]
fn paths_outside(m: usize, n: usize, g: usize, h: f64) -> usize {
    let (m, n) = (m.max(n), m.min(n));
    let mg = m / g;
    let ng = n / g;
    let ng_f = ng as f64;
    let mg_f = mg as f64;

    let xj: Vec<usize> = (0..=n)
        .map(|j| (mg_f.mul_add(j as f64, h) / ng_f).ceil() as usize)
        .filter(|&x| x <= m)
        .collect();

    let lxj = xj.len();

    if lxj == 0 {
        binomial(m + n, n)
    } else {
        let mut b: Vec<usize> = (0..lxj).map(|_| 0).collect();
        b[0] = 1;
        for j in 1..lxj {
            let mut bj = binomial(xj[j] + j, j);
            for i in 0..j {
                let bin = binomial(xj[j] - xj[i] + j - i, j - i);
                let dec = bin * b[i];
                bj -= dec;
            }
            b[j] = bj;
        }
        let mut num_paths = 0;
        for j in 0..lxj {
            let bin = binomial((m - xj[j]) + (n - j), n - j);
            let term = b[j] * bin;
            num_paths += term
        }
        num_paths
    }
}

/// Compute the proportion of paths that pass outside the lines x - y = ± h
fn paths_outside_proportion(n: usize, h: f64) -> f64 {
    let mut p = 0.0;
    let n_f = n as f64;
    let k_max = (n_f / h) as usize;

    for k in (0..=k_max).rev() {
        let mut p1 = 1.0;
        for j in 0..(h as usize) {
            let j_f = j as f64;
            let k_f = k as f64;
            p1 = (k_f.mul_add(-h, n_f) - j_f) * p1
                / (k_f.mul_add(h, n_f) + j_f + 1.0);
        }
        p = p1 * (1.0 - p);
    }
    2.0 * p
}

/// Compute the proportion of paths that stay inside lines x - y = ± h
#[allow(clippy::many_single_char_names)]
fn paths_inside_proportion(m: usize, n: usize, g: usize, h: f64) -> f64 {
    let (m, n) = (m.max(n), m.min(n));
    let n_f = n as f64;
    let mg = m / g;
    let ng = n / g;

    let mg_f = mg as f64;
    let ng_f = ng as f64;

    let mut min_j = 0;
    let mut max_j = (n + 1).min((h / (mg as f64)).ceil() as usize);
    let mut cur_len = max_j - min_j;

    let len_a = (n + 1).min(2 * max_j + 2);
    let mut a: Vec<f64> = (0..len_a)
        .map(|i| if i >= min_j && i < max_j { 1.0 } else { 0.0 })
        .collect();
    for i in 1..=m {
        let i_f = i as f64;
        let last_min_j = min_j;
        let last_len = cur_len;

        min_j =
            ((ng_f.mul_add(i_f, -h) / mg_f).floor() + 1.0).max(0.0) as usize;
        min_j = min_j.min(n);

        max_j =
            (((ng_f.mul_add(i_f, h) / mg_f).floor() + 1.0) as usize).max(n + 1);
        if max_j <= min_j {
            return 0.0;
        }

        for j in 0..(max_j - min_j) {
            a[j] = a[min_j - last_min_j..max_j - last_min_j].iter().sum();
        }
        cur_len = max_j - min_j;
        if last_len > cur_len {
            for a_part in
                a.iter_mut().skip(max_j - min_j).take(last_len - cur_len)
            {
                *a_part = 0.0;
            }
        }
        let scaling_factor = i_f / (n_f + i_f);
        a = a.iter().map(|x| x * scaling_factor).collect();
    }
    a[max_j - min_j - 1]
}

fn mmul(xs: &[Vec<f64>], ys: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = xs.len();
    let mut zs = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            zs[i][j] =
                (0..m).fold(0.0, |acc, k| xs[i][k].mul_add(ys[k][j], acc))
        }
    }
    zs
}

fn mpow(xs: &[Vec<f64>], ea: i32, n: usize) -> (Vec<Vec<f64>>, i32) {
    let m = xs.len();
    if n == 1 {
        (xs.to_owned(), ea)
    } else {
        let (mut zs, mut ev) = mpow(xs, ea, n / 2);
        let ys = mmul(&zs, &zs);
        let eb = 2 * ev;
        if n % 2 == 0 {
            zs = ys;
            ev = eb;
        } else {
            zs = mmul(xs, &ys);
            ev = ea + eb;
        }
        if zs[m / 2][m / 2] > 1E140 {
            zs.iter_mut().for_each(|zs_i| {
                zs_i.iter_mut().for_each(|z| (*z) *= 1E-140);
            });
            ev += 140;
        }
        (zs, ev)
    }
}

// XXX: ks_cdf, mmul, and mpow are directlt translated from the c program in
// Wang, J., Tsang, W. W., & Marsaglia, G. (2003). Evaluating Kolmogorov's
//     distribution. Journal of Statistical Software, 8(18).
// They are not rusty. Please feel free to make them rusty 😘
#[allow(clippy::needless_range_loop)]
#[allow(clippy::many_single_char_names)]
fn ks_cdf(n: usize, d: f64) -> f64 {
    let nf = n as f64;
    let s: f64 = d * d * nf;
    if s > 7.24 || (s > 3.76 && n > 99) {
        2.0_f64.mul_add(
            -(-(2.000_071 + 0.331 / nf.sqrt() + 1.409 / nf) * s).exp(),
            1.0,
        )
    } else {
        let k: usize = ((nf * d) as usize) + 1;
        let m: usize = 2 * k - 1;
        let h: f64 = nf.mul_add(-d, k as f64);

        let mut hs = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                if ((i as i32) - (j as i32) + 1) >= 0 {
                    hs[i][j] = 1.0
                }
            }
        }

        for i in 0..m {
            hs[i][0] -= h.powi((i as i32) + 1);
            hs[m - 1][i] -= h.powi((m as i32) - (i as i32))
        }

        hs[m - 1][0] += if 2.0_f64.mul_add(h, -1.0) > 0.0 {
            2.0_f64.mul_add(h, -1.0).powi(m as i32)
        } else {
            0.0
        };

        for i in 0..m {
            for j in 0..m {
                if (i as i32) - (j as i32) + 1 > 0 {
                    for g in 1..=i - j + 1 {
                        hs[i][j] /= g as f64;
                    }
                }
            }
        }

        let (qs, mut eq) = mpow(&hs, 0, n);
        let mut s = qs[k - 1][k - 1];
        for i in 1..n {
            s *= (i as f64) / nf;
            if s < 1e-140 {
                s *= 1e140;
                eq -= 140;
            }
        }
        s * 10.0_f64.powi(eq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Gaussian;
    use crate::traits::Cdf;

    const TOL: f64 = 1E-12;

    #[test]
    fn ks_cdf_normal() {
        assert::close(ks_cdf(10, 0.274), 0.628_479_615_456_504_3, TOL);
    }

    #[test]
    fn ks_cdf_large_n() {
        assert::close(ks_cdf(1000, 0.074), 0.999_967_173_529_903_7, TOL);
    }

    #[test]
    fn ks_test_pval() {
        let xs: Vec<f64> =
            vec![0.42, 0.24, 0.86, 0.85, 0.82, 0.82, 0.25, 0.78, 0.13, 0.27];

        let g = Gaussian::standard();
        let cdf = |x: f64| g.cdf(&x);
        let (ks, p) = ks_test(&xs, cdf);

        assert::close(ks, 0.551_716_786_654_561_1, TOL);
        assert::close(p, 0.002_180_450_252_694_976_5, TOL);
    }

    #[test]
    fn ks_two_sample_exact() {
        let xs = [
            0.956_920_26,
            1.134_881_2,
            -0.765_792_39,
            -0.580_656_53,
            -0.051_223_93,
            0.715_987_54,
            1.398_735_28,
            0.427_905_27,
            1.845_807_64,
            0.642_285_21,
        ];

        let ys = [
            0.694_867_8,
            -0.374_182_5,
            0.366_572_79,
            1.158_341_74,
            -0.324_217_06,
            -0.384_992_95,
            1.449_769_91,
            0.250_460_8,
            -0.536_947_74,
            1.422_219_93,
        ];

        let (stat, alpha) =
            ks_two_sample(&xs, &ys, KsMode::Exact, KsAlternative::TwoSided)
                .unwrap();
        assert::close(stat, 0.3, 1E-8);
        assert::close(alpha, 0.786_929_788_477_776_1, 1E-8);

        let (stat, alpha) =
            ks_two_sample(&xs, &ys, KsMode::Exact, KsAlternative::Less)
                .unwrap();
        assert::close(stat, 0.3, 1E-8);
        assert::close(alpha, 0.419_580_419_580_419_53, 1E-8);

        let (stat, alpha) =
            ks_two_sample(&xs, &ys, KsMode::Exact, KsAlternative::Greater)
                .unwrap();
        assert::close(stat, 0.2, 1E-8);
        assert::close(alpha, 0.681_818_181_818_181_8, 1E-8);
    }

    #[test]
    fn ks_two_sample_asymp() {
        let xs = [
            0.956_920_26,
            1.134_881_2,
            -0.765_792_39,
            -0.580_656_53,
            -0.051_223_93,
            0.715_987_54,
            1.398_735_28,
            0.427_905_27,
            1.845_807_64,
            0.642_285_21,
        ];

        let ys = [
            0.694_867_8,
            -0.374_182_5,
            0.366_572_79,
            1.158_341_74,
            -0.324_217_06,
            -0.384_992_95,
            1.449_769_91,
            0.250_460_8,
            -0.536_947_74,
            1.422_219_93,
        ];

        let (stat, alpha) = ks_two_sample(
            &xs,
            &ys,
            KsMode::Asymptotic,
            KsAlternative::TwoSided,
        )
        .unwrap();
        assert::close(stat, 0.3, 1E-8);
        assert::close(alpha, 0.759_097_838_420_394_8, 1E-8);

        let (stat, alpha) =
            ks_two_sample(&xs, &ys, KsMode::Asymptotic, KsAlternative::Less)
                .unwrap();
        assert::close(stat, 0.3, 1E-8);
        assert::close(alpha, 0.301_194_211_912_202_14, 1E-8);

        let (stat, alpha) =
            ks_two_sample(&xs, &ys, KsMode::Asymptotic, KsAlternative::Greater)
                .unwrap();
        assert::close(stat, 0.2, 1E-8);
        assert::close(alpha, 0.548_811_636_094_026_4, 1E-8);
    }
}
