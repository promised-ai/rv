/// Univariate one-sample [Kolmogorov-Smirnov](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
/// test.
///
/// Given a set of samples, `xs`, and a distribution `F`, the KS test
/// determines whether `xs` were generated by `F`.
///
/// # Example
///
/// ```rust
/// # extern crate rv;
/// extern crate rand;
///
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
pub fn ks_test<F: Fn(f64) -> f64>(xs: &Vec<f64>, cdf: F) -> (f64, f64) {
    let mut xs_r: Vec<f64> = xs.clone().to_vec();
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

fn mmul(xs: &Vec<Vec<f64>>, ys: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let m = xs.len();
    let mut zs = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            zs[i][j] = (0..m).fold(0.0, |acc, k| acc + xs[i][k] * ys[k][j])
        }
    }
    zs
}

fn mpow(xs: &Vec<Vec<f64>>, ea: i32, n: usize) -> (Vec<Vec<f64>>, i32) {
    let m = xs.len();
    if n == 1 {
        (xs.clone(), ea)
    } else {
        let (mut zs, mut ev) = mpow(xs, ea, n / 2);
        let ys = mmul(&zs, &zs);
        let eb = 2 * ev;
        if n % 2 == 0 {
            zs = ys;
            ev = eb;
        } else {
            zs = mmul(&xs, &ys);
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
fn ks_cdf(n: usize, d: f64) -> f64 {
    let nf = n as f64;
    let s: f64 = d * d * nf;
    if s > 7.24 || (s > 3.76 && n > 99) {
        1.0 - 2.0 * (-(2.000071 + 0.331 / nf.sqrt() + 1.409 / nf) * s).exp()
    } else {
        let k: usize = ((nf * d) as usize) + 1;
        let m: usize = 2 * k - 1;
        let h: f64 = (k as f64) - nf * d;

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

        hs[m - 1][0] += if 2.0 * h - 1.0 > 0.0 {
            (2.0 * h - 1.0).powi(m as i32)
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
    extern crate assert;
    use dist::Gaussian;
    use traits::Cdf;

    const TOL: f64 = 1E-12;

    #[test]
    fn ks_cdf_normal() {
        assert::close(ks_cdf(10, 0.274), 0.6284796154565043, TOL);
    }

    #[test]
    fn ks_cdf_large_n() {
        assert::close(ks_cdf(1000, 0.074), 0.9999671735299037, TOL);
    }

    #[test]
    fn ks_test_pval() {
        let xs: Vec<f64> =
            vec![0.42, 0.24, 0.86, 0.85, 0.82, 0.82, 0.25, 0.78, 0.13, 0.27];

        let g = Gaussian::standard();
        let cdf = |x: f64| g.cdf(&x);
        let (ks, p) = ks_test(&xs, cdf);

        assert::close(ks, 0.55171678665456114, TOL);
        assert::close(p, 0.0021804502526949765, TOL);
    }
}
