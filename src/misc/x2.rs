extern crate special;

use self::special::Gamma;

/// Χ<sup>2</sup> (Chi-squared) test.
///
/// # Example
///
/// Test whether the observed counts were generated uniformly.
///
/// ```
/// # extern crate rv;
/// use rv::misc::x2_test;
///
/// // The observed counts/frequencies
/// let f_obs: Vec<u32> = vec![28, 31, 40, 35];
///
/// // The probabilty with which each entry should occur
/// let ps: Vec<f64> = vec![0.25; 4];
///
/// let (stat, p) = x2_test(&f_obs, &ps);
/// assert!(p > 0.05);
/// ```
pub fn x2_test(f_obs: &[u32], ps: &[f64]) -> (f64, f64) {
    let k = f_obs.len();
    let nf = f_obs.iter().fold(0, |acc, ct| acc + ct) as f64;
    let x2 = nf
        * f_obs.iter().zip(ps.iter()).fold(0.0, |acc, (&o, &p)| {
            acc + (f64::from(o) / nf - p).powi(2) / p
        });

    let df = (k - 1) as f64;
    let p = 1.0 - (x2 / 2.0).inc_gamma(df / 2.0);
    (x2, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate assert;

    const TOL: f64 = 1E-12;

    #[test]
    fn gof() {
        let f_obs: Vec<u32> = vec![28, 31, 40, 35];
        let ps: Vec<f64> = vec![0.25; 4];
        let (x2, p) = x2_test(&f_obs, &ps);

        assert::close(x2, 2.4179104477611939, TOL);
        assert::close(p, 0.49030930696538833, TOL);
    }
}
