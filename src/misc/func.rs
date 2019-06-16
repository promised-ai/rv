use crate::consts::LN_PI;
use rand::distributions::Open01;
use rand::Rng;
use special::Gamma;
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
pub fn vec_to_string<T: Debug>(xs: &Vec<T>, max_entries: usize) -> String {
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
/// # extern crate rv;
/// # use rv::misc::cumsum;
/// #
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
/// extern crate rand;
/// extern crate rv;
///
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
            if x > maxval {
                maxval = x;
                max_ixs = vec![i];
            } else if x == maxval {
                max_ixs.push(i);
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

// TODO: Replace this with x.mod_ecu(y) when `euclidean_division` is
// stabilized.
/// Euclidean modulo
///
/// # Example
///
/// Taken from the [rust
/// documentation](https://doc.rust-lang.org/std/primitive.f64.html#method.mod_euc)
///
/// ```rust
/// # extern crate rv;
/// # use rv::misc::mod_euc;
/// let a: f64 = 7.0;
/// let b = 4.0;
/// assert_eq!(mod_euc(a, b), 3.0);
/// assert_eq!(mod_euc(-a, b), 1.0);
/// assert_eq!(mod_euc(a, -b), 3.0);
/// assert_eq!(mod_euc(-a, -b), 1.0);
/// // limitation due to round-off error
/// assert!(mod_euc(-std::f64::EPSILON, 3.0) != 0.0);
/// ```
pub fn mod_euc(lhs: f64, rhs: f64) -> f64 {
    let r = lhs % rhs;
    if r < 0.0 {
        return if rhs > 0.0 { r + rhs } else { r - rhs };
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn argmax_empty_is_empty() {
        let xs: Vec<f64> = vec![];
        assert_eq!(argmax(&xs), vec![]);
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
}
