extern crate rand;

use self::rand::distributions::Open01;
use self::rand::Rng;
use std::ops::AddAssign;

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
/// # use rv::utils::cumsum;
/// #
/// let xs: Vec<i32> = vec![1, 1, 2, 1];
/// assert_eq!(cumsum(&xs), vec![1, 2, 4, 5]);
/// ```
pub fn cumsum<T>(xs: &[T]) -> Vec<T>
where
    T: AddAssign + Clone,
{
    let mut summed: Vec<T> = xs.to_vec();
    for i in 1..xs.len() {
        summed[i] += summed[i - 1].clone();
    }
    summed
}

/// Draw `n` indices in proportion to their `weights`
pub fn pflip(weights: &[f64], n: usize, rng: &mut impl Rng) -> Vec<usize> {
    if weights.is_empty() {
        panic!("Empty container");
    }
    let ws: Vec<f64> = cumsum(weights);
    let scale: f64 = *ws.last().unwrap();
    let u = rand::distributions::Uniform::new(0.0, 1.0);

    (0..n)
        .map(|_| {
            let r = rng.sample(u) * scale;
            match ws.iter().position(|&w| w > r) {
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
/// Assumes that the sum of the weights is 0 (the sum of `exp(ln_weights)` is
/// 1).
///
/// # Examples
///
/// ```rust
/// extern crate rand;
/// extern crate rv;
///
/// use rv::utils::ln_pflip;
///
/// let weights: Vec<f64> = vec![0.4, 0.2, 0.3, 0.1];
/// let ln_weights: Vec<f64> = weights.iter().map(|&w| w.ln()).collect();
///
/// let xs = ln_pflip(&ln_weights, 100, &mut rand::thread_rng());
///
/// assert_eq!(xs.len(), 100);
/// assert!(xs.iter().all(|&x| x <= 3));
/// assert!(!xs.iter().any(|&x| x > 3));
/// ```
pub fn ln_pflip<R: Rng>(
    ln_weights: &[f64],
    n: usize,
    rng: &mut R,
) -> Vec<usize> {
    let mut cdf: Vec<f64> = ln_weights.iter().map(|w| w.exp()).collect();

    // doing this instead of calling pflip shaves about 30% off the runtime.
    for i in 1..cdf.len() {
        cdf[i] += cdf[i - 1];
    }

    (0..n)
        .map(|_| {
            let r = rng.sample(Open01);
            cdf.iter()
                .position(|&w| w > r)
                .expect(format!("Could not draw from {:?}", cdf).as_str())
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
/// use rv::utils::argmax;
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
        for i in 1..xs.len() {
            let x = &xs[i];
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

#[cfg(test)]
mod tests {
    use super::*;
    extern crate assert;

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

}
