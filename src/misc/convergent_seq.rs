// use core::iter::Map;
use itertools::Itertools;
use num::Zero;
// use itertools::TupleWindows;

/// A trait for sequences that can be checked for convergence.
pub trait ConvergentSequence: Iterator<Item = f64> + Sized {
    /// Applies Aitken's Δ² process to accelerate the convergence of a sequence.
    /// See https://en.wikipedia.org/wiki/Aitken%27s_delta-squared_process and
    /// https://en.wikipedia.org/wiki/Shanks_transformation
    ///
    /// # Returns
    ///
    /// An iterator over the accelerated sequence.
    fn aitken(self) -> impl Iterator<Item = f64> {
        self.tuple_windows::<(_, _, _)>().filter_map(|(x, y, z)| {
            let dx = z - y;
            let dx2 = y - x;
            let ddx = dx - dx2;

            // We can't handle a segment like [2,4,6]
            // But e.g. [2, 2, 2] may have already converged
            if ddx.is_zero() {
                if dx.is_zero() {
                    Some(z)
                } else {
                    None
                }
            } else {
                Some(z - dx.powi(2) / dx)
            }
        })
    }

    /// Finds the limit of the sequence within a given tolerance using Aitken's
    /// Δ² process. This should *only* be applied to sequences that are known to
    /// converge.
    ///
    /// # Arguments
    ///
    /// * `tol` - The tolerance within which to find the limit.
    ///
    /// # Returns
    ///
    /// The limit of the sequence as a floating-point number.
    ///
    /// # Panics
    ///
    /// Runs forever if the sequence does not converge within the given
    /// tolerance.
    fn limit(self, tol: f64) -> f64 {
        self.aitken()
            .aitken()
            .aitken()
            .aitken()
            .tuple_windows::<(_, _)>()
            .filter_map(
                |(a, b)| {
                    if (a - b).abs() < tol {
                        Some(b)
                    } else {
                        None
                    }
                },
            )
            .next()
            .unwrap()
    }
}

impl<T> ConvergentSequence for T where T: Iterator<Item = f64> + Sized {}

#[cfg(test)]
mod tests {
    use super::*;
    use num::Integer;

    #[test]
    fn test_aitken_limit() {
        let seq = (0..)
            .map(|n| {
                let sign = if n.is_even() { 1.0 } else { -1.0 };
                let val = sign / (2 * n + 1) as f64;
                dbg!(val);
                val
            })
            .scan(0.0, |acc, x| {
                *acc += x;
                Some(*acc)
            });
        let limit = seq.limit(1e-10);
        let pi_over_4 = std::f64::consts::PI / 4.0;
        assert!((limit - pi_over_4).abs() < 1e-10, "The limit calculated using Aitken's Δ² process did not converge to π/4 within the tolerance.");
    }
}
