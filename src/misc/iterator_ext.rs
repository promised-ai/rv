use core::iter::Map;
use itertools::TupleWindows;
use itertools::{iterate, Itertools};

/// Provides extension methods for iterators over `f64` values.
pub trait IteratorExt: Iterator<Item = f64> + Sized {
    fn aitken(
        self,
    ) -> Map<TupleWindows<Self, (f64, f64, f64)>, fn((f64, f64, f64)) -> f64>
    {
        self.tuple_windows::<(_, _, _)>().map(|(x, x2, x3)| {
            let dx = x2 - x;
            let dx2 = x3 - x2;
            let ddx = dx2 - dx;
            x - dx.powi(2) / ddx
        })
    }

    fn limit(self, tol: f64) -> f64 {
        self.tuple_windows::<(_, _)>()
            .filter_map(
                |(a, b)| {
                    if (a - b).abs() < tol {
                        Some(b)
                    } else {
                        None
                    }
                },
            )
            .nth(0)
            .unwrap()
    }
}

impl<T> IteratorExt for T where T: Iterator<Item = f64> + Sized {}

#[cfg(test)]
mod tests {
    use super::*;
    use num::Integer;

    #[test]
    fn test_aitken_limit() {
        let seq = (0..)
            .map(|n| {
                let sign = if n.is_even() { 1.0 } else { -1.0 };

                sign / (2 * n + 1) as f64
            })
            .scan(0.0, |acc, x| {
                *acc += x;
                Some(*acc)
            });
        let limit = seq.aitken().limit(1e-5);
        let pi_over_4 = std::f64::consts::PI / 4.0;
        assert!((limit - pi_over_4).abs() < 1e-5, "The limit calculated using Aitken's Δ² process did not converge to π/4 within the tolerance.");
    }
}
