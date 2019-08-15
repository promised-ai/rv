//! Root finding methods.

use crate::{OptimizeError, Result};


/// Newton-Raphson root finding algorithm.
pub fn newton_raphson<F>(
    x0: f64,
    tol: f64,
    max_iter: usize,
    f: F,
) -> Result<f64>
where
    F: Fn(f64) -> (f64, f64),
{
    let mut x = x0;

    for _ in 0..max_iter {
        let (f_x, g_x) = f(x);
        // Break if we're likely to become
        if g_x.abs() < std::f64::EPSILON {
            return Err(OptimizeError::NumericalDivergence);
        }

        let x_prev = x;
        // preform NR update
        x -= f_x / g_x;

        if (x - x_prev).abs() <= tol {
            return Ok(x);
        }
    }

    Err(OptimizeError::MaxIterationReached)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_raphson_x_squared() {
        fn fg(x: f64) -> (f64, f64) {
            let y = &x * &x;
            let dy_dx = 2.0 * x;
            (y, dy_dx)
        }

        let min_res = newton_raphson(10.0, 1E-5, 1_000, fg);
        assert!(min_res.is_ok(), "Should return an Ok");
        assert::close(min_res.unwrap(), 0.0, 1E-5);
    }

    #[test]
    fn newton_raphson_zero_deriv() {
        fn fg(x: f64) -> (f64, f64) {
            let y = &x * &x;
            let dy_dx = 2.0 * x;
            (y, dy_dx)
        }

        match newton_raphson(0.0, 1E-5, 1_000, fg) {
            Err(OptimizeError::NumericalDivergence) => (),
            _ => panic!("Should not be anything but Numerical Divergence"),
        }
    }
}
