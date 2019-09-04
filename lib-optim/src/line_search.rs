//! Line search methods

use crate::OptimizeError;
use log::debug;

/// Wolfe Algorithm Parameters
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WolfeParams {
    /// Slope coefficient number 1
    pub c1: f64,
    /// Slope coefficient number 2
    pub c2: f64,
    /// Minimum acceptable value of x.
    pub amin: f64,
    /// Maximal acceptable value of x.
    pub amax: f64,
    /// Desired tolerance
    pub xtol: f64,
    /// Maximum number of iterators
    pub max_iter: usize,
}

impl Default for WolfeParams {
    fn default() -> Self {
        Self {
            c1: 1E-4,
            c2: 0.9,
            amax: 50.0,
            amin: 1E-8,
            xtol: 1E-14,
            max_iter: 10,
        }
    }
}

/// Find the minimum of a quadratic polynomial that goes through
/// the points (a, fa) and (b, fb) with slope fpa at a.
#[inline]
fn quad_min(a: f64, fa: f64, fpa: f64, b: f64, fb: f64) -> Option<f64> {
    let dab = b - a;
    if dab == 0.0 {
        return None;
    }
    let c2 = (2.0 / dab) * (((fa - fb) / dab) - fpa);
    if c2 == 0.0 {
        None
    } else {
        Some(a - fa / c2)
    }
}

/// Finds the minimizer for a cubic polynomial that passes through the points
/// (a, fa), (b, fb), (c, fc) with derivative (a, fpa).
#[inline]
fn cubic_min(
    a: f64,
    fa: f64,
    fpa: f64,
    b: f64,
    fb: f64,
    c: f64,
    fc: f64,
) -> Option<f64> {
    let c0 = fpa;
    let db = b - a;
    let dc = c - a;
    let denom = (db * db).powi(2) * (db - dc);
    if denom == 0.0 {
        return None;
    }
    let fu = fb - fa - c0 * db;
    let fv = fc - fa - c0 * dc;
    let c2 = (dc * dc * fu - db * db * fv) / denom;
    if c2 == 0.0 {
        return None;
    }
    let c1 = (-dc * dc * dc * fu + db * db * db * fv) / denom;
    let radical = c1 * c1 - 3.0 * c2 * c0;
    if radical < 0.0 {
        None
    } else {
        let res = a + (-c1 + radical.sqrt()) / (3.0 * c2);
        if res.is_nan() {
            None
        } else {
            Some(res)
        }
    }
}

/// Wolfe Zoom function
/// Reference: Algorithm 3.6 of Numerical Optimization, Jorge Nocedal & Stephen Wright, 1960.
fn zoom<F>(
    alpha_lo: f64,
    alpha_hi: f64,
    phi_lo: f64,
    derphi_lo: f64,
    phi_hi: f64,
    phi_0: f64,
    derphi_0: f64,
    c1: f64,
    c2: f64,
    max_iter: usize,
    f: F,
) -> Result<f64, OptimizeError>
where
    F: Fn(f64) -> (f64, f64),
{
    debug!("zoom(a_lo = {}, a_hi = {}, phi_lo = {}, derphi_lo = {}, phi_hi = {}, phi_0 = {}, derphi_0 = {})", alpha_lo, alpha_hi, phi_lo, derphi_lo, phi_hi, phi_0, derphi_0);
    const DELTA1: f64 = 0.2;
    const DELTA2: f64 = 0.1;

    let mut alpha_hi = alpha_hi;
    let mut alpha_lo = alpha_lo;
    let mut phi_lo = phi_lo;
    let mut derphi_lo = derphi_lo;
    let mut phi_hi = phi_hi;

    let mut phi_rec = phi_0;
    let mut alpha_rec = 0.0;
    let mut aj: Option<f64> = None;
    let mut cchk: f64 = 0.0;

    for i in 0..max_iter {
        debug!("iteration = {}, a_lo = {:0.3}, a_rec = {:0.3}, a_hi = {:0.3}, a_j = {:0.3}",
            i, alpha_lo, alpha_rec, alpha_hi, aj.unwrap_or(-1.0)
        );
        let delta_alpha = alpha_hi - alpha_lo;
        let (a, b) = if delta_alpha < 0.0 {
            (alpha_hi, alpha_lo)
        } else {
            (alpha_lo, alpha_hi)
        };

        if i > 0 {
            cchk = DELTA1 * delta_alpha;
            aj = cubic_min(
                alpha_lo, phi_lo, derphi_lo, alpha_hi, phi_hi, alpha_rec,
                phi_rec,
            );
            debug!("cubic guess = {}", aj.unwrap_or(-1.0));
        }

        if i == 0
            || aj.is_none()
            || aj.unwrap().is_nan()
            || aj.unwrap() > b - cchk
            || aj.unwrap() < a + cchk
        {
            let qchk = DELTA2 * delta_alpha;
            aj = quad_min(alpha_lo, phi_lo, derphi_lo, alpha_hi, phi_hi);
            // println!("            quad guess = {}", aj.unwrap_or(-1.0));
            if aj.is_none()
                || aj.unwrap().is_nan()
                || aj.unwrap() > b - qchk
                || aj.unwrap() < a + qchk
            {
                aj = Some(alpha_lo + 0.5 * delta_alpha);
                debug!("bisect guess = {}", aj.unwrap_or(-1.0));
            }
        }

        let (phi_aj, derphi_aj) = f(aj.unwrap());
        debug!("phi_aj = {}, derphi_aj = {}", phi_aj, derphi_aj);
        debug!(
            "phi_0 + c1 * aj * derphi_0 = {}",
            phi_0 + c1 * aj.unwrap() * derphi_0
        );
        debug!("del = {}", phi_aj - (phi_0 + c1 * aj.unwrap() * derphi_0));

        if phi_aj > phi_0 + c1 * aj.unwrap() * derphi_0 || phi_aj >= phi_lo {
            alpha_rec = alpha_hi;
            phi_rec = phi_hi;
            alpha_hi = aj.unwrap();
            phi_hi = phi_aj;
        } else {
            // println!("            -c2 * derphi_0 = {}", -c2 * derphi_0);
            if derphi_aj.abs() <= -c2 * derphi_0 {
                return Ok(aj.unwrap());
            } else if derphi_aj * delta_alpha >= 0.0 {
                phi_rec = phi_hi;
                alpha_rec = alpha_hi;
                alpha_hi = alpha_lo;
                phi_hi = phi_lo;
            } else {
                phi_rec = phi_lo;
                alpha_rec = alpha_lo;
            }
            alpha_lo = aj.unwrap();
            phi_lo = phi_aj;
            derphi_lo = derphi_aj;
        }
    }
    Err(OptimizeError::MaxIterationReached)
}

/// Wolfe Line Search Method
/// Finds a point that "roughly" minimizes the function `f(x)` s.t. `x > 0`.
/// Reference: Algorithm 3.3 of Numerical Optimization, Jorge Nocedal & Stephen Wright, 1960.
pub fn wolfe_search<F>(params: &WolfeParams, f: F) -> Result<f64, OptimizeError>
where
    F: Fn(f64) -> (f64, f64),
{
    let mut alpha0 = 0.0;
    let mut alpha1 = 1.0;

    let (phi_0, derphi_0) = f(alpha0);
    let (mut phi_a0, mut derphi_a0) = (phi_0, derphi_0);
    let (mut phi_a1, mut derphi_a1) = f(alpha1);

    debug!(
        "wolfe_search (init): phi_0 = {}, derphi_0 = {}",
        phi_0, derphi_0
    );

    for i in 0..params.max_iter {
        debug!(
            "    wolfe_search: i = {}, alpha0 = {}, alpha1 = {}",
            i, alpha0, alpha1
        );
        if alpha1 == 0.0 {
            return Err(OptimizeError::RoundingError);
        }

        if (phi_a1 > phi_0 + params.c1 * alpha1 * derphi_0)
            || (phi_a1 >= phi_a0 && i > 0)
        {
            return zoom(
                alpha0,
                alpha1,
                phi_a0,
                derphi_a0,
                phi_a1,
                phi_0,
                derphi_0,
                params.c1,
                params.c2,
                params.max_iter,
                f,
            );
        }

        if derphi_a1.abs() <= -params.c2 * derphi_0 {
            return Ok(alpha1);
        }

        if derphi_a1 >= 0.0 {
            return zoom(
                alpha1,
                alpha0,
                phi_a1,
                derphi_a1,
                phi_a0,
                phi_0,
                derphi_0,
                params.c1,
                params.c2,
                params.max_iter,
                f,
            );
        }
        let alpha2 = (2.0 * alpha1).max(params.amax);
        alpha0 = alpha1;
        alpha1 = alpha2;
        phi_a0 = phi_a1;
        derphi_a0 = derphi_a1;
        let phi_derphi_a1 = f(alpha1);
        phi_a1 = phi_derphi_a1.0;
        derphi_a1 = phi_derphi_a1.1;
    }

    Err(OptimizeError::MaxIterationReached)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wolfe_search_x_squared() {
        let res = wolfe_search(&WolfeParams::default(), |x| {
            let y = (x - 1.0).powi(2) + (x - 1.0);
            let dy_dx = 2.0 * (x - 1.0) + 1.0;
            (y, dy_dx)
        });

        assert!(res.is_ok());
        assert::close(res.unwrap(), 0.5, 1E-10);
    }

    #[test]
    fn wolfe_search_x_cubed() {
        let res = wolfe_search(&WolfeParams::default(), |x| {
            let y = -(x - 1.0).powi(3) - (x - 1.0).powi(2);
            let dy_dx = -3.0 * x.powi(2) + 4.0 * x - 1.0;
            (y, dy_dx)
        });

        assert!(res.is_ok());
        assert::close(res.unwrap(), 0.5, 1E-10);
    }

    #[test]
    fn wolfe_search_multiregion() {
        let res = wolfe_search(&WolfeParams::default(), |x| {
            let pi: f64 = std::f64::consts::PI;
            let y = (-x).exp() * (2.0 * pi * x - pi / 2.0).sin().powi(2);
            let dy_dx = -(-x).exp()
                * (2.0 * pi * x).cos()
                * (4.0 * pi * (2.0 * pi * x).sin() + (2.0 * pi * x).cos());
            (y, dy_dx)
        });

        assert!(res.is_ok());
        assert::close(res.unwrap(), 1.0, 1E-10);
    }
}
