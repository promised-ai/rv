const QUAD_EPS: f64 = 1E-8;

fn simpsons_rule<F>(func: &F, a: f64, b: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let c = (a + b) / 2.0;
    let h3 = (b - a).abs() / 6.0;
    h3 * (func(a) + 4.0 * func(c) + func(b))
}

fn recursive_asr<F>(func: &F, a: f64, b: f64, eps: f64, whole: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let c = (a + b) / 2.0;
    let left = simpsons_rule(&func, a, c);
    let right = simpsons_rule(&func, c, b);
    if (left + right - whole).abs() <= 15.0 * eps {
        left + right + (left + right - whole) / 15.0
    } else {
        recursive_asr(func, a, c, eps / 2.0, left)
            + recursive_asr(func, c, b, eps / 2.0, right)
    }
}

/// Adaptive Simpson's quadrature
///
/// # Example
///
/// Integrate f: x<sup>2</sup> over the interval [0, 1].
///
/// ```
/// use rv::misc::quad;
///
/// let func = |x: f64| x.powi(2);
/// let q = quad(func, 0.0, 1.0);
///
/// assert!((q - 1.0/3.0).abs() < 1E-8);
/// ```
pub fn quad<F>(func: F, a: f64, b: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    recursive_asr(&func, a, b, QUAD_EPS, simpsons_rule(&func, a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn quad_of_x2() {
        let func = |x: f64| x.powi(2);
        let q = quad(func, 0.0, 1.0);
        assert::close(q, 1.0 / 3.0, QUAD_EPS);
    }

    #[test]
    fn quad_of_sin() {
        let func = |x: f64| x.sin();
        let q = quad(func, 0.0, 5.0 * PI);
        assert::close(q, 2.0, QUAD_EPS);
    }
}
