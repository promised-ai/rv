const QUAD_EPS: f64 = 1E-8;

#[derive(Debug, Clone)]
pub(crate) struct QuadConfig<'a> {
    pub max_depth: u32,
    pub err_tol: f64,
    pub seed_points: Option<&'a Vec<f64>>,
}

impl<'a> Default for QuadConfig<'a> {
    fn default() -> Self {
        QuadConfig {
            max_depth: 12,
            err_tol: 1e-16,
            seed_points: None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn quad_recr<F>(
    func: &F,
    a: f64,
    fa: f64,
    m: f64,
    fm: f64,
    b: f64,
    fb: f64,
    err: f64,
    whole: f64,
    depth: u32,
    max_depth: u32,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let (ml, fml, left) = simpsons_rule(&func, a, fa, m, fm);
    let (mr, fmr, right) = simpsons_rule(&func, m, fm, b, fb);
    let eps = left + right - whole;
    if eps.abs() <= 15.0 * err || depth == max_depth {
        left + right + eps / 15.0
    } else {
        quad_recr(
            func,
            a,
            fa,
            ml,
            fml,
            m,
            fm,
            err / 2.0,
            left,
            depth + 1,
            max_depth,
        ) + quad_recr(
            func,
            m,
            fm,
            mr,
            fmr,
            b,
            fb,
            err / 2.0,
            right,
            depth + 1,
            max_depth,
        )
    }
}

#[allow(clippy::too_many_arguments, clippy::clippy::many_single_char_names)]
pub(crate) fn quadp<F>(f: &F, a: f64, b: f64, config: QuadConfig) -> f64
where
    F: Fn(f64) -> f64,
{
    let default_points = vec![a, (a + b) / 2.0, b];
    let points = match config.seed_points {
        Some(points) => points,
        None => &default_points,
    };

    let tol = config.err_tol / (points.len() + 1) as f64;
    let fa = f(a);

    let (c, fc, res) = points.iter().fold((a, fa, 0.0), |(a, fa, res), &b| {
        let fb = f(b);
        let (m, fm, q) = simpsons_rule(&f, a, fa, b, fb);
        (
            b,
            fb,
            res + quad_recr(
                &f,
                a,
                fa,
                m,
                fm,
                b,
                fb,
                tol,
                q,
                1,
                config.max_depth,
            ),
        )
    });

    let fb = f(b);
    let (m, fm, q) = simpsons_rule(&f, c, fc, b, fb);
    res + quad_recr(&f, c, fc, m, fm, b, fb, tol, q, 1, config.max_depth)
}

#[inline]
fn simpsons_rule<F>(
    func: &F,
    a: f64,
    fa: f64,
    b: f64,
    fb: f64,
) -> (f64, f64, f64)
where
    F: Fn(f64) -> f64,
{
    let c = (a + b) / 2.0;
    let h3 = (b - a).abs() / 6.0;
    let fc = func(c);
    (c, fc, h3 * (fa + 4.0 * fc + fb))
}

#[allow(clippy::too_many_arguments)]
fn recursive_asr<F>(
    func: &F,
    a: f64,
    fa: f64,
    b: f64,
    fb: f64,
    eps: f64,
    whole: f64,
    c: f64,
    fc: f64,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let (cl, fcl, left) = simpsons_rule(&func, a, fa, c, fc);
    let (cr, fcr, right) = simpsons_rule(&func, c, fc, b, fb);
    if (left + right - whole).abs() <= 15.0 * eps {
        left + right + (left + right - whole) / 15.0
    } else {
        recursive_asr(func, a, fa, c, fc, eps / 2.0, left, cl, fcl)
            + recursive_asr(func, c, fc, b, fb, eps / 2.0, right, cr, fcr)
    }
}

/// Adaptive Simpson's quadrature with user supplied error tolerance
///
/// # Example
///
/// Integrate f: x<sup>2</sup> over the interval [0, 1].
///
/// ```
/// use rv::misc::quad_eps;
///
/// let func = |x: f64| x.powi(2);
/// let q = quad_eps(func, 0.0, 1.0, Some(1E-10));
///
/// assert!((q - 1.0/3.0).abs() < 1E-10);
/// ```
pub fn quad_eps<F>(func: F, a: f64, b: f64, eps_opt: Option<f64>) -> f64
where
    F: Fn(f64) -> f64,
{
    let eps = eps_opt.unwrap_or(QUAD_EPS);
    let fa = func(a);
    let fb = func(b);
    let (c, fc, whole) = simpsons_rule(&func, a, fa, b, fb);
    recursive_asr(&func, a, fa, b, fb, eps, whole, c, fc)
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
    quad_eps(func, a, b, None)
}

//------------------------------------------
#[inline]
fn try_simpsons_rule<F, E>(
    func: &F,
    a: f64,
    fa: f64,
    b: f64,
    fb: f64,
) -> Result<(f64, f64, f64), E>
where
    F: Fn(f64) -> Result<f64, E>,
{
    let c = (a + b) / 2.0;
    let fc = func(c)?;
    let h3 = (b - a).abs() / 6.0;
    Ok((c, fc, h3 * (fa + 4.0 * fc + fb)))
}

#[allow(clippy::too_many_arguments)]
fn try_recursive_asr<F, E>(
    func: &F,
    a: f64,
    fa: f64,
    b: f64,
    fb: f64,
    eps: f64,
    whole: f64,
    c: f64,
    fc: f64,
) -> Result<f64, E>
where
    F: Fn(f64) -> Result<f64, E>,
{
    let (cl, fcl, left) = try_simpsons_rule(&func, a, fa, c, fc)?;
    let (cr, fcr, right) = try_simpsons_rule(&func, c, fc, b, fb)?;
    if (left + right - whole).abs() <= 15.0 * eps {
        Ok(left + right + (left + right - whole) / 15.0)
    } else {
        try_recursive_asr(func, a, fa, c, fc, eps / 2.0, left, cl, fcl)
            .and_then(|left| {
                try_recursive_asr(func, c, fc, b, fb, eps / 2.0, right, cr, fcr)
                    .map(|right| left + right)
            })
    }
}

/// Adaptive Simpson's quadrature with user supplied error tolerance over
/// functions that can fail.
///
/// # Example
///
/// Integrate f: x<sup>2</sup> over the interval [0, 1].
///
/// ```
/// use rv::misc::try_quad_eps;
///
/// let func = |x: f64| {
///     if x > 2.0 {
///         Err(String::from("> 2.0"))
///     } else {
///         Ok(x.powi(2))
///     }
/// };
/// let q = try_quad_eps(func, 0.0, 1.0, Some(1E-10)).unwrap();
///
/// assert!((q - 1.0/3.0).abs() < 1E-10);
/// ```
pub fn try_quad_eps<F, E>(
    func: F,
    a: f64,
    b: f64,
    eps_opt: Option<f64>,
) -> Result<f64, E>
where
    F: Fn(f64) -> Result<f64, E>,
{
    let eps = eps_opt.unwrap_or(QUAD_EPS);
    let fa: f64 = func(a)?;
    let fb: f64 = func(b)?;
    let (c, fc, whole) = try_simpsons_rule(&func, a, fa, b, fb)?;
    try_recursive_asr(&func, a, fa, b, fb, eps, whole, c, fc)
}

/// Adaptive Simpson's quadrature on functions that can fail.
///
/// # Example
///
/// Integrate f: x<sup>2</sup> over the interval [0, 1].
///
/// ```
/// use rv::misc::try_quad;
///
/// let func = |x: f64| {
///     if x > 2.0 {
///         Err(String::from("> 2.0"))
///     } else {
///         Ok(x.powi(2))
///     }
/// };
/// let q = try_quad(func, 0.0, 1.0).unwrap();
///
/// assert!((q - 1.0/3.0).abs() < 1E-8);
/// ```
///
/// Errors if the function to evaluate returns an error
///
/// ```
/// use rv::misc::try_quad;
///
/// let func = |x: f64| {
///     if x > 0.5 {
///         Err(String::from("whoops"))
///     } else {
///         Ok(x.powi(2))
///     }
/// };
/// let q = try_quad(func, 0.0, 1.0);
///
/// assert!(q.is_err());
/// ```
pub fn try_quad<F, E>(func: F, a: f64, b: f64) -> Result<f64, E>
where
    F: Fn(f64) -> Result<f64, E>,
{
    try_quad_eps(func, a, b, None)
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

    #[test]
    fn quadp_of_x2() {
        let func = |x: f64| x.powi(2);
        let q = quadp(&func, 0.0, 1.0, QuadConfig::default());
        assert::close(q, 1.0 / 3.0, QUAD_EPS);
    }

    #[test]
    fn quadp_of_sin() {
        let func = |x: f64| x.sin();
        let q = quadp(&func, 0.0, 5.0 * PI, QuadConfig::default());
        assert::close(q, 2.0, QUAD_EPS);
    }

    #[test]
    fn try_quad_of_x2() {
        fn func(x: f64) -> Result<f64, u8> {
            Ok(x.powi(2))
        }
        let q = try_quad(func, 0.0, 1.0).unwrap();
        assert::close(q, 1.0 / 3.0, QUAD_EPS);
    }

    #[test]
    fn try_quad_of_sin() {
        fn func(x: f64) -> Result<f64, u8> {
            Ok(x.sin())
        }
        let q = try_quad(func, 0.0, 5.0 * PI).unwrap();
        assert::close(q, 2.0, QUAD_EPS);
    }
}
