use crate::traits::Rv;

pub(crate) fn count_entropy_range<Fx: Rv<u32>>(
    fx: &Fx,
    mid: u32,
    lower: u32,
    upper: u32,
) -> f64 {
    let mut h = 0.0;

    debug_assert!(lower <= mid && mid <= upper);

    // left side
    let mut left = mid;
    loop {
        let ln_f = fx.ln_f(&left);
        let f = ln_f.exp();
        h -= f * ln_f;
        if left == 0 || (left <= lower && f < 1E-16) {
            break;
        }
        left -= 1;
    }

    // right side
    let mut right = mid + 1;
    loop {
        let ln_f = fx.ln_f(&right);
        let f = ln_f.exp();
        h -= f * ln_f;
        if right >= upper && f < 1E-16 {
            return h;
        }
        right += 1;
    }
}

/// Compute the entropy for count-type distributions by enumeration.
///
/// # Notes
/// - Assumes a unimodal distribution.
/// - Enumeration begins at `mid` and proceeds in both directions until f(x) is
///   less than a threshold (1e-16)
pub(crate) fn count_entropy<Fx: Rv<u32>>(fx: &Fx, mid: u32) -> f64 {
    count_entropy_range(fx, mid, mid, mid + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Poisson;

    #[test]
    fn count_entropy_value() {
        let pois = Poisson::new(1.2).unwrap();
        let h = count_entropy(&pois, 2);
        assert::close(h, 1.410_005_889_743_196_8, 1e-9);
    }
}
