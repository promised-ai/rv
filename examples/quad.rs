use rv::dist::{Gaussian, Mixture};
use rv::misc::quad_eps;
use rv::traits::Rv;
use std::time::Instant;

use peroxide::numerical::integral::{
    gauss_kronrod_quadrature, gauss_legendre_quadrature, Integral,
};

const EPS: f64 = 1e-10;
const A: f64 = -20.0;
const B: f64 = 20.0;

fn single_quad(mm: &Mixture<Gaussian>) -> f64 {
    quad_eps(|x| mm.f(&x), -7.0, 7.0, Some(EPS))
}

fn multi_quad(mm: &Mixture<Gaussian>) -> f64 {
    let interval = B - A;

    let q1 = {
        let frac = (-3.0 - A) / interval;
        quad_eps(|x| mm.f(&x), A, -3.0, Some(EPS * frac))
    };

    let q2 = {
        let frac = 6.0 / interval;
        quad_eps(|x| mm.f(&x), -3.0, 3.0, Some(EPS * frac))
    };

    let q3 = {
        let frac = (B - 3.0) / interval;
        quad_eps(|x| mm.f(&x), 3.0, B, Some(EPS * frac))
    };

    q1 + q2 + q3
}

fn gk_quad(mm: &Mixture<Gaussian>) -> f64 {
    let interval = B - A;

    let q1 = {
        let frac = (-3.0 - A) / interval;
        gauss_kronrod_quadrature(
            |x| mm.f(&x),
            (A, -3.0),
            Integral::G7K15(EPS * frac),
        )
    };

    let q2 = {
        let frac = 6.0 / interval;
        gauss_kronrod_quadrature(
            |x| mm.f(&x),
            (-3.0, 3.0),
            Integral::G7K15(EPS * frac),
        )
    };

    let q3 = {
        let frac = (B - 3.0) / interval;
        gauss_kronrod_quadrature(
            |x| mm.f(&x),
            (3.0, B),
            Integral::G7K15(EPS * frac),
        )
    };

    q1 + q2 + q3
}

fn gl_quad(mm: &Mixture<Gaussian>, n: usize) -> f64 {
    let q1 = gauss_legendre_quadrature(|x| mm.f(&x), n, (A, -3.0));
    let q2 = gauss_legendre_quadrature(|x| mm.f(&x), n, (-3.0, 3.0));
    let q3 = gauss_legendre_quadrature(|x| mm.f(&x), n, (3.0, B));

    q1 + q2 + q3
}

fn gle_quad(mm: &Mixture<Gaussian>, n: usize) -> f64 {
    let interval = B - A;
    let q1 = {
        let frac = (-3.0 - A) / interval;
        gauss_kronrod_quadrature(
            |x| mm.f(&x),
            (A, -3.0),
            Integral::G7K15(EPS * frac),
        )
    };
    let q2 = gauss_legendre_quadrature(|x| mm.f(&x), n, (-3.0, 3.0));
    let q3 = {
        let frac = (B - 3.0) / interval;
        gauss_kronrod_quadrature(
            |x| mm.f(&x),
            (3.0, B),
            Integral::G7K15(EPS * frac),
        )
    };

    q1 + q2 + q3
}

fn main() {
    let mm = Mixture::new_unchecked(
        vec![0.5, 0.5],
        vec![
            Gaussian::new_unchecked(-3.0, 1.0),
            Gaussian::new_unchecked(3.0, 1.0),
        ],
    );

    let q = 1.0;

    let t_s = Instant::now();
    let q_s = single_quad(&mm);
    println!(
        "Adaptive     {} - {}ns, err: {:+e}",
        q_s,
        t_s.elapsed().as_nanos(),
        (q - q_s).abs()
    );

    let t_m = Instant::now();
    let q_m = multi_quad(&mm);
    println!(
        "Multi        {} - {}ns, err: {:+e}",
        q_m,
        t_m.elapsed().as_nanos(),
        (q - q_m).abs()
    );

    let t_g = Instant::now();
    let q_g = gk_quad(&mm);
    println!(
        "GK Multi     {} - {}ns, err: {:+e}",
        q_g,
        t_g.elapsed().as_nanos(),
        (q - q_g).abs()
    );

    let t_l = Instant::now();
    let q_l = gl_quad(&mm, 30);
    println!(
        "GL multi     {} - {}ns, err: {:+e}",
        q_l,
        t_l.elapsed().as_nanos(),
        (q - q_l).abs()
    );

    let t_e = Instant::now();
    let q_e = gle_quad(&mm, 30);
    println!(
        "Hybrid Multi {} - {}ns, err: {:+e}",
        q_e,
        t_e.elapsed().as_nanos(),
        (q - q_e).abs()
    );
}
