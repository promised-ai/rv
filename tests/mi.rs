use rand::Rng;
use rv::dist::{Gaussian, Mixture};
use rv::traits::*;

#[test]
fn bivariate_mixture_mi() {
    use rv::misc::LogSumExp;

    let n_samples = 100_000;
    let n_f = n_samples as f64;

    let mut rng = rand::rng();

    let mx = Mixture::uniform(vec![
        Gaussian::new_unchecked(-2.0, 1.0),
        Gaussian::new_unchecked(0.0, 1.0),
        Gaussian::new_unchecked(2.0, 1.0),
    ])
    .unwrap();

    let my = Mixture::uniform(vec![
        Gaussian::new_unchecked(-2.0, 0.5),
        Gaussian::new_unchecked(0.0, 0.5),
        Gaussian::new_unchecked(2.0, 1.0),
    ])
    .unwrap();

    let k = my.k();

    let hx = mx.entropy();
    let hy = my.entropy();

    let hx_est = -mx
        .sample_stream(&mut rng)
        .take(n_samples)
        .map(|x: f64| mx.ln_f(&x))
        .sum::<f64>()
        / n_f;

    let hy_est = -my
        .sample_stream(&mut rng)
        .take(n_samples)
        .map(|y: f64| my.ln_f(&y))
        .sum::<f64>()
        / n_f;

    let lnk = (k as f64).ln();

    let (mi_est, hxy_est) = {
        let (mi_sum, hxy_sum) =
            (0..n_samples).fold((0.0, 0.0), |(mi, hxy), _| {
                let cpnt_ix = rng.random_range(0..k);

                let x: f64 = mx.components()[cpnt_ix].draw(&mut rng);
                let y: f64 = my.components()[cpnt_ix].draw(&mut rng);

                let logpx = mx.ln_f(&x);
                let logpy = my.ln_f(&y);

                let logpxy = {
                    let ps = (0..k).map(|ix| {
                        let px = mx.components()[ix].ln_f(&x);
                        let py = my.components()[ix].ln_f(&y);
                        px + py - lnk
                    });

                    ps.logsumexp()
                };

                (mi + logpxy - logpx - logpy, hxy - logpxy)
            });
        (mi_sum / n_f, hxy_sum / n_f)
    };

    let mi_est_1 = hx_est + hy_est - hxy_est;
    let mi_est_2 = hx + hy - hxy_est;

    let _target = 0.5722305748503224; // estimate from 10 million samples

    println!("mi_est: {mi_est}");
    println!("hx_est: {hx_est}, hy_est: {hy_est}, hxy_est: {hxy_est}, mi_est: {mi_est_1}");
    println!("hx: {hx}, hy: {hy}, mi_est: {mi_est_2}");

    approx::assert_relative_eq!(mi_est, mi_est_1, epsilon = 0.05);
    approx::assert_relative_eq!(mi_est, mi_est_2, epsilon = 0.05);
}
