use crate::dist::{ChiSquared, Gaussian};
use crate::traits::Cdf;
use nalgebra::{DMatrix, DVector};

/// [Mardia's
/// test](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Multivariate_normality_tests) for multivariate normality.
#[allow(clippy::many_single_char_names)]
#[must_use]
pub fn mardia(xs: &[DVector<f64>]) -> (f64, f64) {
    let dims = xs[0].len();
    let n = xs.len() as f64;

    let xbar: DVector<f64> =
        xs.iter().fold(DVector::zeros(dims), |acc, x| acc + x) / n;

    let cov: DMatrix<f64> =
        xs.iter().fold(DMatrix::zeros(dims, dims), |acc, x| {
            let diff = x - &xbar;
            acc + &diff * &diff.transpose()
        }) / n;

    let inv_cov = cov.try_inverse().unwrap();

    let mut a: f64 = 0.0;
    for i in 0..xs.len() {
        for j in 0..xs.len() {
            let y = (&xs[i] - &xbar).transpose() * &inv_cov * (&xs[j] - &xbar);
            a += y[0].powi(3);
        }
    }
    a *= 1.0 / (6.0 * n);

    let bsum = xs.iter().fold(0.0, |acc, x| {
        let diff = x - &xbar;
        let y = &diff.transpose() * &inv_cov * &diff;
        y[0].mul_add(y[0], acc)
    });

    let k = dims as f64;
    let b = (n / (8.0 * k * (k + 2.0))).sqrt()
        * n.recip().mul_add(bsum, -k * (k + 2.0));

    let g = Gaussian::standard();
    let pb = g.sf(&b);

    let df = k * (k + 1.0) * (k + 2.0) / 6.0;
    let x2 = ChiSquared::new(df).unwrap();
    let pa = x2.sf(&a);

    (pa, pb)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dist::MvGaussian;
    use crate::traits::*;

    const MARDIA_PVAL: f64 = 0.05;
    const NTRIES: usize = 5;

    #[test]
    fn should_pass_for_normal_data() {
        let mut rng = rand::rng();
        let mvg = MvGaussian::standard(4).unwrap();

        let passed = (0..NTRIES).fold(false, |acc, _| {
            if acc {
                acc
            } else {
                let xys = mvg.sample(500, &mut rng);
                let (pa, pb) = mardia(&xys);
                pa > MARDIA_PVAL && pb > MARDIA_PVAL
            }
        });

        assert!(passed);
    }

    #[test]
    fn should_not_pass_for_normal_data() {
        let mut rng = rand::rng();
        let n = 500;
        let xs = ChiSquared::new(2.0).unwrap().sample(n, &mut rng);
        let ys = Gaussian::standard().sample(n, &mut rng);
        let xys: Vec<DVector<f64>> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                let xyv = vec![x, y];
                DVector::from_row_slice(&xyv)
            })
            .collect();
        let (pa, pb) = mardia(&xys);
        assert!(pa < MARDIA_PVAL && pb < MARDIA_PVAL);
    }
}
