#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::traits::*;
use rand::Rng;

/// An empirical distribution derived from samples.
///
/// __WARNING__: The `ln_f` and `f` methods are poor approximations.
/// They both are likely be have unbound errors.
///
/// ```rust
/// use rv::dist::{Gaussian, Empirical};
/// use rv::prelude::*;
/// use rv::misc::linspace;
/// use rand_xoshiro::Xoshiro256Plus;
/// use rand::SeedableRng;
///
/// let mut rng = Xoshiro256Plus::seed_from_u64(0xABCD);
/// let gen = Gaussian::standard();
///
/// let sample: Vec<f64> = gen.sample(1000, &mut rng);
/// let emp_dist = Empirical::new(sample);
///
/// let ln_f_err: Vec<f64> = linspace(emp_dist.range().0, emp_dist.range().1, 1000)
///     .iter()
///     .map(|x| {
///         gen.ln_f(x) - emp_dist.ln_f(x)
///     }).collect();
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde1", serde(rename_all = "snake_case"))]
pub struct Empirical {
    xs: Vec<f64>,
    range: (f64, f64),
}

#[derive(Clone, Copy, Debug)]
enum Pos {
    First,
    Last,
    Present(usize),
    Absent(usize),
}

pub struct EmpiricalParameters {
    pub xs: Vec<f64>,
}

impl Parameterized for Empirical {
    type Parameters = EmpiricalParameters;

    fn emit_params(&self) -> Self::Parameters {
        Self::Parameters {
            xs: self.xs.clone(),
        }
    }

    fn from_params(params: Self::Parameters) -> Self {
        Self::new(params.xs)
    }
}

impl Empirical {
    /// Create a new Empirical distribution with the given observed values
    pub fn new(mut xs: Vec<f64>) -> Self {
        xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let min = xs[0];
        let max = xs[xs.len() - 1];
        Empirical {
            xs,
            range: (min, max),
        }
    }

    fn pos(&self, x: f64) -> Pos {
        if x < self.range.0 {
            Pos::First
        } else if x >= self.range.1 {
            Pos::Last
        } else {
            self.xs
                .binary_search_by(|&probe| probe.partial_cmp(&x).unwrap())
                .map_or_else(Pos::Absent, Pos::Present)
        }
    }

    /// Return the CDF at X
    fn empcdf(&self, pos: Pos) -> f64 {
        match pos {
            Pos::First => 0.0,
            Pos::Last => 1.0,
            Pos::Present(ix) => ix as f64 / self.xs.len() as f64,
            Pos::Absent(ix) => ix as f64 / self.xs.len() as f64,
        }
    }

    /// Compute the CDF of a number of values
    pub fn empcdfs(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .map(|&value| {
                let pos = self.pos(value);
                self.empcdf(pos)
            })
            .collect()
    }

    /// A utility for computing a P-P plot.
    pub fn pp(&self, other: &Self) -> (Vec<f64>, Vec<f64>) {
        let mut xys = self.xs.clone();
        xys.append(&mut other.xs.clone());
        xys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        (self.empcdfs(&xys), other.empcdfs(&xys))
    }

    /// Area between CDF-CDF (1-1) line
    pub fn err(&self, other: &Self) -> f64 {
        let (fxs, fys) = self.pp(other);
        let diff: Vec<f64> = fxs
            .iter()
            .zip(fys.iter())
            .map(|(fx, fy)| (fx - fy).abs())
            .collect();

        let mut q = 0.0;
        for i in 1..fxs.len() {
            let step = fxs[i] - fxs[i - 1];
            let trap = diff[i] + diff[i - 1];
            q += step * trap
        }
        q / 2.0
    }

    /// Return the range of non-zero support for this distribution.
    pub fn range(&self) -> &(f64, f64) {
        &self.range
    }
}

impl HasDensity<f64> for Empirical {
    fn f(&self, x: &f64) -> f64 {
        eprintln!("WARNING: empirical.f is unstable. You probably don't want to use it.");
        match self.pos(*x) {
            Pos::First => 0.0,
            Pos::Last => 0.0,
            Pos::Present(0) => 0.0,
            Pos::Present(ix) => {
                let cdf_x = self.empcdf(Pos::Present(ix));
                let cdf_y = self.empcdf(Pos::Present(ix - 1));
                let y = self.xs[ix - 1];
                let h = x - y;
                (cdf_x - cdf_y) / h
            }
            Pos::Absent(ix) => {
                let cdf_x = self.empcdf(Pos::Absent(ix));
                let cdf_y = self.empcdf(Pos::Present(ix - 1));
                let y = self.xs[ix - 1];
                let h = x - y;
                (cdf_x - cdf_y) / h
            }
        }
    }

    fn ln_f(&self, x: &f64) -> f64 {
        self.f(x).ln()
    }
}

impl Sampleable<f64> for Empirical {
    fn draw<R: Rng>(&self, rng: &mut R) -> f64 {
        let n = self.xs.len();
        let ix: usize = rng.random_range(0..n);
        self.xs[ix]
    }
}

impl Cdf<f64> for Empirical {
    fn cdf(&self, x: &f64) -> f64 {
        let pos = self.pos(*x);
        self.empcdf(pos)
    }
}

impl Mean<f64> for Empirical {
    fn mean(&self) -> Option<f64> {
        let n = self.xs.len() as f64;
        Some(self.xs.iter().sum::<f64>() / n)
    }
}

impl Variance<f64> for Empirical {
    fn variance(&self) -> Option<f64> {
        let n = self.xs.len() as f64;
        self.mean().map(|m| {
            self.xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / n
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Gaussian;
    use crate::misc::linspace;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    #[ignore = "This failure is expected, ln_f should not be used."]
    fn gaussian_sample() {
        let mut rng = Xoshiro256Plus::seed_from_u64(0xABCD);
        let gen = Gaussian::standard();
        let sample: Vec<f64> = gen.sample(10000, &mut rng);
        let emp_dist = Empirical::new(sample);

        let (f_errs, cdf_errs): (Vec<f64>, Vec<f64>) =
            linspace(emp_dist.range().0, emp_dist.range().1, 1000)
                .into_iter()
                .map(|x| {
                    let ft = gen.f(&x);
                    let fe = emp_dist.f(&x);
                    let cdf_t = gen.cdf(&x);
                    let cdf_e = emp_dist.cdf(&x);
                    (fe - ft, cdf_e - cdf_t)
                })
                .unzip();

        let max_f_err = f_errs
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let max_cdf_err = cdf_errs
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        assert!(max_cdf_err < 1E-5);
        assert!(max_f_err < 1E-5);
    }

    #[test]
    fn draw_smoke() {
        let mut rng = rand::rng();
        // create a distribution with only a few bins so that draw hits all the
        // bins.
        let xs = vec![0.0, 1.0, 2.0];
        let emp_dist = Empirical::new(xs);

        for _ in 0..1_000 {
            let _x: f64 = emp_dist.draw(&mut rng);
        }
    }
}
