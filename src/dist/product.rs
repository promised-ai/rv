//! Distribution over multiple data types
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::Datum;
use crate::dist::Distribution;
use crate::traits::*;

/// A product distribution is the distribution of independent distributions.
///
/// # Notes
///
/// The `ProductDistribution` is an abstraction around `Vec<Dist>`, which allows
/// implementation of `Rv<Vec<Datum>>`.
///
/// # Example
///
/// Create a mixture of product distributions of Categorical * Gaussian
///
/// ```
/// use rv::data::Datum;
/// use rv::dist::{
///     Categorical, Gaussian, Mixture, ProductDistribution, Distribution
/// };
/// use rv::traits::*;
///
/// // NOTE: Because the ProductDistribution is an abstraction around Vec<Dist>,
/// // the user must take care to get the order of distributions in each
/// // ProductDistribution correct.
/// let prod_1 = ProductDistribution::new(vec![
///     Distribution::Categorical(Categorical::new(&[0.1, 0.9]).unwrap()),
///     Distribution::Gaussian(Gaussian::new(3.0, 1.0).unwrap()),
/// ]);
///
/// let prod_2 = ProductDistribution::new(vec![
///     Distribution::Categorical(Categorical::new(&[0.9, 0.1]).unwrap()),
///     Distribution::Gaussian(Gaussian::new(-3.0, 1.0).unwrap()),
/// ]);
///
/// let prodmix = Mixture::new(vec![0.5, 0.5], vec![prod_1, prod_2]).unwrap();
///
/// let mut rng = rand::thread_rng();
///
/// let x: Datum = prodmix.draw(&mut rng);
/// let fx = prodmix.f(&x);
///
/// println!("draw: {:?}", x);
/// println!("f(x): {}", fx);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ProductDistribution {
    dists: Vec<Distribution>,
}

impl ProductDistribution {
    /// Create a new product distribution
    ///
    /// # Example
    ///
    /// ```
    /// use rv::data::Datum;
    /// use rv::dist::{
    ///     Categorical, Gaussian, Mixture, ProductDistribution, Distribution
    /// };
    /// use rv::traits::*;
    ///
    /// let prod = ProductDistribution::new(vec![
    ///     Distribution::Categorical(Categorical::new(&[0.1, 0.9]).unwrap()),
    ///     Distribution::Gaussian(Gaussian::new(3.0, 1.0).unwrap()),
    /// ]);
    ///
    /// let mut rng = rand::thread_rng();
    /// let x: Datum = prod.draw(&mut rng);
    /// ```
    pub fn new(dists: Vec<Distribution>) -> Self {
        Self { dists }
    }
}

impl HasDensity<Vec<Datum>> for ProductDistribution {
    fn ln_f(&self, x: &Vec<Datum>) -> f64 {
        self.dists
            .iter()
            .zip(x.iter())
            .map(|(dist, x_i)| dist.ln_f(x_i))
            .sum()
    }
}

impl Sampleable<Vec<Datum>> for ProductDistribution {
    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Vec<Datum> {
        self.dists.iter().map(|dist| dist.draw(rng)).collect()
    }
}

impl HasDensity<Datum> for ProductDistribution {
    fn ln_f(&self, x: &Datum) -> f64 {
        match x {
            Datum::Compound(ref xs) => self.ln_f(xs),
            _ => panic!("unsupported data type for product distribution"),
        }
    }
}

impl Sampleable<Datum> for ProductDistribution {
    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Datum {
        Datum::Compound(self.draw(rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Datum;
    use crate::dist::{Categorical, Distribution, Gaussian};

    fn catgauss_mix() -> ProductDistribution {
        ProductDistribution::new(vec![
            Distribution::Categorical(Categorical::new(&[0.1, 0.9]).unwrap()),
            Distribution::Gaussian(Gaussian::standard()),
        ])
    }

    #[test]
    fn ln_f() {
        let gauss = Gaussian::standard();
        let cat = Categorical::new(&[0.1, 0.9]).unwrap();

        let x_cat = 0_u8;
        let x_gauss = 1.2_f64;

        let x_prod =
            Datum::Compound(vec![Datum::U8(x_cat), Datum::F64(x_gauss)]);

        let ln_f = cat.ln_f(&x_cat) + gauss.ln_f(&x_gauss);
        let ln_f_prod = catgauss_mix().ln_f(&x_prod);

        assert::close(ln_f, ln_f_prod, 1e-12);
    }
}
