#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::Datum;
use crate::traits::Rv;

/// Represents any distribution
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum Distribution {
    Bernoulli(super::Bernoulli),
    Beta(super::Beta),
    BetaBinomial(super::BetaBinomial),
    Binomial(super::Binomial),
    Categorical(super::Categorical),
    Cauchy(super::Cauchy),
    ChiSquared(super::ChiSquared),
    Dirichlet(super::Dirichlet),
    SymmetricDirichlet(super::SymmetricDirichlet),
    Exponential(super::Exponential),
    Gamma(super::Gamma),
    Gaussian(super::Gaussian),
    Geometric(super::Geometric),
    Gev(super::Gev),
    InvChiSquared(super::InvChiSquared),
    InvGamma(super::InvGamma),
    InvGaussian(super::InvGaussian),
    KsTwoAsymptotic(super::KsTwoAsymptotic),
    Kumaraswamy(super::Kumaraswamy),
    Laplace(super::Laplace),
    LogNormal(super::LogNormal),
    #[cfg(feature = "arraydist")]
    MvGaussian(super::MvGaussian),
    NegBinomial(super::NegBinomial),
    Pareto(super::Pareto),
    Poisson(super::Poisson),
    Product(super::ProductDistribution),
    ScaledInvChiSquared(super::ScaledInvChiSquared),
    Skellam(super::Skellam),
    StudentsT(super::StudentsT),
    Uniform(super::Uniform),
    VonMises(super::VonMises),
    #[cfg(feature = "arraydist")]
    InvWishart(super::InvWishart),
}

impl Rv<Datum> for Distribution {
    fn f(&self, x: &Datum) -> f64 {
        match self {
            Distribution::Bernoulli(inner) => inner.f(x),
            Distribution::Beta(inner) => inner.f(x),
            Distribution::BetaBinomial(inner) => inner.f(x),
            Distribution::Binomial(inner) => inner.f(x),
            Distribution::Categorical(inner) => inner.f(x),
            Distribution::Cauchy(inner) => inner.f(x),
            Distribution::ChiSquared(inner) => inner.f(x),
            Distribution::Dirichlet(inner) => inner.f(x),
            Distribution::SymmetricDirichlet(inner) => inner.f(x),
            Distribution::Exponential(inner) => inner.f(x),
            Distribution::Gamma(inner) => inner.f(x),
            Distribution::Gaussian(inner) => inner.f(x),
            Distribution::Geometric(inner) => inner.f(x),
            Distribution::Gev(inner) => inner.f(x),
            Distribution::InvChiSquared(inner) => inner.f(x),
            Distribution::InvGamma(inner) => inner.f(x),
            Distribution::InvGaussian(inner) => inner.f(x),
            Distribution::KsTwoAsymptotic(inner) => inner.f(x),
            Distribution::Kumaraswamy(inner) => inner.f(x),
            Distribution::Laplace(inner) => inner.f(x),
            Distribution::LogNormal(inner) => inner.f(x),
            #[cfg(feature = "arraydist")]
            Distribution::MvGaussian(inner) => inner.f(x),
            Distribution::NegBinomial(inner) => inner.f(x),
            Distribution::Pareto(inner) => inner.f(x),
            Distribution::Poisson(inner) => inner.f(x),
            Distribution::Product(inner) => inner.f(x),
            Distribution::ScaledInvChiSquared(inner) => inner.f(x),
            Distribution::Skellam(inner) => inner.f(x),
            Distribution::StudentsT(inner) => inner.f(x),
            Distribution::Uniform(inner) => inner.f(x),
            Distribution::VonMises(inner) => inner.f(x),
            #[cfg(feature = "arraydist")]
            Distribution::InvWishart(inner) => inner.f(x),
        }
    }

    fn ln_f(&self, x: &Datum) -> f64 {
        match self {
            Distribution::Bernoulli(inner) => inner.ln_f(x),
            Distribution::Beta(inner) => inner.ln_f(x),
            Distribution::BetaBinomial(inner) => inner.ln_f(x),
            Distribution::Binomial(inner) => inner.ln_f(x),
            Distribution::Categorical(inner) => inner.ln_f(x),
            Distribution::Cauchy(inner) => inner.ln_f(x),
            Distribution::ChiSquared(inner) => inner.ln_f(x),
            Distribution::Dirichlet(inner) => inner.ln_f(x),
            Distribution::SymmetricDirichlet(inner) => inner.ln_f(x),
            Distribution::Exponential(inner) => inner.ln_f(x),
            Distribution::Gamma(inner) => inner.ln_f(x),
            Distribution::Gaussian(inner) => inner.ln_f(x),
            Distribution::Geometric(inner) => inner.ln_f(x),
            Distribution::Gev(inner) => inner.ln_f(x),
            Distribution::InvChiSquared(inner) => inner.ln_f(x),
            Distribution::InvGamma(inner) => inner.ln_f(x),
            Distribution::InvGaussian(inner) => inner.ln_f(x),
            Distribution::KsTwoAsymptotic(inner) => inner.ln_f(x),
            Distribution::Kumaraswamy(inner) => inner.ln_f(x),
            Distribution::Laplace(inner) => inner.ln_f(x),
            Distribution::LogNormal(inner) => inner.ln_f(x),
            #[cfg(feature = "arraydist")]
            Distribution::MvGaussian(inner) => inner.ln_f(x),
            Distribution::NegBinomial(inner) => inner.ln_f(x),
            Distribution::Pareto(inner) => inner.ln_f(x),
            Distribution::Poisson(inner) => inner.ln_f(x),
            Distribution::Product(inner) => inner.ln_f(x),
            Distribution::ScaledInvChiSquared(inner) => inner.ln_f(x),
            Distribution::Skellam(inner) => inner.ln_f(x),
            Distribution::StudentsT(inner) => inner.ln_f(x),
            Distribution::Uniform(inner) => inner.ln_f(x),
            Distribution::VonMises(inner) => inner.ln_f(x),
            #[cfg(feature = "arraydist")]
            Distribution::InvWishart(inner) => inner.ln_f(x),
        }
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Datum {
        match self {
            Distribution::Bernoulli(inner) => inner.draw(rng),
            Distribution::Beta(inner) => inner.draw(rng),
            Distribution::BetaBinomial(inner) => inner.draw(rng),
            Distribution::Binomial(inner) => inner.draw(rng),
            Distribution::Categorical(inner) => inner.draw(rng),
            Distribution::Cauchy(inner) => inner.draw(rng),
            Distribution::ChiSquared(inner) => inner.draw(rng),
            Distribution::Dirichlet(inner) => inner.draw(rng),
            Distribution::SymmetricDirichlet(inner) => inner.draw(rng),
            Distribution::Exponential(inner) => inner.draw(rng),
            Distribution::Gamma(inner) => inner.draw(rng),
            Distribution::Gaussian(inner) => inner.draw(rng),
            Distribution::Geometric(inner) => inner.draw(rng),
            Distribution::Gev(inner) => inner.draw(rng),
            Distribution::InvChiSquared(inner) => inner.draw(rng),
            Distribution::InvGamma(inner) => inner.draw(rng),
            Distribution::InvGaussian(inner) => inner.draw(rng),
            Distribution::KsTwoAsymptotic(inner) => inner.draw(rng),
            Distribution::Kumaraswamy(inner) => inner.draw(rng),
            Distribution::Laplace(inner) => inner.draw(rng),
            Distribution::LogNormal(inner) => inner.draw(rng),
            #[cfg(feature = "arraydist")]
            Distribution::MvGaussian(inner) => inner.draw(rng),
            Distribution::NegBinomial(inner) => inner.draw(rng),
            Distribution::Pareto(inner) => inner.draw(rng),
            Distribution::Poisson(inner) => inner.draw(rng),
            Distribution::Product(inner) => inner.draw(rng),
            Distribution::ScaledInvChiSquared(inner) => inner.draw(rng),
            Distribution::Skellam(inner) => inner.draw(rng),
            Distribution::StudentsT(inner) => inner.draw(rng),
            Distribution::Uniform(inner) => inner.draw(rng),
            Distribution::VonMises(inner) => inner.draw(rng),
            #[cfg(feature = "arraydist")]
            Distribution::InvWishart(inner) => inner.draw(rng),
        }
    }

    fn sample<R: rand::Rng>(&self, n: usize, rng: &mut R) -> Vec<Datum> {
        match self {
            Distribution::Bernoulli(inner) => inner.sample(n, rng),
            Distribution::Beta(inner) => inner.sample(n, rng),
            Distribution::BetaBinomial(inner) => inner.sample(n, rng),
            Distribution::Binomial(inner) => inner.sample(n, rng),
            Distribution::Categorical(inner) => inner.sample(n, rng),
            Distribution::Cauchy(inner) => inner.sample(n, rng),
            Distribution::ChiSquared(inner) => inner.sample(n, rng),
            Distribution::Dirichlet(inner) => inner.sample(n, rng),
            Distribution::SymmetricDirichlet(inner) => inner.sample(n, rng),
            Distribution::Exponential(inner) => inner.sample(n, rng),
            Distribution::Gamma(inner) => inner.sample(n, rng),
            Distribution::Gaussian(inner) => inner.sample(n, rng),
            Distribution::Geometric(inner) => inner.sample(n, rng),
            Distribution::Gev(inner) => inner.sample(n, rng),
            Distribution::InvChiSquared(inner) => inner.sample(n, rng),
            Distribution::InvGamma(inner) => inner.sample(n, rng),
            Distribution::InvGaussian(inner) => inner.sample(n, rng),
            Distribution::KsTwoAsymptotic(inner) => inner.sample(n, rng),
            Distribution::Kumaraswamy(inner) => inner.sample(n, rng),
            Distribution::Laplace(inner) => inner.sample(n, rng),
            Distribution::LogNormal(inner) => inner.sample(n, rng),
            #[cfg(feature = "arraydist")]
            Distribution::MvGaussian(inner) => inner.sample(n, rng),
            Distribution::NegBinomial(inner) => inner.sample(n, rng),
            Distribution::Pareto(inner) => inner.sample(n, rng),
            Distribution::Poisson(inner) => inner.sample(n, rng),
            Distribution::Product(inner) => inner.sample(n, rng),
            Distribution::ScaledInvChiSquared(inner) => inner.sample(n, rng),
            Distribution::Skellam(inner) => inner.sample(n, rng),
            Distribution::StudentsT(inner) => inner.sample(n, rng),
            Distribution::Uniform(inner) => inner.sample(n, rng),
            Distribution::VonMises(inner) => inner.sample(n, rng),
            #[cfg(feature = "arraydist")]
            Distribution::InvWishart(inner) => inner.sample(n, rng),
        }
    }

    fn sample_stream<'r, R: rand::Rng>(
        &'r self,
        rng: &'r mut R,
    ) -> Box<dyn Iterator<Item = Datum> + 'r> {
        match self {
            Distribution::Bernoulli(inner) => inner.sample_stream(rng),
            Distribution::Beta(inner) => inner.sample_stream(rng),
            Distribution::BetaBinomial(inner) => inner.sample_stream(rng),
            Distribution::Binomial(inner) => inner.sample_stream(rng),
            Distribution::Categorical(inner) => inner.sample_stream(rng),
            Distribution::Cauchy(inner) => inner.sample_stream(rng),
            Distribution::ChiSquared(inner) => inner.sample_stream(rng),
            Distribution::Dirichlet(inner) => inner.sample_stream(rng),
            Distribution::SymmetricDirichlet(inner) => inner.sample_stream(rng),
            Distribution::Exponential(inner) => inner.sample_stream(rng),
            Distribution::Gamma(inner) => inner.sample_stream(rng),
            Distribution::Gaussian(inner) => inner.sample_stream(rng),
            Distribution::Geometric(inner) => inner.sample_stream(rng),
            Distribution::Gev(inner) => inner.sample_stream(rng),
            Distribution::InvChiSquared(inner) => inner.sample_stream(rng),
            Distribution::InvGamma(inner) => inner.sample_stream(rng),
            Distribution::InvGaussian(inner) => inner.sample_stream(rng),
            Distribution::KsTwoAsymptotic(inner) => inner.sample_stream(rng),
            Distribution::Kumaraswamy(inner) => inner.sample_stream(rng),
            Distribution::Laplace(inner) => inner.sample_stream(rng),
            Distribution::LogNormal(inner) => inner.sample_stream(rng),
            #[cfg(feature = "arraydist")]
            Distribution::MvGaussian(inner) => inner.sample_stream(rng),
            Distribution::NegBinomial(inner) => inner.sample_stream(rng),
            Distribution::Pareto(inner) => inner.sample_stream(rng),
            Distribution::Poisson(inner) => inner.sample_stream(rng),
            Distribution::Product(inner) => inner.sample_stream(rng),
            Distribution::ScaledInvChiSquared(inner) => {
                inner.sample_stream(rng)
            }
            Distribution::Skellam(inner) => inner.sample_stream(rng),
            Distribution::StudentsT(inner) => inner.sample_stream(rng),
            Distribution::Uniform(inner) => inner.sample_stream(rng),
            Distribution::VonMises(inner) => inner.sample_stream(rng),
            #[cfg(feature = "arraydist")]
            Distribution::InvWishart(inner) => inner.sample_stream(rng),
        }
    }
}

impl Rv<Datum> for super::Mixture<Vec<Distribution>> {
    fn ln_f(&self, x: &Datum) -> f64 {
        if let Datum::Compound(xs) = x {
            assert_eq!(xs.len(), self.components()[0].len());
            let ln_fs: Vec<f64> = self
                .weights()
                .iter()
                .zip(self.components().iter())
                .map(|(&w, cpnts)| {
                    w.ln()
                        + xs.iter()
                            .zip(cpnts.iter())
                            .map(|(x, cpnt)| cpnt.ln_f(x))
                            .sum::<f64>()
                })
                .collect();
            crate::misc::logsumexp(&ln_fs)
        } else {
            panic!("Mixture of Vec<Dist> accepts Datum::Compound")
        }
    }

    fn sample<R: rand::Rng>(&self, n: usize, rng: &mut R) -> Vec<Datum> {
        let cpnt_ixs = crate::misc::pflip(self.weights(), n, rng);
        cpnt_ixs
            .iter()
            .map(|&ix| {
                let data = self.components()[ix]
                    .iter()
                    .map(|cpnt| cpnt.draw(rng))
                    .collect();
                Datum::Compound(data)
            })
            .collect()
    }

    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Datum {
        self.sample(1, rng).pop().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_basic_impls;

    test_basic_impls!(
        Distribution::Bernoulli(crate::dist::Bernoulli::uniform()),
        Datum::Bool(true),
        bernoulli
    );

    test_basic_impls!(
        Distribution::Beta(crate::dist::Beta::jeffreys()),
        Datum::F64(0.5),
        beta
    );

    test_basic_impls!(
        Distribution::BetaBinomial(
            crate::dist::BetaBinomial::new(10, 0.5, 1.2).unwrap()
        ),
        Datum::U32(3),
        beta_binom
    );

    test_basic_impls!(
        Distribution::Binomial(crate::dist::Binomial::new(10, 0.5).unwrap()),
        Datum::U32(3),
        binom
    );

    test_basic_impls!(
        Distribution::Categorical(crate::dist::Categorical::uniform(4)),
        Datum::U8(2),
        categorical
    );

    test_basic_impls!(
        Distribution::Cauchy(crate::dist::Cauchy::new(0.5, 1.0).unwrap()),
        Datum::F64(2.0),
        cauchy
    );

    test_basic_impls!(
        Distribution::ChiSquared(crate::dist::ChiSquared::new(0.5).unwrap()),
        Datum::F64(2.0),
        chi_squared
    );

    test_basic_impls!(
        Distribution::Dirichlet(
            crate::dist::Dirichlet::new(vec![5.0, 2.0, 0.5]).unwrap()
        ),
        Datum::Vec(vec![0.2, 0.1, 0.7]),
        dirichlet
    );

    test_basic_impls!(
        Distribution::SymmetricDirichlet(
            crate::dist::SymmetricDirichlet::new(0.5, 3).unwrap()
        ),
        Datum::Vec(vec![0.2, 0.1, 0.7]),
        symmetric_dirichlet
    );

    test_basic_impls!(
        Distribution::Exponential(crate::dist::Exponential::new(0.5).unwrap()),
        Datum::F64(2.0),
        exponential
    );

    test_basic_impls!(
        Distribution::Gamma(crate::dist::Gamma::new(0.5, 1.0).unwrap()),
        Datum::F64(2.0),
        gamma
    );

    test_basic_impls!(
        Distribution::Gaussian(crate::dist::Gaussian::standard()),
        Datum::F64(0.5),
        gaussian
    );

    test_basic_impls!(
        Distribution::Geometric(crate::dist::Geometric::new(0.5).unwrap()),
        Datum::U16(2),
        geometric
    );
}
