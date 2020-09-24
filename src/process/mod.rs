use std::marker::PhantomData;
use std::{fmt::Debug, iter::FromIterator};

use argmin::core::Error as ArgminError;
use argmin::prelude::ArgminOp;
use argmin::prelude::Executor;
use nalgebra::DVector;
use nalgebra::Scalar;
use serde::{Deserialize, Serialize};

use crate::traits::Rv;

pub mod gaussian;

pub trait Parameter:
    Clone
    + Serialize
    + for<'de> Deserialize<'de>
    + IntoIterator<Item = f64>
    + FromIterator<f64>
{
}

impl<P> Parameter for P where
    P: Clone
        + Serialize
        + for<'de> Deserialize<'de>
        + IntoIterator<Item = f64>
        + FromIterator<f64>
{
}

pub trait RandomProcess<X>
where
    X: Scalar + Debug,
{
    /// Type of the indexing set.
    type Index;

    /// Parameter type
    type Parameter: Parameter;

    /// Type of the sample function, aka trajectory of the process.
    type SampleFunction: Rv<DVector<X>>;

    /// Create a sample function at the indices given.
    fn sample_function(&self, indices: &[Self::Index]) -> Self::SampleFunction;

    /// Compute the log marginal likelihood
    fn ln_m(&self) -> f64;

    /// Compute the log marginal likelihood with an different set of parameters and compute the
    /// gradient.
    fn ln_m_with_parameters(
        &self,
        parameter: Self::Parameter,
    ) -> (f64, Self::Parameter);

    /// Get the parameters
    fn parameters(&self) -> Self::Parameter;

    /// Set with the given parameters
    fn set_parameters(&mut self, parameters: Self::Parameter);
}

/// Random Process which can be optimized to reach a maximum likelihood estimate.
pub trait RandomProcessMle<X>: RandomProcess<X>
where
    Self: Sized,
    X: Scalar + Debug,
{
    /// Error type from Optimization errors
    type Solver: argmin::core::Solver<RandomProcessMleOp<Self, X>>;

    /// Generator for optimizer
    fn generate_solver() -> Self::Solver;

    /// Run the optimization
    fn optimize(self, max_iters: u64) -> Result<Self, argmin::core::Error> {
        let params = self.parameters();
        let op = RandomProcessMleOp {
            process: self,
            phantom_x: PhantomData,
        };
        let solver = Self::generate_solver();

        let res = Executor::new(op, solver, params)
            .max_iters(max_iters)
            .run()?;
        let best = res.state().get_best_param();
        let mut new_model = res.operator.process;
        new_model.set_parameters(best);
        Ok(new_model)
    }
}

pub struct RandomProcessMleOp<P, X>
where
    P: RandomProcessMle<X>,
    X: Scalar + Debug,
{
    process: P,
    phantom_x: PhantomData<X>,
}

impl<'a, P, X> ArgminOp for RandomProcessMleOp<P, X>
where
    P: RandomProcessMle<X>,
    X: Scalar + Debug,
{
    type Param = <P as RandomProcess<X>>::Parameter;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        Ok(-self.process.ln_m_with_parameters(param.clone()).0)
    }

    fn gradient(
        &self,
        param: &Self::Param,
    ) -> Result<Self::Param, ArgminError> {
        Ok(Self::Param::from_iter(
            self.process
                .ln_m_with_parameters(param.clone())
                .1
                .into_iter()
                .map(|x| -x),
        ))
    }
}
