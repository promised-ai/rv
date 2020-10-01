use std::marker::PhantomData;
use std::{fmt::Debug, iter::FromIterator};

use argmin::core::Error as ArgminError;
use argmin::prelude::ArgminOp;
use argmin::prelude::Executor;
use nalgebra::DVector;
use nalgebra::Scalar;
use rand::Rng;

use crate::traits::Rv;

pub mod gaussian;

/// Parameters Much implement this trait
pub trait Param:
    Clone
    + Into<Vec<f64>>
    + From<Vec<f64>>
    + IntoIterator<Item = f64>
    + FromIterator<f64>
    + Debug
{
}

impl<P> Param for P where
    P: Clone
        + Into<Vec<f64>>
        + From<Vec<f64>>
        + IntoIterator<Item = f64>
        + FromIterator<f64>
        + Debug
{
}

/// A representation of a generic random process
pub trait RandomProcess<X>
where
    X: Scalar + Debug,
    Self: Sized,
{
    /// Error from
    type Error: std::error::Error + Send + Sync + 'static;
    /// Type of the indexing set.
    type Index;

    /// Parameter type
    type Param: Param;

    /// Type of the sample function, aka trajectory of the process.
    type SampleFunction: Rv<DVector<X>>;

    /// Create a sample function at the indices given.
    fn sample_function(&self, indices: &[Self::Index]) -> Self::SampleFunction;

    /// Compute the log marginal likelihood
    fn ln_m(&self) -> f64;

    /// Compute the log marginal likelihood with an different set of parameters and compute the
    /// gradient.
    fn ln_m_with_params(
        &self,
        parameter: Self::Param,
    ) -> Result<(f64, Self::Param), Self::Error>;

    /// Get the parameters
    fn parameters(&self) -> Self::Param;

    /// Set with the given parameters
    fn set_parameters(
        self,
        parameters: Self::Param,
    ) -> Result<Self, Self::Error>;
}

/// Random Process which can be optimized to reach a maximum likelihood estimate.
pub trait RandomProcessMle<X>: RandomProcess<X> + Clone
where
    Self: Sized,
    X: Scalar + Debug,
{
    /// Error type from Optimization errors
    type Solver: argmin::core::Solver<RandomProcessMleOp<Self, X>>;

    /// Generator for optimizer
    fn generate_solver() -> Self::Solver;

    /// Create random parameters for this Process
    fn random_params<R: Rng>(&self, rng: &mut R) -> Self::Param;

    /// Run the optimization
    ///
    /// # Arguments
    /// - `max_iters` - Maximum number of iterations per optimization run
    /// - `random_reinits` - Number of times to retry with random initialization
    /// - `rng` - Random number generator for random initialization
    fn optimize<R: Rng>(
        self,
        max_iters: u64,
        random_reinits: usize,
        rng: &mut R,
    ) -> Result<Self, argmin::core::Error> {
        let mut best_params = self.parameters();
        let random_params: Vec<Self::Param> = (0..random_reinits)
            .map(|_| self.random_params(rng))
            .collect();

        let mut best_cost = std::f64::INFINITY;
        let mut successes = 0;
        let mut last_err = None;

        for params in std::iter::once(best_params.clone())
            .chain(random_params.into_iter())
        {
            let solver = Self::generate_solver();
            // TODO: This is waseful, we don't need to copy
            let op = RandomProcessMleOp::new(self.clone());
            let maybe_res = Executor::new(op, solver, params.into())
                .max_iters(max_iters)
                .run();

            match maybe_res {
                Ok(res) => {
                    successes += 1;
                    if best_cost > res.state.best_cost {
                        best_cost = res.state.best_cost;
                        best_params = res.state.best_param.into();
                    }
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        if successes > 0 {
            self.set_parameters(best_params)
                .map_err(argmin::core::Error::from)
        } else {
            Err(last_err.unwrap())
        }
    }
}

/// Random Process Optimization target for Argmin
pub struct RandomProcessMleOp<P, X>
where
    P: RandomProcessMle<X>,
    X: Scalar + Debug,
{
    process: P,
    phantom_x: PhantomData<X>,
}

impl<P, X> RandomProcessMleOp<P, X>
where
    P: RandomProcessMle<X>,
    X: Scalar + Debug,
{
    /// Create a new Process wrapper for optimization
    pub fn new(process: P) -> Self {
        Self {
            process,
            phantom_x: PhantomData,
        }
    }
}

impl<P, X> ArgminOp for RandomProcessMleOp<P, X>
where
    P: RandomProcessMle<X>,
    X: Scalar + Debug,
{
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        self.process.ln_m_with_params(param.clone().into())
            .map(|x| -x.0)
            .map_err(|_| ArgminError::msg(format!("Could not compute ln_m_with_parameters where params = {:?}", param)))
    }

    fn gradient(
        &self,
        param: &Self::Param,
    ) -> Result<Self::Param, ArgminError> {
        self.process
            .ln_m_with_params(param.clone().into())
            .map(|x| Self::Param::from_iter(x.1.into_iter().map(|y| -y)))
            .map_err(|_| ArgminError::msg(format!("Could not compute ln_m_with_parameters where params = {:?}", param)))
    }
}
