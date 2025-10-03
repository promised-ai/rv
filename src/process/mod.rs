use std::fmt::Debug;
use std::marker::PhantomData;

use argmin::argmin_error;
use argmin::core::{CostFunction, Executor, Gradient, IterState, Solver};
use nalgebra::DVector;
use nalgebra::Scalar;
use rand::Rng;

use crate::traits::Rv;

pub mod gaussian;

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
        parameter: &DVector<f64>,
    ) -> Result<(f64, DVector<f64>), Self::Error>;

    /// Get the parameters
    fn parameters(&self) -> DVector<f64>;

    /// Set with the given parameters
    fn set_parameters(
        self,
        parameters: &DVector<f64>,
    ) -> Result<Self, Self::Error>;
}

/// Random Process which can be optimized to reach a maximum likelihood estimate.
pub trait RandomProcessMle<X>: RandomProcess<X> + Clone
where
    Self: Sized,
    X: Scalar + Debug,
{
    /// Error type from Optimization errors
    type Solver: Solver<
            RandomProcessMleOp<Self, X>,
            IterState<DVector<f64>, DVector<f64>, (), (), (), f64>,
        >;

    /// Generator for optimizer
    fn generate_solver() -> Self::Solver;

    /// Create random parameters for this Process
    fn random_params<R: Rng>(&self, rng: &mut R) -> DVector<f64>;

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
        use std::iter::once;

        let mut best_params = self.parameters();
        let random_params =
            (0..random_reinits).map(|_| self.random_params(rng));

        let mut best_cost = f64::INFINITY;
        let mut successes = 0;
        let mut last_err = None;

        for params in once(best_params.clone()).chain(random_params) {
            let solver = Self::generate_solver();
            let op = RandomProcessMleOp::new(&self);
            let maybe_res = Executor::new(op, solver)
                .configure(|state| state.param(params).max_iters(max_iters))
                .run();

            match maybe_res {
                Ok(res) => {
                    successes += 1;
                    if best_cost > res.state.best_cost {
                        best_cost = res.state.best_cost;
                        best_params = res.state.best_param.expect(
                            "Should have a best params if this was successful",
                        );
                    }
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        if successes > 0 {
            self.set_parameters(&best_params)
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
    pub fn new(process: &P) -> Self {
        Self {
            process: process.clone(),
            phantom_x: PhantomData,
        }
    }
}

impl<P, X> CostFunction for RandomProcessMleOp<P, X>
where
    P: RandomProcessMle<X>,
    X: Scalar + Debug,
{
    type Param = DVector<f64>;

    type Output = f64;

    fn cost(
        &self,
        param: &DVector<f64>,
    ) -> Result<Self::Output, argmin::core::Error> {
        self.process.ln_m_with_params(param)
            .map(|x| -x.0)
            .map_err(|_| argmin_error!(InvalidParameter, format!("Could not compute ln_m_with_parameters where params = {:?}", param)))
    }
}

impl<P, X> Gradient for RandomProcessMleOp<P, X>
where
    P: RandomProcessMle<X>,
    X: Scalar + Debug,
{
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;

    fn gradient(
        &self,
        param: &DVector<f64>,
    ) -> Result<Self::Gradient, argmin::core::Error> {
        self.process
            .ln_m_with_params(param)
            .map(|x| -x.1)
            .map_err(|_| argmin_error!(InvalidParameter, format!("Could not compute ln_m_with_parameters where params = {:?}", param)))
    }
}
