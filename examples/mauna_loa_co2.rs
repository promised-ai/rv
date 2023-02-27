//! This is an example of how to use the GaussianProcess code to predict functional
//! form of the CO2 concentration at Mauna Loa from 1958-2008
//!
//! This example appears in the Gaussian Processes for Machine Learning book by Rasmussen and
//! Williams.
//!
//! # Model
//! This model uses the kernel function
//! ```math
//! K(x, x', \theta) = C_1 * RBF(x, x', \theta_1) + C_2 * RBF(x, x', \theta_2) * SE(x, x', \theta_3) + C_3 * RQ(x, x', \theta_4) + C_4 * RBF(x, x', \theta_5)
//! + GN(x, x', \theta_6)
//! ```
//! Where the following are kernels:
//! * `C_k` - A constant kernel derived from `\theta`
//! * `RBF` - A Radial Basis Function kernel
//! * `SE` - Squared Exponential kernel
//! * `RQ` - Rational quadratic kernel
//! * `GN` - Gaussian Noise kernel
//!
//! The following represent the design for each portion of the kernel
//! * `C_1 * RBF(x, x', \theta_1)` - Long term smooth rising trend
//! * `C_2 * RBF(x, x', \theta_2) * SE(x, x', \theta_3)` - Seasonal component
//! * `C_3 * RQ(x, x', \theta_4)` - Mid-term irregularity
//! * `C_4 * RBF(x, x', \theta_5) + GN(x, x', \theta_6)` - Noise
//!
//! # Reference
//! Keeling, R. F., Piper, S. C., Bollenbacher, A. F., and Walker, J. S. Atmospheric Carbon Dioxide
//! Record from Mauna Loa (1958-2008). United States: N. p., 2009. Web. doi:10.3334/CDIAC/atg.035.
#[cfg(feature = "process")]
pub fn main() -> std::io::Result<()> {
    use nalgebra::{DMatrix, DVector};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;
    use rv::process::gaussian::kernel::*;
    use rv::process::gaussian::{GaussianProcess, NoiseModel};
    use rv::process::{RandomProcess, RandomProcessMle};
    use std::fs::File;
    use std::io;
    use std::io::prelude::*;

    // Load the data from the data-file
    let file = File::open("./examples/simplified_mauna_loa_co2.txt")?;
    let reader = io::BufReader::new(file);
    let (mut xs, mut ys): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for line in reader.lines() {
        let line = line?;
        if !line.starts_with('#') {
            let cols: Vec<&str> = line.split(' ').take(2).collect();
            xs.push(cols[0].parse().unwrap());
            ys.push(cols[1].parse().unwrap());
        }
    }

    // Normalizing y_data
    let ys_mean: f64 = ys.iter().sum::<f64>() / (ys.len() as f64);
    let ys_var: f64 = ys.iter().map(|y| (y - ys_mean).powi(2)).sum::<f64>()
        / ((ys.len() - 1) as f64);
    ys = ys.into_iter().map(|y| (y - ys_mean) / ys_var).collect();

    println!("Loaded {} datapoints", xs.len());

    // Create GaussianProcess
    let kernel = ConstantKernel::new_unchecked(2.59_f64.powi(2))
        * RBFKernel::new_unchecked(51.0)
        + ConstantKernel::new_unchecked(0.257_f64.powi(2))
            * RBFKernel::new_unchecked(137.0)
            * ExpSineSquaredKernel::new_unchecked(2.15, 1.0)
        + ConstantKernel::new_unchecked(0.118_f64.powi(2))
            * RationalQuadratic::new_unchecked(2.32, 70.6)
        + ConstantKernel::new_unchecked(0.03_f64.powi(2))
            * RBFKernel::new_unchecked(1.01)
        + WhiteKernel::new_unchecked(0.001);

    println!("kernel = {:#?}", kernel);
    // These parameters define the kernel, keep in mind these are in log-scale
    println!("kernel theta = {:?}", kernel.parameters());

    // The {x,y}-train must be a DMatrix and DVector
    let xs: DMatrix<f64> = DMatrix::from_column_slice(xs.len(), 1, &xs);
    let ys: DVector<f64> = DVector::from_column_slice(&ys);

    println!("xs shape = {:?}", xs.shape());
    println!("ys shape = {:?}", ys.shape());

    // We now train a gaussian process on these training data
    let gp = GaussianProcess::train(kernel, xs, ys, NoiseModel::Uniform(1E-5))
        .expect("This should only fail if the covariance matrix from the kernel is not semi-positive definite");

    // The given parameters given the following log likelihood and the gradient at the location is
    let (ln_m, grad_ln_m) =
        gp.ln_m_with_params(&gp.kernel().parameters()).unwrap();
    println!("ln_m = {}\ngrad_ln_m = {:?}", ln_m, grad_ln_m);

    // Let's find better parametes
    println!("Optimizing...");
    let mut rng = Xoshiro256Plus::seed_from_u64(0xABCD);
    let gp = gp.optimize(10, 0, &mut rng).expect("Failed to optimize");

    println!("Optimum Kernel = {:?}", gp.kernel());
    println!("ln_m = {}", gp.ln_m());

    Ok(())
}

#[cfg(not(feature = "process"))]
pub fn main() {
    panic!("feature \"process\" required to run this example")
}
