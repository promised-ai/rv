//! This is an example of how to use the GaussianProcess code to predict functional
//! form of the CO2 concentration at Mauna Loa from 1958-2008
//!
//! # Reference
//! Keeling, R. F., Piper, S. C., Bollenbacher, A. F., and Walker, J. S. Atmospheric Carbon Dioxide
//! Record from Mauna Loa (1958-2008). United States: N. p., 2009. Web. doi:10.3334/CDIAC/atg.035.

use nalgebra::{DMatrix, DVector};
use rand::{rngs::SmallRng, SeedableRng};
use rv::process::gaussian::kernel::*;
use rv::process::gaussian::{GaussianProcess, NoiseModel};
use rv::process::{RandomProcess, RandomProcessMle};
use std::fs::File;
use std::io;
use std::io::prelude::*;

pub fn main() -> io::Result<()> {
    // Load the data from the data-file
    let file = File::open("./examples/simplified_mauna_loa_co2.txt")?;
    let reader = io::BufReader::new(file);
    let (mut xs, mut ys): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for line in reader.lines() {
        let line = line?;
        if !line.starts_with("#") {
            let cols: Vec<&str> = line.split(" ").take(2).collect();
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
    let kernel = ConstantKernel::new(2.44_f64.powi(2)) * RBFKernel::new(48.1)
        + ConstantKernel::new(0.535_f64.powi(2))
            * RBFKernel::new(4.02e+3)
            * ExpSineSquaredKernel::new(24.4, 1.03)
        + ConstantKernel::new(0.239_f64.powi(2))
            * RationalQuadratic::new(14.9, 4.58e+03)
        + ConstantKernel::new(0.0119_f64.powi(2)) * RBFKernel::new(1.76e+03)
        + WhiteKernel::new(0.0147);

    println!("kernel = {:#?}", kernel);
    println!("kernel theta = {:?}", kernel.parameters());

    let xs: DMatrix<f64> = DMatrix::from_column_slice(xs.len(), 1, &xs);
    let ys: DVector<f64> = DVector::from_column_slice(&ys);

    println!("xs shape = {:?}", xs.shape());
    println!("ys shape = {:?}", ys.shape());

    let gp = GaussianProcess::train(kernel, xs, ys, NoiseModel::Uniform(1E-5))
        .unwrap();

    let (ln_m, grad_ln_m) =
        gp.ln_m_with_parameters(gp.kernel().parameters()).unwrap();
    println!("ln_m = {}\ngrad_ln_m = {:?}", ln_m, grad_ln_m);

    println!("Optimizing...");
    let mut rng = SmallRng::seed_from_u64(0xABCD);
    let gp = gp.optimize(10, 0, &mut rng).expect("Failed to optimize");

    println!("Optimum Kernel = {:?}", gp.kernel());
    println!("ln_m = {}", gp.ln_m());

    Ok(())
}
