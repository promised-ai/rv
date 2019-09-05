//! This is an example of how to use the GaussianProcess code to predict functional
//! form of the CO2 concentration at Mauna Loa from 1958-2008
//!
//! # Reference
//! Keeling, R. F., Piper, S. C., Bollenbacher, A. F., and Walker, J. S. Atmospheric Carbon Dioxide
//! Record from Mauna Loa (1958-2008). United States: N. p., 2009. Web. doi:10.3334/CDIAC/atg.035.

use std::fs::File;
use std::io;
use std::io::prelude::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

use nalgebra::{DMatrix, DVector};

use rv::process::gaussian::kernel::*;
use rv::process::gaussian::*;

use optim::bfgs::BFGSParams;
use optim::line_search::WolfeParams;

use env_logger;

pub fn main() -> io::Result<()> {
    let mut rng = StdRng::seed_from_u64(0x1234);

    env_logger::builder().init();

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

    println!("Loaded {} datapoints", xs.len());

    // Create GaussianProcess
    let kernel = ConstantKernel::new(34.4_f64.powi(2)) * RBFKernel::new(41.8);
      // + ConstantKernel::new(3.27_f64.powi(2)) * RBFKernel::new(180.0) * ExpSineSquaredKernel::new(1.44, 1.0)
      // + ConstantKernel::new(0.446_f64.powi(2)) * RationalQuadratic::new(0.957, 17.7)
      // + ConstantKernel::new(0.197_f64.powi(2)) * RBFKernel::new(0.139)
      // + WhiteKernel::new(0.0336);
    
    println!("kernel = {:#?}", kernel);
    println!("kernel theta = {}", kernel.parameters());

    let xs: DMatrix<f64> = DMatrix::from_column_slice(xs.len(), 1, &xs);
    let ys: DVector<f64> = DVector::from_column_slice(&ys);

    println!("xs shape = {:?}", xs.shape());
    println!("ys shape = {:?}", ys.shape());

    let gp_params = GaussianProcessParams::default()
        .with_noise_model(NoiseModel::Uniform(1E-5))
        .with_bfgs_params(
            BFGSParams::default().with_accuracy(1E-4).with_wofle_params(
                WolfeParams {
                    max_iter: 10,
                    ..WolfeParams::default()
                },
            ),
        );
    let gp = GaussianProcess::train(kernel, &xs, &ys, gp_params).unwrap();

    let (ln_m, grad_ln_m) =
        gp.ln_m_with_parameters(&gp.kernel().parameters()).unwrap();
    println!("ln_m = {}\ngrad_ln_m = {}", ln_m, grad_ln_m);

    println!("Optimizing...");
    let gp = gp.optimize(0, &mut rng).expect("Failed to optimize");

    println!("Optimum Kernel = {:?}", gp.kernel());

    Ok(())
}
