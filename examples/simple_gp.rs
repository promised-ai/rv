use nalgebra::{DMatrix, DVector};
use rand::{rngs::SmallRng, SeedableRng};
#[cfg(feature = "process")]
use rv::process::{
    gaussian::{kernel::*, GaussianProcess, NoiseModel},
    RandomProcess, RandomProcessMle,
};

/// Example of a noiseless Gaussian process.
/// Setting the noise model to a uniform 0 noise may cause numerical instability issues.
#[cfg(feature = "process")]
pub fn noiseless() {
    println!("Starting noiseless");
    let mut small_rng = SmallRng::seed_from_u64(0xABCD);
    let xs: DMatrix<f64> =
        DMatrix::from_column_slice(6, 1, &[1., 3., 5., 6., 7., 8.]);
    let ys: DVector<f64> = xs.map(|x| x * x.sin()).column(0).into();

    // The kernel constructed here is a constant times the radial basis function kernel
    // Default values work here as they will be optimized.
    let kernel = ConstantKernel::default() * RBFKernel::default();

    // We create the gaussian process object, with the kernel, training data, and zero noise
    // Then, start the training and optimize for a max of 1000 iterations with 0 restarts.
    // Restarts can help avoid local minima.
    let gp = GaussianProcess::train(kernel, xs, ys, NoiseModel::Uniform(0.0))
        .expect("Data is valid so this should succeed")
        .optimize(1000, 0, &mut small_rng)
        .expect("Optimization should succeed");

    println!("Optimized Kernel = {:#?}", gp.kernel());
    println!("ln_m = {}", gp.ln_m());
}

/// Example of a noisy gaussian process where the uncertainty of each data point is known.
#[cfg(feature = "process")]
pub fn noisy() {
    println!("Starting noisy");
    let mut small_rng = SmallRng::seed_from_u64(0xABCD);
    let xs: DMatrix<f64> = DMatrix::from_column_slice(
        20,
        1,
        &[
            0.1, 0.61578947, 1.13157895, 1.64736842, 2.16315789, 2.67894737,
            3.19473684, 3.71052632, 4.22631579, 4.74210526, 5.25789474,
            5.77368421, 6.28947368, 6.80526316, 7.32105263, 7.83684211,
            8.35263158, 8.86842105, 9.38421053, 9.9,
        ],
    );
    let ys: DVector<f64> = xs.map(|x| x * x.sin()).column(0).into();

    // Errors in measurements
    let dy: DVector<f64> = DVector::from_column_slice(&[
        0.917022, 1.22032449, 0.50011437, 0.80233257, 0.64675589, 0.59233859,
        0.68626021, 0.84556073, 0.89676747, 1.03881673, 0.91919451, 1.1852195,
        0.70445225, 1.37811744, 0.52738759, 1.17046751, 0.9173048, 1.05868983,
        0.64038694, 0.69810149,
    ]);

    // Add the noise to our data
    let ys = &ys + &dy;

    // Same kernel as noiseless
    let kernel = ConstantKernel::default() * RBFKernel::default();
    // Create a new GP but add per-point noise from dy
    let gp = GaussianProcess::train(
        kernel,
        xs,
        ys,
        NoiseModel::PerPoint(dy.map(|x| x * x)),
    )
    .expect("Data is valid so this should succeed")
    .optimize(100, 10, &mut small_rng)
    .expect("Optimization should succeed");

    println!("Optimized Kernel = {:#?}", gp.kernel());
    println!("kernel parama = {:?}", gp.kernel().parameters());
    println!("ln_m = {}", gp.ln_m());
}

#[cfg(feature = "process")]
pub fn main() {
    noiseless();
    noisy();
}

#[cfg(not(feature = "process"))]
pub fn main() {
    panic!("feature \"process\" required to run this example");
}
