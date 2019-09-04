use rand::rngs::StdRng;
use rand::SeedableRng;

use nalgebra::{DMatrix, DVector};

use rv::process::gaussian::kernel::*;
use rv::process::gaussian::*;


use env_logger;

pub fn main() {
    let mut rng = StdRng::seed_from_u64(0x1234);

    env_logger::builder().init();

    // Data
    let xs: DMatrix<f64> = DMatrix::from_column_slice(
        20,
        1,
        &[
            2.7440675196366238,
            3.5759468318620975,
            3.0138168803582195,
            2.724415914984484,
            2.1182739966945237,
            3.2294705653332807,
            2.1879360563134624,
            4.4588650039103985,
            4.818313802505147,
            1.9172075941288886,
            3.958625190413323,
            2.644474598764522,
            2.8402228054696614,
            4.627983191463305,
            0.3551802909894347,
            0.43564649850770354,
            0.1010919872016286,
            4.16309922773969,
            3.8907837547492523,
            4.3500607412340955,
        ],
    );

    let ys: DVector<f64> = DVector::from_column_slice(&[
        1.2117010670412551,
        -0.5847655768591932,
        0.34353803812764505,
        0.047685140039701424,
        -1.240707194118254,
        0.19651407160709586,
        0.570695199230213,
        -0.008861571318336647,
        1.6098438845869207,
        -0.9806232777843131,
        -0.2955801934702589,
        0.4048311403041103,
        1.1593270179965047,
        1.2187350944444972,
        0.5149991405535084,
        0.6717768981475669,
        -0.29456874332867367,
        -1.0288965583909206,
        -0.563753720241888,
        0.3107528300748319,
    ]);

    let kernel = // ConstantKernel::new(1.0) * RBFKernel::new(1.0)
        //ConstantKernel::new(1.0) * ExpSineSquaredKernel::new(1.0, 1.0)
        ConstantKernel::new(1.0) * RationalQuadratic::new(1.0, 1.0)
        + WhiteKernel::new(1.0);

    let gp_params = GaussianProcessParams::default()
        .with_noise_model(NoiseModel::Uniform(0.0));
    let gp = GaussianProcess::train(kernel, &xs, &ys, gp_params).unwrap();

    let (ln_m, grad_ln_m) = gp.ln_m_with_parameters(&gp.kernel().parameters()).unwrap();
    println!("Initial theta = {}", gp.kernel().parameters());
    println!("Initial ln_m = {}", ln_m);
    println!("Initial grad_ln_m = {}", grad_ln_m);

    println!("Optimizing...");
    let mut gp = gp.optimize(10, &mut rng).expect("Failed to optimize");

    println!("Optimized Kernel = {:#?}", gp.kernel());
    println!("Optimized ln_m = {:?}", gp.ln_m());
}
