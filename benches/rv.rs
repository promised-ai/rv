use criterion::black_box;
use criterion::Benchmark;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use rv::data::Partition;
use rv::prelude::*;

// Takes a list of tuple-like inputs and builds benchmarks.
// Each entry contains:
// - An expression that constructs a valid distribution
// - The name of the distribution all lower case for naming the benchmark group
// - The string ref identifier of the benchmark group
// - The type of the rv used for `draw` and `ln_f`
macro_rules! benchrv {
    {
        $(
            ( $ctor:expr, $fn_name:ident, $bench_name:expr, $xtype:ty )
        );+
    } => {
        $(
            benchrv!($ctor, $fn_name, $bench_name, $xtype);
        )+

        criterion_group!(
            rv_benches,
            $(
                $fn_name,
            )+
        );

        criterion_main!(rv_benches);
    };
    ($ctor:expr, $fn_name:ident, $bench_name:expr, $xtype:ty) => {
        fn $fn_name(c: &mut Criterion) {
            c.bench(
                $bench_name,
                Benchmark::new("ln_f", |b| {
                    let mut rng = rand::thread_rng();
                    let fx = $ctor;
                    let x: $xtype = fx.draw(&mut rng);
                    b.iter(|| fx.ln_f(black_box(&x)))
                })
                .with_function("draw", |b| {
                    let mut rng = rand::thread_rng();
                    let fx = $ctor;
                    b.iter(|| {
                        let _x: $xtype = fx.draw(&mut rng);
                    })
                })
                .with_function("sample 5", |b| {
                    let mut rng = rand::thread_rng();
                    let fx = $ctor;
                    b.iter(|| {
                        let _xs: Vec<$xtype> = fx.sample(5, &mut rng);
                    })
                }),
            );
        }
    };
}

benchrv! {
    (Bernoulli::new(0.5).unwrap(), bernoulli, "rv bernoulli", bool);
    (Beta::jeffreys(), beta, "rv beta", f64);
    (BetaBinomial::new(20, 0.5, 0.3).unwrap(), betabinom, "rv betabinom", u32);
    (Binomial::new(20, 0.5).unwrap(), binomial, "rv binomial", u32);
    (Categorical::uniform(4), categorical, "rv categorical", usize);
    (Cauchy::new(0.0, 1.0).unwrap(), cauchy, "rv cauchy", f64);
    (ChiSquared::new(1.0).unwrap(), chi_squared, "rv chi_squared", f64);
    (Crp::new(1.0, 10).unwrap(), crp, "rv crp", Partition);
    (Dirichlet::jeffreys(4).unwrap(), dirichlet, "rv dirichlet", Vec<f64>);
    (
        DiscreteUniform::new(0u32, 10u32).unwrap(),
        discrete_uniform,
        "rv discrete_uniform",
        u32
    );
    (Exponential::new(1.0).unwrap(), exponential, "rv exponential", f64);
    (Gamma::default(), gamma, "rv gamma", f64);
    (Gaussian::default(), gaussian, "rv gaussian", f64);
    (Geometric::new(0.5).unwrap(), geometric, "rv geometric", u32);
    (Gev::new(0.5, 1.0, 2.0).unwrap(), gev, "rv gev", f64);
    (InvGamma::default(), inv_gamma, "rv inv_gamma", f64);
    (KsTwoAsymptotic::new(), ks2, "rv ks2", f64);
    (Kumaraswamy::default(), kumaraswamy, "rv kumaraswamy", f64);
    (Laplace::default(), laplace, "rv laplace", f64);
    (LogNormal::default(), lognormal, "rv lognormal", f64);
    (
        MvGaussian::standard(4).unwrap(),
        mv_gaussian,
        "rv mv_gaussian",
        DVector<f64>
    );
    (NegBinomial::new(5.0, 0.5).unwrap(), neg_binom, "rv neg_binom", u32);
    (
        {
            let mu = DVector::zeros(4);
            let scale = DMatrix::identity(4, 4);
            NormalInvWishart::new(mu, 5.1, 5, scale).unwrap()
        },
        niw,
        "rv niw",
        MvGaussian
    );
    (NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap(), ng, "rv ng", Gaussian);
    (Pareto::new(1.0, 1.0).unwrap(), pareto, "rv pareto", f64);
    (Skellam::new(2.0, 3.2).unwrap(), skellam, "rv skellam", i32);
    (StudentsT::default(), students, "rv students", f64);
    (Uniform::default(), uniform, "rv uniform", f64);
    (VonMises::new(0.5, 1.0).unwrap(), vonmises, "rv vonmises", f64);
    (InvWishart::identity(4), wishart, "rv wishart", DMatrix<f64>)
}
