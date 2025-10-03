use criterion::black_box;
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
            ( $fxtype: ty, $ctor:expr, $fn_name:ident, $bench_name:expr, $xtype:ty )
        );+
    } => {
        $(
            benchrv!($fxtype, $ctor, $fn_name, $bench_name, $xtype);
        )+

        criterion_group!(
            rv_benches,
            $(
                $fn_name,
            )+
        );

        criterion_main!(rv_benches);
    };
    ($fxtype: ty, $ctor:expr, $fn_name:ident, $bench_name:expr, $xtype:ty) => {
        fn $fn_name(c: &mut Criterion) {
            let mut group = c.benchmark_group($bench_name);
            group.bench_function("ln_f", |b| {
                let mut rng = rand::rng();
                let fx = $ctor;
                let x: $xtype = fx.draw(&mut rng);
                b.iter(|| fx.ln_f(black_box(&x)))
            });
            group.bench_function("draw", |b| {
                let mut rng = rand::rng();
                let fx = $ctor;
                b.iter(|| {
                    let _x: $xtype = fx.draw(&mut rng);
                });
            });
            group.bench_function("sample 5", |b| {
                let mut rng = rand::rng();
                let fx = $ctor;
                b.iter(|| {
                    let _xs: Vec<$xtype> = fx.sample(5, &mut rng);
                })
            });
            group.bench_function("stream, 5", |b| {
                let mut rng = rand::rng();
                let fx = $ctor;
                b.iter(|| {
                    let _count: usize =
                        <$fxtype as Sampleable<$xtype>>::sample_stream(&fx, &mut rng)
                        .take(5)
                        .count();
                })
            });
        }
    };
}

benchrv! {
    (Bernoulli, Bernoulli::new(0.5).unwrap(), bernoulli, "rv bernoulli", bool);
    (Beta, Beta::jeffreys(), beta, "rv beta", f64);
    (BetaBinomial, BetaBinomial::new(20, 0.5, 0.3).unwrap(), betabinom, "rv betabinom", u32);
    (Binomial, Binomial::new(20, 0.5).unwrap(), binomial, "rv binomial", u32);
    (Categorical, Categorical::uniform(4), categorical, "rv categorical", usize);
    (Cauchy, Cauchy::new(0.0, 1.0).unwrap(), cauchy, "rv cauchy", f64);
    (ChiSquared, ChiSquared::new(1.0).unwrap(), chi_squared, "rv chi_squared", f64);
    (Crp, Crp::new(1.0, 10).unwrap(), crp, "rv crp", Partition);
    (Dirichlet, Dirichlet::jeffreys(4).unwrap(), dirichlet, "rv dirichlet", Vec<f64>);
    (SymmetricDirichlet, SymmetricDirichlet::jeffreys(4).unwrap(), symdir, "rv symdir", Vec<f64>);
    (
        DiscreteUniform<u32>,
        DiscreteUniform::new(0u32, 10u32).unwrap(),
        discrete_uniform,
        "rv discrete_uniform",
        u32
    );
    (Exponential, Exponential::new(1.0).unwrap(), exponential, "rv exponential", f64);
    (Gamma, Gamma::default(), gamma, "rv gamma", f64);
    (Gaussian, Gaussian::default(), gaussian, "rv gaussian", f64);
    (Geometric, Geometric::new(0.5).unwrap(), geometric, "rv geometric", u32);
    (Gev, Gev::new(0.5, 1.0, 2.0).unwrap(), gev, "rv gev", f64);
    (InvGamma, InvGamma::default(), inv_gamma, "rv inv_gamma", f64);
    (KsTwoAsymptotic, KsTwoAsymptotic::new(), ks2, "rv ks2", f64);
    (Kumaraswamy, Kumaraswamy::default(), kumaraswamy, "rv kumaraswamy", f64);
    (Laplace, Laplace::default(), laplace, "rv laplace", f64);
    (LogNormal, LogNormal::default(), lognormal, "rv lognormal", f64);
    (
        MvGaussian,
        MvGaussian::standard(4).unwrap(),
        mv_gaussian,
        "rv mv_gaussian",
        DVector<f64>
    );
    (NegBinomial, NegBinomial::new(5.0, 0.5).unwrap(), neg_binom, "rv neg_binom", u32);
    (
        NormalInvWishart,
        {
            let mu = DVector::zeros(4);
            let scale = DMatrix::identity(4, 4);
            NormalInvWishart::new(mu, 5.1, 5, scale).unwrap()
        },
        niw,
        "rv niw",
        MvGaussian
    );
    (NormalGamma, NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap(), ng, "rv ng", Gaussian);
    (NormalInvGamma, NormalInvGamma::new(0.0, 1.0, 1.0, 1.0).unwrap(), nig, "rv nig", Gaussian);
    (NormalInvChiSquared, NormalInvChiSquared::new(0.0, 1.0, 1.0, 1.0).unwrap(), nix, "rv nix", Gaussian);
    (Pareto, Pareto::new(1.0, 1.0).unwrap(), pareto, "rv pareto", f64);
    (Skellam, Skellam::new(2.0, 3.2).unwrap(), skellam, "rv skellam", i32);
    (StudentsT, StudentsT::default(), students, "rv students", f64);
    (Uniform, Uniform::default(), uniform, "rv uniform", f64);
    (UnitPowerLaw, UnitPowerLaw::new(5.0).unwrap(), unit_powerlaw, "rv_unit_powerlaw", f64);
    (VonMises, VonMises::new(0.5, 1.0).unwrap(), vonmises, "rv vonmises", f64);
    (InvWishart, InvWishart::identity(4), wishart, "rv wishart", DMatrix<f64>)
}
