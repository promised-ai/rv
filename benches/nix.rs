use std::hint::black_box;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use rv::data::GaussianSuffStat;
use rv::dist::Gaussian;
use rv::dist::NormalInvChiSquared;
use rv::traits::{ConjugatePrior, DataOrSuffStat, Sampleable, SuffStat};

fn bench_nix_postpred(c: &mut Criterion) {
    let mut group = c.benchmark_group("NIX ln pp(x)");
    let nix = NormalInvChiSquared::new_unchecked(0.1, 1.2, 2.3, 3.4);
    let mut rng = rand::rng();
    let g = Gaussian::standard();

    group.bench_function("No cache".to_string(), |b| {
        b.iter_batched(
            || {
                let stat = {
                    let mut stat = GaussianSuffStat::new();
                    g.sample_stream(&mut rng).take(10).for_each(|x: f64| {
                        stat.observe(&x);
                    });
                    stat
                };
                let y: f64 = g.draw(&mut rng);
                (y, stat)
            },
            |(y, stat)| {
                black_box(nix.ln_pp(&y, &DataOrSuffStat::SuffStat(&stat)))
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("With cache".to_string(), |b| {
        b.iter_batched(
            || {
                let stat = {
                    let mut stat = GaussianSuffStat::new();
                    g.sample_stream(&mut rng).take(10).for_each(|x: f64| {
                        stat.observe(&x);
                    });
                    stat
                };
                let y: f64 = g.draw(&mut rng);
                let stat: DataOrSuffStat<f64, _> =
                    DataOrSuffStat::SuffStat(&stat);
                let cache = nix.ln_pp_cache(&stat);
                (y, cache)
            },
            |(y, cache)| black_box(nix.ln_pp_with_cache(&cache, &y)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_gauss_stat(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gaussian Suffstat");

    let mut rng = rand::rng();
    let g = Gaussian::standard();

    group.bench_function("Forget".to_string(), |b| {
        b.iter_batched(
            || {
                let mut stat = GaussianSuffStat::new();
                for _ in 0..3 {
                    let x: f64 = g.draw(&mut rng);
                    stat.observe(&x);
                }
                let x: f64 = g.draw(&mut rng);
                stat.observe(&x);
                (x, stat)
            },
            |(x, mut stat)| {
                stat.forget(&x);
                black_box(());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Observe".to_string(), |b| {
        b.iter_batched(
            || {
                let mut stat = GaussianSuffStat::new();
                let x: f64 = g.draw(&mut rng);
                stat.observe(&x);
                let x: f64 = g.draw(&mut rng);
                (x, stat)
            },
            |(x, mut stat)| {
                stat.observe(&x);
                black_box(());
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(nix_benches, bench_nix_postpred, bench_gauss_stat);
criterion_main!(nix_benches);
