use criterion::black_box;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use rv::data::GaussianSuffStat;
use rv::dist::Gaussian;
use rv::dist::NormalInvChiSquared;
use rv::traits::*;

fn bench_nix_postpred(c: &mut Criterion) {
    let mut group = c.benchmark_group("NIX ln pp(x)");
    let nix = NormalInvChiSquared::new_unchecked(0.1, 1.2, 2.3, 3.4);
    let mut rng = rand::thread_rng();
    let g = Gaussian::standard();

    group.bench_function(format!("No cache"), |b| {
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

    group.bench_function(format!("With cache"), |b| {
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
                let cache = nix.ln_pp_cache(&DataOrSuffStat::SuffStat(&stat));
                (y, cache)
            },
            |(y, cache)| black_box(nix.ln_pp_with_cache(&cache, &y)),
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(nix_benches, bench_nix_postpred);
criterion_main!(nix_benches);
