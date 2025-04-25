use criterion::BatchSize;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use rv::dist::Gev;
use rv::traits::Sampleable;

fn bench_gev_draw_0(c: &mut Criterion) {
    let gev = Gev::new(0.0, 1.0, 0.0).unwrap();
    c.bench_function("GEV(0, 1, 0), draw 1", move |b| {
        b.iter_batched_ref(
            rand::thread_rng,
            |mut rng| {
                let _x: f64 = gev.draw(&mut rng);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_gev_draw_one_half(c: &mut Criterion) {
    let gev = Gev::new(0.0, 1.0, 0.5).unwrap();
    c.bench_function("GEV(0, 1, 0.5), draw 1", move |b| {
        b.iter_batched_ref(
            rand::thread_rng,
            |mut rng| {
                let _x: f64 = gev.draw(&mut rng);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_gev_draw_negative_one_half(c: &mut Criterion) {
    let gev = Gev::new(0.0, 1.0, -0.5).unwrap();
    c.bench_function("GEV(0, 1, -0.5), draw 1", move |b| {
        b.iter_batched_ref(
            rand::thread_rng,
            |mut rng| {
                let _x: f64 = gev.draw(&mut rng);
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    gev_benches,
    bench_gev_draw_0,
    bench_gev_draw_one_half,
    bench_gev_draw_negative_one_half,
);
criterion_main!(gev_benches);
