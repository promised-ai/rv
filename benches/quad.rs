use criterion::black_box;
use criterion::Benchmark;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use rv::misc::quad_eps;
use rv::misc::try_quad_eps;

const EPS: f64 = 1e-8;
const LOWER: f64 = -3.0;
const UPPER: f64 = 3.0;

fn quad_fn(x: f64) -> f64 {
    (-0.5 * (x - 2.0)).exp()
}

fn try_quad_fn(x: f64) -> Result<f64, usize> {
    Ok((-0.5 * (x - 2.0)).exp())
}

fn bench_quad_vs_try_quad(c: &mut Criterion) {
    c.bench(
        "quad vs try_quad",
        Benchmark::new("quad", |b| {
            b.iter(|| black_box(quad_eps(quad_fn, LOWER, UPPER, Some(EPS))))
        })
        .with_function("try_quad", |b| {
            b.iter(|| {
                black_box(try_quad_eps(try_quad_fn, LOWER, UPPER, Some(EPS)))
            })
        }),
    );
}

criterion_group!(quad_benches, bench_quad_vs_try_quad);
criterion_main!(quad_benches);
