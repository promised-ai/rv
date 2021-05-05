use criterion::black_box;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use nalgebra::DMatrix;
use rv::dist::InvWishart;
use rv::traits::Rv;

fn bench_wishart_draw(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "InvWisart, draw 1",
        |b, &&dims| {
            let iw = InvWishart::identity(dims);
            b.iter_batched_ref(
                rand::thread_rng,
                |mut rng| {
                    black_box(iw.draw(&mut rng));
                },
                BatchSize::SmallInput,
            )
        },
        &[2, 3, 5, 10],
    );
}

fn bench_wishart_ln_f(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "InvWisart, ln f(x)",
        |b, &&dims| {
            let iw = &InvWishart::identity(dims);
            let x = &DMatrix::<f64>::identity(dims, dims);
            b.iter(|| black_box(iw.ln_f(&x)))
        },
        &[2, 3, 5, 10],
    );
}

criterion_group!(wishart_benches, bench_wishart_draw, bench_wishart_ln_f,);
criterion_main!(wishart_benches);
