use criterion::black_box;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use nalgebra::DVector;
use rv::dist::MvGaussian;
use rv::traits::{ContinuousDistr, Rv};

fn bench_mvg_draw(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "MvGaussian, draw 1",
        |b, &&dims| {
            let mvg = MvGaussian::standard(dims).unwrap();
            b.iter_batched_ref(
                rand::thread_rng,
                |mut rng| black_box(mvg.draw(&mut rng)),
                BatchSize::SmallInput,
            )
        },
        &[2, 3, 5, 10],
    );
}

// There is some pre-computation that makes sampling more efficient than
// repeatedly calling `draw`
fn bench_mvg_sample(c: &mut Criterion) {
    let mvg = MvGaussian::standard(10).unwrap();
    c.bench_function_over_inputs(
        "10-D MvGaussian, sample",
        move |b, &&n| {
            b.iter_batched_ref(
                rand::thread_rng,
                |mut rng| black_box(mvg.sample(n, &mut rng)),
                BatchSize::SmallInput,
            )
        },
        &[1, 10, 50, 100],
    );
}

fn bench_mvg_ln_f(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "MvGaussian ln f(x)",
        |b, &&dims| {
            let mvg = &MvGaussian::standard(dims).unwrap();
            let x = DVector::<f64>::zeros(dims);
            b.iter(|| mvg.ln_pdf(&x))
        },
        &[2, 3, 5, 10],
    );
}

criterion_group!(
    mvg_benches,
    bench_mvg_draw,
    bench_mvg_sample,
    bench_mvg_ln_f
);
criterion_main!(mvg_benches);
