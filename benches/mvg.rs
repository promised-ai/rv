use criterion::black_box;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use nalgebra::DVector;
use rv::dist::MvGaussian;
use rv::traits::*;

fn bench_mvg_draw(c: &mut Criterion) {
    let mut group = c.benchmark_group("MvGaussian, draw 1");
    for dims in [2, 3, 5, 10] {
        group.bench_with_input(format!("{} dims", dims), &dims, |b, &dims| {
            let mvg = MvGaussian::standard(dims).unwrap();
            b.iter_batched_ref(
                rand::thread_rng,
                |mut rng| black_box::<DVector<f64>>(mvg.draw(&mut rng)),
                BatchSize::SmallInput,
            )
        });
    }
}

// There is some pre-computation that makes sampling more efficient than
// repeatedly calling `draw`
fn bench_mvg_sample(c: &mut Criterion) {
    let mvg = MvGaussian::standard(10).unwrap();
    let mut group = c.benchmark_group("10-D MvGaussian, sample");
    for n in [1, 10, 50, 100] {
        group.bench_with_input(n.to_string(), &n, |b, &n| {
            b.iter_batched_ref(
                rand::thread_rng,
                |mut rng| {
                    black_box::<Vec<DVector<f64>>>(mvg.sample(n, &mut rng))
                },
                BatchSize::SmallInput,
            );
        });
    }
}

fn bench_mvg_ln_f(c: &mut Criterion) {
    let mut group = c.benchmark_group("MvGaussian ln f(x)");
    for dims in [2, 3, 5, 10] {
        let mvg = &MvGaussian::standard(dims).unwrap();
        let x = DVector::<f64>::zeros(dims);
        group.bench_function(format!("{} dims", dims), |b| {
            b.iter(|| black_box(mvg.ln_pdf(&x)))
        });
    }
}

criterion_group!(
    mvg_benches,
    bench_mvg_draw,
    bench_mvg_sample,
    bench_mvg_ln_f
);
criterion_main!(mvg_benches);
