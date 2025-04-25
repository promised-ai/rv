use criterion::black_box;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use nalgebra::DMatrix;
use rv::dist::InvWishart;
use rv::traits::{HasDensity, Sampleable};

fn bench_wishart(c: &mut Criterion) {
    let mut group = c.benchmark_group("InvWishart");
    for dims in [2, 3, 5, 10] {
        group.bench_with_input(
            format!("draw - {dims} dims"),
            &dims,
            |b, &dims| {
                let iw = InvWishart::identity(dims);
                b.iter_batched_ref(
                    rand::thread_rng,
                    |mut rng| {
                        black_box::<DMatrix<f64>>(iw.draw(&mut rng));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_with_input(
            format!("ln f(x) - {dims} dims"),
            &dims,
            |b, &dims| {
                let iw = &InvWishart::identity(dims);
                let x = &DMatrix::<f64>::identity(dims, dims);
                b.iter(|| black_box(iw.ln_f(x)));
            },
        );
    }
}

criterion_group!(wishart_benches, bench_wishart);
criterion_main!(wishart_benches);
