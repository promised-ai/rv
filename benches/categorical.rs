use criterion::BatchSize;
use criterion::Criterion;
use criterion::ParameterizedBenchmark;
use criterion::{criterion_group, criterion_main};
use rv::dist::Categorical;
use rv::traits::Rv;

fn bench_cat_draw(c: &mut Criterion) {
    c.bench(
        "Categorical draw compare",
        ParameterizedBenchmark::new(
            "u8",
            |b, &k| {
                let cat = &Categorical::uniform(k);
                b.iter_batched_ref(
                    rand::thread_rng,
                    |mut rng| {
                        let _x: u8 = cat.draw(&mut rng);
                    },
                    BatchSize::SmallInput,
                )
            },
            vec![2, 3, 4, 10, 20, 50],
        )
        .with_function("usize", |b, &k| {
            let cat = &Categorical::uniform(k);
            b.iter_batched_ref(
                rand::thread_rng,
                |mut rng| {
                    let _x: usize = cat.draw(&mut rng);
                },
                BatchSize::SmallInput,
            )
        }),
    );
}

criterion_group!(cat_benches, bench_cat_draw,);
criterion_main!(cat_benches);
