use criterion::BatchSize;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use rv::dist::Categorical;
use rv::traits::*;

fn bench_cat_draw(c: &mut Criterion) {
    let mut group = c.benchmark_group("Categorical draw compare");
    for k in [2, 3, 4, 10, 20, 50] {
        let cat = &Categorical::uniform(k);
        group.bench_function(&format!("u8, k = {}", k), move |b| {
            b.iter_batched_ref(
                rand::thread_rng,
                |mut rng| {
                    let _x: u8 = cat.draw(&mut rng);
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function(&format!("usize, k = {}", k), move |b| {
            b.iter_batched_ref(
                rand::thread_rng,
                |mut rng| {
                    let _x: usize = cat.draw(&mut rng);
                },
                BatchSize::SmallInput,
            )
        });
    }
}

criterion_group!(cat_benches, bench_cat_draw,);
criterion_main!(cat_benches);
