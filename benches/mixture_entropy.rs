use criterion::BatchSize;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use rv::dist::{Gaussian, Mixture, NormalGamma, SymmetricDirichlet};
use rv::traits::*;

fn bench_gmm_entropy(c: &mut Criterion) {
    let ng = NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0);
    let symdir = SymmetricDirichlet::new_unchecked(1.0, 4);

    c.bench_function("4-component GMM entropy", move |b| {
        b.iter_batched_ref(
            || {
                let mut rng = rand::thread_rng();
                let weights: Vec<f64> = symdir.draw(&mut rng);
                let components: Vec<Gaussian> = ng.sample(4, &mut rng);
                Mixture::new(weights, components).unwrap()
            },
            |mm| {
                let _q: f64 = mm.entropy();
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(mixture_entropy, bench_gmm_entropy);
criterion_main!(mixture_entropy);
