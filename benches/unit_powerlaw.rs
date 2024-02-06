use std::f64;

use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use rv::traits::Rv;

fn draw_rv<R: rand::Rng>(mut rng: &mut R) -> f64 {
    let powlaw = rv::dist::UnitPowerLaw::new(5.0).unwrap();
    powlaw.draw(&mut rng)
}

fn draw_2u<R: rand::Rng>(rng: &mut R) -> f64 {
    let a = rng.gen::<f64>().powf(1.0 / 5.0);
    let b = rng.gen::<f64>().powf(1.0 / 2.0);
    a / (a + b)
}

fn draw_2u_recip<R: rand::Rng>(rng: &mut R) -> f64 {
    let a = rng.gen::<f64>().powf(5.0_f64.recip());
    let b = rng.gen::<f64>().powf(2.0_f64.recip());

    a / (a + b)
}

fn bench_powlaw_draw(c: &mut Criterion) {
    let mut group = c.benchmark_group("unit_powerlaw_draw");

    group.bench_function("draw_rv", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| draw_rv(&mut rng))
    });
    group.bench_function("draw_2_uniform", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| draw_2u(&mut rng))
    });
    group.bench_function("draw_2_uniform_recip", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| draw_2u_recip(&mut rng))
    });
}

fn bench_powlaw_beta_kumaraswamy_ln_pdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_powlaw_beta_kumaraswamy_ln_pdf");
    group.bench_function("powlaw", |b| {
        let powlaw = rv::dist::UnitPowerLaw::new(2.0).unwrap();
        b.iter(|| {
            let _x = powlaw.ln_f(&0.5_f64);
        });
    });
    group.bench_function("beta", |b| {
        let beta = rv::dist::Beta::new(2.0, 1.0).unwrap();
        b.iter(|| {
            let _x = beta.ln_f(&0.5_f64);
        });
    });
    group.bench_function("kumaraswamy", |b| {
        let kuma = rv::dist::Kumaraswamy::new(2.0, 1.0).unwrap();
        b.iter(|| {
            let _x = kuma.ln_f(&0.5_f64);
        });
    });
}

fn bench_powlaw_beta_kumaraswamy_draw(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_powlaw_beta_kumaraswamy_draw");
    group.bench_function("powlaw", |b| {
        let mut rng = rand::thread_rng();
        let beta = rv::dist::UnitPowerLaw::new(2.0).unwrap();
        b.iter(|| {
            let _x: f64 = beta.draw(&mut rng);
        });
    });
    group.bench_function("beta", |b| {
        let mut rng = rand::thread_rng();
        let beta = rv::dist::Beta::new(2.0, 1.0).unwrap();
        b.iter(|| {
            let _x: f64 = beta.draw(&mut rng);
        });
    });
    group.bench_function("kumaraswamy", |b| {
        let mut rng = rand::thread_rng();
        let kuma = rv::dist::Kumaraswamy::new(2.0, 1.0).unwrap();
        b.iter(|| {
            let _x: f64 = kuma.draw(&mut rng);
        });
    });
}

criterion_group!(
    unit_powerlaw_benches,
    bench_powlaw_draw,
    bench_powlaw_beta_kumaraswamy_ln_pdf,
    bench_powlaw_beta_kumaraswamy_draw,
);
criterion_main!(unit_powerlaw_benches);
