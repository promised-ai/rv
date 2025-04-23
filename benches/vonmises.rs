use criterion::black_box;
use criterion::AxisScale;
use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::PlotConfiguration;
use criterion::{criterion_group, criterion_main};
use rand::Rng;
use rv::consts::TWO_PI;
use rv::dist::VonMises;
use rv::misc::bessel::log_i0;
use rv::prelude::*;
use std::f64::consts::PI;

fn bench_vm_draw(c: &mut Criterion) {
    let mut group = c.benchmark_group("VonMises draw");

    // Configure the plot
    let plot_config =
        PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    // Test with different k values to see how precision affects sampling speed
    for k in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let vm = &VonMises::new(0.0, k).unwrap();
        group.bench_with_input(BenchmarkId::new("k", k), &k, move |b, _| {
            b.iter_batched_ref(
                rand::thread_rng,
                |rng| {
                    black_box::<f64>(vm.draw(rng));
                },
                BatchSize::SmallInput,
            )
        });
    }
}

fn bench_vm_ln_f(c: &mut Criterion) {
    let mut group = c.benchmark_group("VonMises ln_f");

    // Configure the plot
    let plot_config =
        PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    // Test with different k values to see how precision affects density calculation speed
    for k in [0.0, 0.5, 1.0, 2.0, 5.0] {
        let vm = &VonMises::new(0.0, k).unwrap();
        group.bench_with_input(BenchmarkId::new("k", k), &k, move |b, _| {
            b.iter_batched_ref(
                rand::thread_rng,
                |rng| {
                    let x: f64 = rng.gen_range(0.0..TWO_PI);
                    black_box(vm.ln_f(&x));
                },
                BatchSize::SmallInput,
            )
        });
    }
}

fn bench_log_i0(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_I0");

    // Configure the plot
    let plot_config =
        PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    // Test with different k values to see how precision affects density calculation speed
    for k in [0.0, 0.5, 1.0, 2.0, 5.0] {
        group.bench_with_input(BenchmarkId::new("k", k), &k, move |b, _| {
            b.iter_batched_ref(
                rand::thread_rng,
                |rng| {
                    let x: f64 = rng.gen_range(0.0..2.0 * PI);
                    black_box(log_i0(x));
                },
                BatchSize::SmallInput,
            )
        });
    }
}

criterion_group!(benches, bench_vm_draw, bench_vm_ln_f, bench_log_i0);
criterion_main!(benches);
