use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use rv::misc::log_product;

pub fn bench_log_product(c: &mut Criterion) {
    let min_positive = f64::MIN_POSITIVE;
    let max_x = min_positive.powf(1.0 / 1000.0);
    let input_size = 1000;
    let inputs: Vec<f64> = (0..=10)
        .map(|i| min_positive * (max_x / min_positive).powf(i as f64 / 10.0))
        .collect();

    let inputs: [f64; 11] = [
        1e-100, 1e-90, 1e-80, 1e-70, 1e-60, 1e-50, 1e-40, 1e-30, 1e-20, 1e-10,
        1e-0,
    ];

    let input_size = 1000;

    let mut group = c.benchmark_group("log_product_comparison");
    group.plot_config(PlotConfiguration::default());

    for &x in &inputs {
        let input = vec![x; input_size];
        let x = x.log10();

        group.bench_with_input(
            BenchmarkId::new("log_product", x),
            &input,
            |b, data| b.iter(|| log_product(data.iter().cloned())),
        );

        group.bench_with_input(
            BenchmarkId::new("sum_of_logs", x),
            &input,
            |b, data| b.iter(|| data.iter().map(|&x| x.ln()).sum::<f64>()),
        );

        group.bench_with_input(
            BenchmarkId::new("log_of_product", x),
            &input,
            |b, data| b.iter(|| data.iter().product::<f64>().ln()),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_log_product);
criterion_main!(benches);
