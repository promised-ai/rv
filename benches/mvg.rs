extern crate rand;
extern crate rv;
extern crate test;

use self::rv::dist::MvGaussian;
use self::rv::traits::{ContinuousDistr, Rv};
use self::test::Bencher;

#[bench]
fn bench_draw(b: &mut test::Bencher) {
    let mut rng = rand::thread_rng();
    let mvg = MvGaussian::standard(10).unwrap();
    b.iter(|| mvg.draw(&mut rng));
}

#[bench]
fn bench_sample_100(b: &mut test::Bencher) {
    let mut rng = rand::thread_rng();
    let mvg = MvGaussian::standard(10).unwrap();
    b.iter(|| mvg.sample(100, &mut rng));
}

#[bench]
fn bench_ln_f(b: &mut test::Bencher) {
    let mvg = MvGaussian::standard(10).unwrap();
    let x = DVector::<f64>::zeros(10);
    b.iter(|| mvg.ln_pdf(&x));
}
