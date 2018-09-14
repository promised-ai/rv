extern crate rand;
extern crate rv;
extern crate test;

use self::rv::dist::Gev;
use self::rv::traits::{ContinuousDistr, Rv};
use self::test::Bencher;

#[bench]
fn bench_draw_0(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let gev = Gev::new(0.0, 1.0, 0.0).unwrap();
    b.iter(|| {
        let _sample: f64 = gev.draw(&mut rng);
    });
}

#[bench]
fn bench_draw_one_half(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let gev = Gev::new(0.0, 1.0, 0.5).unwrap();
    b.iter(|| {
        let _sample: f64 = gev.draw(&mut rng);
    });
}

#[bench]
fn bench_draw_negative_one_half(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let gev = Gev::new(0.0, 1.0, -0.5).unwrap();
    b.iter(|| {
        let _sample: f64 = gev.draw(&mut rng);
    });
}
