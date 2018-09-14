extern crate rand;
extern crate rv;
extern crate test;

use self::rv::dist::InvWishart;
use self::rv::traits::Rv;
use self::test::Bencher;

#[bench]
fn bench_draw(b: &mut test::Bencher) {
    let mut rng = rand::thread_rng();
    let iw = InvWishart::identity(10);
    b.iter(|| iw.draw(&mut rng));
}

#[bench]
fn bench_ln_f(b: &mut test::Bencher) {
    let iw = InvWishart::identity(10);
    let x = DMatrix::<f64>::identity(10, 10);
    b.iter(|| iw.ln_f(&x));
}
