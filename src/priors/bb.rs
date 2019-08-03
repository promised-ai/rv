use rand::Rng;
use special::Beta as SBeta;

use crate::data::{BernoulliSuffStat, DataOrSuffStat};
use crate::dist::{Bernoulli, Beta};
use crate::traits::*;

impl Rv<Bernoulli> for Beta {
    fn ln_f(&self, x: &Bernoulli) -> f64 {
        self.ln_f(&x.p())
    }

    fn draw<R: Rng>(&self, mut rng: &mut R) -> Bernoulli {
        let p: f64 = self.draw(&mut rng);
        Bernoulli::new(p).expect("Failed to draw valid weight")
    }
}

impl Support<Bernoulli> for Beta {
    fn supports(&self, x: &Bernoulli) -> bool {
        0.0 < x.p() && x.p() < 1.0
    }
}

impl ContinuousDistr<Bernoulli> for Beta {}

impl<'a> ConjugatePrior<'a, bool, Bernoulli> for Beta {
    type Posterior = Beta;
    fn posterior<S>(&self, x: &S) -> Beta
    where
        S: Into<&'a BernoulliSuffStat>,
    {
        let stat: &BernoulliSuffStat = x.into();
        let n = stat.n();
        let k = stat.k();

        let a = self.alpha() + k as f64;
        let b = self.beta() + (n - k) as f64;

        Beta::new(a, b).expect("Invalid posterior parameters")
    }

    fn ln_m<S>(&self, x: &S) -> f64
    where
        S: Into<&'a BernoulliSuffStat>,
    {
        let post =
            <Beta as ConjugatePrior<bool, Bernoulli>>::posterior(self, x);

        post.alpha().ln_beta(post.beta()) - self.alpha().ln_beta(self.beta())
    }

    fn ln_pp<S>(&self, y: &bool, x: &S) -> f64
    where
        S: Into<&'a BernoulliSuffStat>,
    {
        //  P(y=1 | xs) happens to be the posterior mean
        let post =
            <Beta as ConjugatePrior<bool, Bernoulli>>::posterior(self, x);
        let p: f64 = post.mean().expect("Mean undefined");
        if *y {
            p.ln()
        } else {
            (1.0 - p).ln()
        }
    }
}

macro_rules! impl_int_traits {
    ($kind:ty) => {
        impl<'a> ConjugatePrior<'a, $kind, Bernoulli> for Beta {
            type Posterior = Self;
            fn posterior<S>(&self, x: &S) -> Self
            where
                S: Into<&'a BernoulliSuffStat>,
            {
                let stat: BernoulliSuffStat = x.into();
                let n = stat.n();
                let k = stat.k();

                let a = self.alpha() + k as f64;
                let b = self.beta() + (n - k) as f64;

                Beta::new(a, b).expect("Invalid posterior parameters")
            }

            fn ln_m<S>(&self, x: &S) -> f64
            where
                S: Into<&'a BernoulliSuffStat>,
            {
                let post =
                    <Beta as ConjugatePrior<$kind, Bernoulli>>::posterior(
                        self, x,
                    );
                post.alpha().ln_beta(post.beta())
                    - self.alpha().ln_beta(self.beta())
            }

            fn ln_pp<S>(&self, y: &$kind, x: &S) -> f64
            where
                S: Into<&'a BernoulliSuffStat>,
            {
                //  P(y=1 | xs) happens to be the posterior mean
                let post =
                    <Beta as ConjugatePrior<$kind, Bernoulli>>::posterior(
                        self, x,
                    );
                let p: f64 = post.mean().expect("Mean undefined");
                if *y == 1 {
                    p.ln()
                } else {
                    (1.0 - p).ln()
                }
            }
        }
    };
}

impl_int_traits!(u8);
impl_int_traits!(u16);
impl_int_traits!(u32);
impl_int_traits!(u64);
impl_int_traits!(usize);

impl_int_traits!(i8);
impl_int_traits!(i16);
impl_int_traits!(i32);
impl_int_traits!(i64);
impl_int_traits!(isize);

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn posterior_from_data_bool() {
        let data = vec![false, true, false, true, true];
        let xs = DataOrSuffStat::Data::<bool, Bernoulli>(&data);

        let posterior = Beta::new(1.0, 1.0).unwrap().posterior(&xs);

        assert::close(posterior.alpha(), 4.0, TOL);
        assert::close(posterior.beta(), 3.0, TOL);
    }

    #[test]
    fn posterior_from_data_u16() {
        let data: Vec<u16> = vec![0, 1, 0, 1, 1];
        let xs = DataOrSuffStat::Data::<u16, Bernoulli>(&data);

        let posterior = Beta::new(1.0, 1.0).unwrap().posterior(&xs);

        assert::close(posterior.alpha(), 4.0, TOL);
        assert::close(posterior.beta(), 3.0, TOL);
    }
}
