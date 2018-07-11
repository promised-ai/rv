extern crate rand;
extern crate special;

use self::rand::distributions::Uniform;
use self::rand::Rng;
use std::marker::PhantomData;
use traits::*;

/// Bernoulli distribution with success probability *p*
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Bernoulli<T> {
    p: f64,
    _phantom: PhantomData<T>,
}

impl<T> Bernoulli<T> {
    pub fn new(p: f64) -> Self {
        Bernoulli {
            p: p,
            _phantom: PhantomData,
        }
    }

    /// A Bernoulli distribution with a 50% chance of success
    pub fn uniform() -> Self {
        Bernoulli::new(0.5)
    }

    /// The complement of `p`, i.e. `(1 - p)`.
    #[inline]
    pub fn q(&self) -> f64 {
        1.0 - self.p
    }
}

impl<T> Default for Bernoulli<T> {
    fn default() -> Self {
        Bernoulli::uniform()
    }
}

macro_rules! impl_int_traits {
    ($kind:ty) => {
        impl Rv for Bernoulli<$kind> {
            type DatumType = $kind;

            fn f(&self, x: &$kind) -> f64 {
                if *x == 1 {
                    self.p
                } else {
                    1.0_f64 - self.p
                }
            }

            fn ln_f(&self, x: &$kind) -> f64 {
                self.f(x).ln()
            }

            fn ln_normalizer(&self) -> f64 {
                0.0
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let u = Uniform::new(0.0, 1.0);
                if rng.sample(u) < self.p {
                    1
                } else {
                    0
                }
            }

            fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<$kind> {
                let u = Uniform::new(0.0, 1.0);
                (0..n)
                    .map(|_| if rng.sample(u) < self.p { 1 } else { 0 })
                    .collect()
            }
        }

        impl Support for Bernoulli<$kind> {
            fn contains(&self, x: &$kind) -> bool {
                *x == 0 || *x == 1
            }
        }

        impl DiscreteDistr for Bernoulli<$kind> {
            fn pmf(&self, x: &$kind) -> f64 {
                self.f(x)
            }

            fn ln_pmf(&self, x: &$kind) -> f64 {
                self.ln_f(x)
            }
        }

        impl Cdf for Bernoulli<$kind> {
            fn cdf(&self, x: &$kind) -> f64 {
                if *x == 0 {
                    self.q()
                } else if *x > 0 {
                    1.0
                } else {
                    0.0
                }
            }
        }

        impl Mode for Bernoulli<$kind> {
            fn mode(&self) -> Option<$kind> {
                let q = self.q();
                if self.p < q {
                    Some(0)
                } else if self.p == q {
                    None
                } else {
                    Some(1)
                }
            }
        }
    };
}

macro_rules! impl_common_traits {
    ($kind:ty) => {
        impl Mean<f64> for Bernoulli<$kind> {
            fn mean(&self) -> Option<f64> {
                Some(self.p)
            }
        }

        impl Median<f64> for Bernoulli<$kind> {
            fn median(&self) -> Option<f64> {
                let q = self.q();
                if self.p < q {
                    Some(0.0)
                } else if self.p == q {
                    Some(0.5)
                } else {
                    Some(1.0)
                }
            }
        }
        impl Variance<f64> for Bernoulli<$kind> {
            fn variance(&self) -> Option<f64> {
                Some(self.p * (1.0 - self.p))
            }
        }

        impl Entropy for Bernoulli<$kind> {
            fn entropy(&self) -> f64 {
                let q = self.q();
                -q * q.ln() - self.p * self.p.ln()
            }
        }

        impl Skewness for Bernoulli<$kind> {
            fn skewness(&self) -> Option<f64> {
                Some((1.0 - 2.0 * self.p) / (self.p * self.q()).sqrt())
            }
        }

        impl Kurtosis for Bernoulli<$kind> {
            fn kurtosis(&self) -> Option<f64> {
                let q = self.q();
                Some((1.0 - 6.0 * self.p * q) / (self.p * q))
            }
        }
    };
}

impl Rv for Bernoulli<bool> {
    type DatumType = bool;

    fn f(&self, x: &bool) -> f64 {
        if *x {
            self.p
        } else {
            1.0_f64 - self.p
        }
    }

    fn ln_f(&self, x: &bool) -> f64 {
        self.f(x).ln()
    }

    fn ln_normalizer(&self) -> f64 {
        0.0
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> bool {
        let u = Uniform::new(0.0, 1.0);
        rng.sample(u) < self.p
    }

    fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<bool> {
        let u = Uniform::new(0.0, 1.0);
        (0..n).map(|_| rng.sample(u) < self.p).collect()
    }
}

impl Support for Bernoulli<bool> {
    fn contains(&self, _x: &bool) -> bool {
        true
    }
}

impl DiscreteDistr for Bernoulli<bool> {
    fn pmf(&self, x: &bool) -> f64 {
        self.f(x)
    }

    fn ln_pmf(&self, x: &bool) -> f64 {
        self.ln_f(x)
    }
}

impl Cdf for Bernoulli<bool> {
    fn cdf(&self, x: &bool) -> f64 {
        if *x {
            1.0
        } else {
            self.q()
        }
    }
}

impl Mode for Bernoulli<bool> {
    fn mode(&self) -> Option<bool> {
        let q = self.q();
        if self.p < q {
            Some(false)
        } else if self.p == q {
            None
        } else {
            Some(true)
        }
    }
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

impl_common_traits!(bool);
impl_common_traits!(u8);
impl_common_traits!(u16);
impl_common_traits!(u32);
impl_common_traits!(u64);
impl_common_traits!(usize);
impl_common_traits!(i8);
impl_common_traits!(i16);
impl_common_traits!(i32);
impl_common_traits!(i64);
impl_common_traits!(isize);

#[cfg(test)]
mod tests {
    extern crate assert;
    use super::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn new() {
        let b: Bernoulli<bool> = Bernoulli::new(0.1);
        assert::close(b.p, 0.1, TOL);
    }

    #[test]
    fn uniform_p_should_be_one_half() {
        let b: Bernoulli<bool> = Bernoulli::uniform();
        assert::close(b.p, 0.5, TOL);
    }

    #[test]
    fn q_should_be_the_compliment_of_p() {
        let b: Bernoulli<bool> = Bernoulli::new(0.1);
        assert::close(b.q(), 0.9, TOL);
    }

    #[test]
    fn pmf_of_1_should_be_p() {
        let b1: Bernoulli<bool> = Bernoulli::new(0.1);
        assert::close(b1.pmf(&true), 0.1, TOL);

        let b2: Bernoulli<bool> = Bernoulli::new(0.85);
        assert::close(b2.pmf(&true), 0.85, TOL);
    }

    #[test]
    fn ln_pmf_of_1_should_be_ln_p() {
        let b1: Bernoulli<bool> = Bernoulli::new(0.1);
        assert::close(b1.ln_pmf(&true), 0.1_f64.ln(), TOL);

        let b2: Bernoulli<bool> = Bernoulli::new(0.85);
        assert::close(b2.ln_pmf(&true), 0.85_f64.ln(), TOL);
    }

    #[test]
    fn pmf_of_0_should_be_q() {
        let b1: Bernoulli<bool> = Bernoulli::new(0.1);
        assert::close(b1.pmf(&false), 0.9, TOL);

        let b2: Bernoulli<bool> = Bernoulli::new(0.85);
        assert::close(b2.pmf(&false), 0.15, TOL);
    }

    #[test]
    fn ln_pmf_of_0_should_be_ln_q() {
        let b1: Bernoulli<bool> = Bernoulli::new(0.1);
        assert::close(b1.ln_pmf(&false), 0.9_f64.ln(), TOL);

        let b2: Bernoulli<bool> = Bernoulli::new(0.85);
        assert::close(b2.ln_pmf(&false), 0.15_f64.ln(), TOL);
    }

    #[test]
    fn sample_should_draw_the_correct_number_of_samples() {
        let mut rng = rand::thread_rng();
        let n = 103;
        let xs = Bernoulli::<bool>::uniform().sample(n, &mut rng);
        assert_eq!(xs.len(), n);
    }

    #[test]
    fn contains_both_true_and_false() {
        let b: Bernoulli<bool> = Bernoulli::uniform();
        assert!(b.contains(&true));
        assert!(b.contains(&false));
    }

    #[test]
    fn cmf_of_zero_is_q() {
        let b: Bernoulli<bool> = Bernoulli::new(0.1);
        assert::close(b.cdf(&false), 0.9, TOL);
    }

    #[test]
    fn cmf_of_one_is_one() {
        let b: Bernoulli<bool> = Bernoulli::new(0.1);
        assert::close(b.cdf(&true), 1.0, TOL);
    }

    #[test]
    fn mean_is_p() {
        assert::close(Bernoulli::<bool>::new(0.1).mean().unwrap(), 0.1, TOL);
        assert::close(Bernoulli::<bool>::new(0.7).mean().unwrap(), 0.7, TOL);
    }

    #[test]
    fn median_for_low_p_is_zero() {
        assert::close(Bernoulli::<bool>::new(0.1).median().unwrap(), 0.0, TOL);
        assert::close(Bernoulli::<bool>::new(0.499).median().unwrap(), 0.0, TOL);
    }

    #[test]
    fn median_for_high_p_is_one() {
        assert::close(Bernoulli::<bool>::new(0.9).median().unwrap(), 1.0, TOL);
        assert::close(Bernoulli::<bool>::new(0.5001).median().unwrap(), 1.0, TOL);
    }

    #[test]
    fn median_for_p_one_half_is_one_half() {
        assert::close(Bernoulli::<bool>::new(0.5).median().unwrap(), 0.5, TOL);
        assert::close(Bernoulli::<bool>::uniform().median().unwrap(), 0.5, TOL);
    }

    #[test]
    fn mode_for_high_p_is_true() {
        assert!(Bernoulli::<bool>::new(0.5001).mode().unwrap());
        assert!(Bernoulli::<bool>::new(0.8).mode().unwrap());
    }

    #[test]
    fn mode_for_low_p_is_true() {
        assert!(!Bernoulli::<bool>::new(0.4999).mode().unwrap());
        assert!(!Bernoulli::<bool>::new(0.2).mode().unwrap());
    }

    #[test]
    fn mode_for_even_p_is_none() {
        assert!(Bernoulli::<bool>::new(0.5).mode().is_none());
        assert!(Bernoulli::<bool>::uniform().mode().is_none());
    }

    #[test]
    fn variance_for_uniform() {
        assert::close(Bernoulli::<bool>::uniform().variance().unwrap(), 0.25, TOL);
    }

    #[test]
    fn variance() {
        assert::close(Bernoulli::<bool>::new(0.1).variance().unwrap(), 0.09, TOL);
        assert::close(Bernoulli::<bool>::new(0.9).variance().unwrap(), 0.09, TOL);
    }

    #[test]
    fn entropy() {
        let b1 = Bernoulli::<bool>::new(0.1);
        let b2 = Bernoulli::<bool>::new(0.9);
        assert::close(b1.entropy(), 0.3250829733914482, TOL);
        assert::close(b2.entropy(), 0.3250829733914482, TOL);
    }

    #[test]
    fn unifrom_entropy() {
        let b: Bernoulli<bool> = Bernoulli::uniform();
        assert::close(b.entropy(), 0.6931471805599453, TOL);
    }

    #[test]
    fn uniform_skewness_should_be_zero() {
        let b: Bernoulli<bool> = Bernoulli::uniform();
        assert::close(b.skewness().unwrap(), 0.0, TOL);
    }

    #[test]
    fn skewness() {
        let b: Bernoulli<bool> = Bernoulli::new(0.3);
        assert::close(b.skewness().unwrap(), 0.8728715609439696, TOL);
    }

    #[test]
    fn uniform_kurtosis() {
        let b: Bernoulli<bool> = Bernoulli::uniform();
        assert::close(b.kurtosis().unwrap(), -2.0, TOL);
    }
}
