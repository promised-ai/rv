use dist::CategoricalDatum;
use traits::SuffStat;

// Bernoulli
// ---------
pub struct BernoulliSuffStat {
    pub n: usize,
    pub k: usize,
}

impl BernoulliSuffStat {
    pub fn new() -> Self {
        BernoulliSuffStat { n: 0, k: 0 }
    }
}

impl Default for BernoulliSuffStat {
    fn default() -> Self {
        BernoulliSuffStat::new()
    }
}

impl SuffStat<bool> for BernoulliSuffStat {
    fn observe(&mut self, x: &bool) {
        self.n += 1;
        if *x {
            self.k += 1
        }
    }

    fn forget(&mut self, x: &bool) {
        self.n -= 1;
        if *x {
            self.k -= 1
        }
    }
}

macro_rules! impl_bernoulli_suffstat {
    ($kind:ty) => {
        impl SuffStat<$kind> for BernoulliSuffStat {
            fn observe(&mut self, x: &$kind) {
                self.n += 1;
                if *x == 1 {
                    self.k += 1
                }
            }

            fn forget(&mut self, x: &$kind) {
                self.n -= 1;
                if *x == 1 {
                    self.k -= 1
                }
            }
        }
    };
}

impl_bernoulli_suffstat!(u8);
impl_bernoulli_suffstat!(u16);
impl_bernoulli_suffstat!(u32);
impl_bernoulli_suffstat!(u64);
impl_bernoulli_suffstat!(usize);
impl_bernoulli_suffstat!(i8);
impl_bernoulli_suffstat!(i16);
impl_bernoulli_suffstat!(i32);
impl_bernoulli_suffstat!(i64);
impl_bernoulli_suffstat!(isize);

// Categorical
// -----------
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CategoricalSuffStat {
    pub n: usize,
    pub counts: Vec<f64>,
}

impl CategoricalSuffStat {
    pub fn new(k: usize) -> Self {
        CategoricalSuffStat {
            n: 0,
            counts: vec![0.0; k],
        }
    }
}

impl<X: CategoricalDatum> SuffStat<X> for CategoricalSuffStat {
    fn observe(&mut self, x: &X) {
        let ix: usize = (*x).into();
        self.n += 1;
        self.counts[ix] += 1.0;
    }

    fn forget(&mut self, x: &X) {
        let ix: usize = (*x).into();
        self.n -= 1;
        self.counts[ix] -= 1.0;
    }
}

// Gaussian
// --------
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussianSuffStat {
    /// Number of observations
    pub n: usize,
    /// Sum of `x`
    pub sum_x: f64,
    /// Sum of `x^2`
    pub sum_x_sq: f64,
}

impl GaussianSuffStat {
    pub fn new() -> Self {
        GaussianSuffStat {
            n: 0,
            sum_x: 0.0,
            sum_x_sq: 0.0,
        }
    }
}

impl Default for GaussianSuffStat {
    fn default() -> Self {
        GaussianSuffStat::new()
    }
}

macro_rules! impl_gaussian_suffstat {
    ($kind:ty) => {
        // TODO: store in a more nuerically stable form
        impl SuffStat<$kind> for GaussianSuffStat {
            fn observe(&mut self, x: &$kind) {
                let xf = f64::from(*x);
                self.n += 1;
                self.sum_x += xf;
                self.sum_x_sq += xf.powi(2);
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = f64::from(*x);
                    self.n -= 1;
                    self.sum_x -= xf;
                    self.sum_x_sq -= xf.powi(2);
                } else {
                    self.n = 0;
                    self.sum_x = 0.0;
                    self.sum_x_sq = 0.0;
                }
            }
        }
    };
}

impl_gaussian_suffstat!(f32);
impl_gaussian_suffstat!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    mod bernoulli {
        use super::*;

        #[test]
        fn new_should_be_empty() {
            let stat = BernoulliSuffStat::new();
            assert_eq!(stat.n, 0);
            assert_eq!(stat.k, 0);
        }

        #[test]
        fn observe_1() {
            let mut stat = BernoulliSuffStat::new();
            stat.observe(&1_u8);
            assert_eq!(stat.n, 1);
            assert_eq!(stat.k, 1);
        }

        #[test]
        fn observe_true() {
            let mut stat = BernoulliSuffStat::new();
            stat.observe(&true);
            assert_eq!(stat.n, 1);
            assert_eq!(stat.k, 1);
        }

        #[test]
        fn observe_0() {
            let mut stat = BernoulliSuffStat::new();
            stat.observe(&0_i8);
            assert_eq!(stat.n, 1);
            assert_eq!(stat.k, 0);
        }

        #[test]
        fn observe_false() {
            let mut stat = BernoulliSuffStat::new();
            stat.observe(&false);
            assert_eq!(stat.n, 1);
            assert_eq!(stat.k, 0);
        }
    }

    mod categorical {
        extern crate assert;
        use super::*;

        #[test]
        fn new() {
            let sf = CategoricalSuffStat::new(4);
            assert_eq!(sf.counts.len(), 4);
            assert_eq!(sf.n, 0);
            assert!(sf.counts.iter().all(|&ct| ct.abs() < 1E-12))
        }
    }
}
