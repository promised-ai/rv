use traits::SuffStat;

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

macro_rules! impl_int_traits {
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
