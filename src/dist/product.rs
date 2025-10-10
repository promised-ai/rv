//! This module implements common traits for Product Distributions Tuple Products, e.g. (Gaussian, Gaussian)

use crate::traits::{
    Cdf, ContinuousDistr, DiscreteDistr, Entropy, HasDensity, Mean, Median,
    Mode, Sampleable, Support,
};

#[cfg(feature = "experimental")]
use crate::{
    prelude::DataOrSuffStat,
    traits::{ConjugatePrior, HasSuffStat, SuffStat},
};

macro_rules! tuple_sampleable {
    ($($len:expr => ($($n:tt $t:ident $x:ident)*))+) => {
        $(
            impl<$($t,)*$($x,)*> Sampleable<($($x,)*)> for ($($t,)*)
            where
                $($t: Sampleable<$x>,)*
            {
                #[allow(unused_variables)]
                #[allow(clippy::unused_unit)]
                fn draw<R: rand::Rng>(&self, rng: &mut R) -> ($($x,)*) {
                    (
                        $(self.$n.draw(rng),)*
                    )
                }
            }
        )+
    };
}

macro_rules! tuple_has_density {
    ($($len:expr => ($($n:tt $t:ident $x:ident)*))+) => {
        $(
            impl<$($t,)*$($x,)*> HasDensity<($($x,)*)> for ($($t,)*)
            where
                $($t: HasDensity<$x>,)*
            {
                #[allow(unused_variables)]
                fn ln_f(&self, x: &($($x,)*)) -> f64 {
                    0.0 $(+ self.$n.ln_f(&x.$n))*
                }
            }
        )+
    };
}

macro_rules! tuple_support {
    ($($len:expr => ($($n:tt $t:ident $x:ident)*))+) => {
        $(
            impl<$($t,)*$($x,)*> Support<($($x,)*)> for ($($t,)*)
            where
                $($t: Support<$x>,)*
            {
                #[allow(unused_variables)]
                fn supports(&self, x: &($($x,)*)) -> bool {
                    true $(&& self.$n.supports(&x.$n))*
                }
            }
        )+
    };
}

macro_rules! tuple_discrete_continuous {
    ($($len:expr => ($($n:tt $t:ident $x:ident)*))+) => {
        $(
            impl<$($t,)*$($x,)*> ContinuousDistr<($($x,)*)> for ($($t,)*)
            where
                $($t: ContinuousDistr<$x>,)*
            {
            }

            impl<$($t,)*$($x,)*> DiscreteDistr<($($x,)*)> for ($($t,)*)
            where
                $($t: DiscreteDistr<$x>,)*
            {
            }

        )+
    };
}

macro_rules! tuple_cdf {
    ($($len:expr => ($($n:tt $t:ident $x:ident)*))+) => {
        $(
            impl<$($t,)*$($x,)*> Cdf<($($x,)*)> for ($($t,)*)
            where
                $($t: Cdf<$x>,)*
            {
                #[allow(unused_variables)]
                fn cdf(&self, x: &($($x,)*)) -> f64 {
                    1.0 $(* self.$n.cdf(&x.$n))*
                }
            }
        )+
    };
}

macro_rules! tuple_entropy {
    ($($len:expr => ($($n:tt $t:ident)*))+) => {
        $(
            impl<$($t,)*> Entropy for ($($t,)*)
            where
                $($t: Entropy,)*
            {
                fn entropy(&self) -> f64 {
                    0.0 $(+ self.$n.entropy())*
                }
            }
        )+
    };
}

macro_rules! tuple_mean_median_mode {
    ($($len:expr => ($($n:tt $t:ident $x:ident)*))+) => {
        $(
            impl<$($t,)*$($x,)*> Mean<($($x,)*)> for ($($t,)*)
            where
                $($t: Mean<$x>,)*
            {
                fn mean(&self) -> Option<($($x,)*)> {
                    Some(($(self.$n.mean()?,)*))
                }
            }

            impl<$($t,)*$($x,)*> Median<($($x,)*)> for ($($t,)*)
            where
                $($t: Median<$x>,)*
            {
                fn median(&self) -> Option<($($x,)*)> {
                    Some(($(self.$n.median()?,)*))
                }
            }

            impl<$($t,)*$($x,)*> Mode<($($x,)*)> for ($($t,)*)
            where
                $($t: Mode<$x>,)*
            {
                fn mode(&self) -> Option<($($x,)*)> {
                    Some(($(self.$n.mode()?,)*))
                }
            }


        )+
    };
}

macro_rules! tuple_suff_stat {
    ($($len:expr => ($($n:tt $t:ident $x:ident)*))+) => {
        $(

            #[cfg(feature = "experimental")]
            impl<$($t,)*$($x,)*> SuffStat<($($x,)*)> for ($($t,)*)
            where
                $($t: SuffStat<$x>,)*
            {
                fn n(&self) -> usize {
                    panic!("The number of observations for a product distribution is poorly defined, as not all suff stats will have the same number of observations");
                }

                #[allow(unused_variables)]
                fn observe(&mut self, x: &($($x,)*)) {
                    $(
                        self.$n.observe(&x.$n);
                    )*
                }

                #[allow(unused_variables)]
                fn forget(&mut self, x: &($($x,)*)) {
                    $(
                        self.$n.forget(&x.$n);
                    )*
                }

                #[allow(unused_variables)]
                fn merge(&mut self, other: Self) {
                    $(
                        self.$n.merge(other.$n);
                    )*
                }
            }
        )+
    };
}

macro_rules! tuple_has_suffstat {
    ($($len:expr => ($($n:tt $t:ident $x:ident)*))+) => {
        $(
            #[cfg(feature = "experimental")]
            impl<$($t,)*$($x,)*> HasSuffStat<($($x,)*)> for ($($t,)*)
            where
                $($t: HasSuffStat<$x>,)*
            {
                type Stat = ($($t::Stat,)*);

                #[allow(clippy::unused_unit)]
                fn empty_suffstat(&self) -> Self::Stat {
                    ($(self.$n.empty_suffstat(),)*)
                }

                #[allow(unused_variables)]
                fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
                    0.0 $(+ self.$n.ln_f_stat(&stat.$n))*
                }
            }
        )+
    };
}

macro_rules! tuple_conjugate_prior {
    ($($len:expr => ($($n:tt $t:ident $x:ident $f:ident)*))+) => {
        $(
            #[cfg(feature = "experimental")]
            impl<$($t,)*$($x,)*$($f,)*> ConjugatePrior<($($x,)*),($($f,)*)> for ($($t,)*)
            where
                $($x: Copy,)*
                $($t: ConjugatePrior<$x, $f>,)*
                $($f: HasSuffStat<$x> + HasDensity<$x>,)*
            {
                type Posterior = ($($t::Posterior,)*);
                type MCache = ($($t::MCache,)*);
                type PpCache = ($($t::PpCache,)*);

                #[allow(unused_variables)]
                #[allow(clippy::unused_unit)]
                fn posterior_from_suffstat(&self, stat: &($($f::Stat,)*)) -> Self::Posterior {
                    ($(self.$n.posterior_from_suffstat(&stat.$n),)*)
                }

                fn empty_stat(&self) -> <($($f,)*) as HasSuffStat<($($x,)*)>>::Stat {
                    #[allow(clippy::unused_unit)]
                    ($(self.$n.empty_stat(),)*)
                }

                fn ln_m_cache(&self) -> Self::MCache {
                    #[allow(clippy::unused_unit)]
                    ($(self.$n.ln_m_cache(),)*)
                }

                #[allow(unused_variables)]
                fn ln_m_with_cache(
                    &self,
                    cache: &Self::MCache,
                    x: &DataOrSuffStat<($($x,)*), ($($f,)*)>,
                ) -> f64 {
                    match x {
                        DataOrSuffStat::Data(items) => {
                            0.0 $(+ self.$n.ln_m_with_cache(&cache.$n, &DataOrSuffStat::Data(&items.iter().map(|x| x.$n).collect::<Vec<_>>())))*
                        }
                        DataOrSuffStat::SuffStat(stats) => {
                            0.0 $(+ self.$n.ln_m_with_cache(&cache.$n, &DataOrSuffStat::SuffStat(&stats.$n)))*
                        }
                    }
                }

                #[allow(unused_variables)]
                #[allow(clippy::unused_unit)]
                fn ln_pp_cache(
                    &self,
                    x: &DataOrSuffStat<($($x,)*), ($($f,)*)>,
                ) -> Self::PpCache {
                    match x {
                        DataOrSuffStat::Data(items) => {
                            ($(self.$n.ln_pp_cache(&DataOrSuffStat::Data(&items.iter().map(|x| x.$n).collect::<Vec<_>>())),)*)
                        }
                        DataOrSuffStat::SuffStat(stats) => {
                            ($(self.$n.ln_pp_cache(&DataOrSuffStat::SuffStat(&stats.$n)),)*)
                        }
                    }
                }

                #[allow(unused_variables)]
                fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &($($x,)*)) -> f64 {
                    0.0 $(+ self.$n.ln_pp_with_cache(&cache.$n, &y.$n))*
                }
            }
        )+
    };
}

macro_rules! tuple_impls {
    ($($len:expr => ($($n:tt $t:ident $x:ident $f:ident)*))+) => {
        tuple_sampleable!(
            $(
                $len => ($($n $t $x)*)
            )+
        );
        tuple_has_density!(
            $(
                $len => ($($n $t $x)*)
            )+
        );
        tuple_support!(
            $(
                $len => ($($n $t $x)*)
            )+
        );
        tuple_discrete_continuous!(
            $(
                $len => ($($n $t $x)*)
            )+
        );
        tuple_mean_median_mode!(
            $(
                $len => ($($n $t $x)*)
            )+
        );
        tuple_cdf!(
            $(
                $len => ($($n $t $x)*)
            )+
        );
        tuple_suff_stat!(
            $(
                $len => ($($n $t $x)*)
            )+
        );
        tuple_has_suffstat!(
            $(
                $len => ($($n $t $x)*)
            )+
        );
        tuple_entropy!(
            $(
                $len => ($($n $t)*)
            )+
        );
        tuple_conjugate_prior!(
            $(
                $len => ($($n $t $x $f)*)
            )+
        );
    };
}

tuple_impls!(
    0 => ()
    1 => (0 T0 X0 F0)
    2 => (0 T0 X0 F0 1 T1 X1 F1)
    3 => (0 T0 X0 F0 1 T1 X1 F1 2 T2 X2 F2)
    4 => (0 T0 X0 F0 1 T1 X1 F1 2 T2 X2 F2 3 T3 X3 F3)
    5 => (0 T0 X0 F0 1 T1 X1 F1 2 T2 X2 F2 3 T3 X3 F3 4 T4 X4 F4)
    6 => (0 T0 X0 F0 1 T1 X1 F1 2 T2 X2 F2 3 T3 X3 F3 4 T4 X4 F4 5 T5 X5 F5)
    7 => (0 T0 X0 F0 1 T1 X1 F1 2 T2 X2 F2 3 T3 X3 F3 4 T4 X4 F4 5 T5 X5 F5 6 T6 X6 F6)
    8 => (0 T0 X0 F0 1 T1 X1 F1 2 T2 X2 F2 3 T3 X3 F3 4 T4 X4 F4 5 T5 X5 F5 6 T6 X6 F6 7 T7 X7 F7)
    9 => (0 T0 X0 F0 1 T1 X1 F1 2 T2 X2 F2 3 T3 X3 F3 4 T4 X4 F4 5 T5 X5 F5 6 T6 X6 F6 7 T7 X7 F7 8 T8 X8 F8)
    10 => (0 T0 X0 F0 1 T1 X1 F1 2 T2 X2 F2 3 T3 X3 F3 4 T4 X4 F4 5 T5 X5 F5 6 T6 X6 F6 7 T7 X7 F7 8 T8 X8 F8 9 T9 X9 F9)
    11 => (0 T0 X0 F0 1 T1 X1 F1 2 T2 X2 F2 3 T3 X3 F3 4 T4 X4 F4 5 T5 X5 F5 6 T6 X6 F6 7 T7 X7 F7 8 T8 X8 F8 9 T9 X9 F9 10 T10 X10 F10)
    12 => (0 T0 X0 F0 1 T1 X1 F1 2 T2 X2 F2 3 T3 X3 F3 4 T4 X4 F4 5 T5 X5 F5 6 T6 X6 F6 7 T7 X7 F7 8 T8 X8 F8 9 T9 X9 F9 10 T10 X10 F10 11 T11 X11 F11)
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dist::Gaussian, traits::HasDensity};
    use rand::SeedableRng;

    macro_rules! tuple_sampleable_test {
        ($($len:expr => ($($n:tt $x:tt)*))+) => {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1234);
            $(
                let d: ($($n,)*) = ($(<$n>::standard(),)*);
                let _x: ($($x,)*) = d.draw(&mut rng);
            )+
        };
    }

    macro_rules! tuple_density_test {
        ($($len:expr => ($($n:tt $x:tt)*))+) => {
            $(
                let d: ($($n,)*) = ($(<$n>::standard(),)*);
                let lnf0 = d.ln_f(&($($x::default(),)*));
                assert::close(lnf0, ($len as f64) * Gaussian::standard().ln_f(&0.0_f64), 1e-10);
            )+
        };
    }

    macro_rules! tuple_support_test{
        ($($len:expr => ($($n:tt $x:tt)*))+) => {
            // TODO: This isn't really checking anything...
            let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1234);
            $(
                let d: ($($n,)*) = ($(<$n>::standard(),)*);
                let x: ($($x,)*) = d.draw(&mut rng);
                let supported = d.supports(&x);
                assert!(supported);
            )+
        };
    }

    macro_rules! tuple_continuous_test {
        ($($len:expr => ($($n:tt $x:tt)*))+) => {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1234);
            $(
                let d: ($($n,)*) = ($(<$n>::standard(),)*);
                let x: ($($x,)*) = d.draw(&mut rng);
                assert::close(d.ln_f(&x), d.ln_pdf(&x), 1e-10);
            )+
        };
    }

    macro_rules! tuple_mean_median_mode_test {
        ($($len:expr => ($($n:ty, $x:ty),*))+) => {
            $(
                let d: ($($n,)*) = ($(<$n>::standard(),)*);
                let mean: ($($x,)*) = d.mean().unwrap();
                let median: ($($x,)*) = d.median().unwrap();
                let mode: ($($x,)*) = d.mode().unwrap();

                assert_eq!(mean, mode);
                assert_eq!(mean, median);

                let mean: [f64; $len] = mean.into();

                for m in mean {
                    assert_eq!(m, 0.0);
                }
            )+
        };
    }

    macro_rules! tuple_cdf_test {
        ($($len:expr => ($($n:tt $x:tt)*))+) => {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1234);
            $(
                let d: ($($n,)*) = ($(<$n>::standard(),)*);
                let x: ($($x,)*) = d.draw(&mut rng);

                let cdf = d.cdf(&x);
                let dparts: [_; $len] = d.into();
                let xparts: [f64; $len] = x.into();

                let cdf_product: f64 = dparts.into_iter()
                    .zip(xparts.into_iter())
                    .map(|(d, x)| d.cdf(&x))
                    .reduce(|acc, x| acc * x)
                    .unwrap();

                assert::close(cdf, cdf_product, 1e-10);
            )+
        };
    }

    macro_rules! tuple_entropy_test {
        ($($len:expr => ($($n:tt $x:tt)*))+) => {
            $(
                let d: ($($n,)*) = ($(<$n>::standard(),)*);

                let entropy = d.entropy();
                let dparts: [_; $len] = d.into();

                let entropy_sum: f64 = dparts.into_iter()
                    .map(|d| d.entropy())
                    .sum();

                assert::close(entropy, entropy_sum, 1e-10);
            )+
        };
    }

    macro_rules! tuple_has_suffstat_test {
        ($($len:expr => ($($n:tt $x:tt)*))+) => {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1234);
            use $crate::data::GaussianSuffStat;
            $(
                let d: ($($n,)*) = ($(<$n>::standard(),)*);
                let mut stat = <($($n,)*) as HasSuffStat<($($x,)*)>>::empty_suffstat(&d);
                let sample: Vec<($($x,)*)> = d.sample(5, &mut rng);
                //assert_eq!(SuffStat::<($($x,)*)>::n(&stat), 0);

                sample.iter().for_each(|x| stat.observe(x));
                //assert_eq!(SuffStat::<($($x,)*)>::n(&stat), 5);


                let mut stat_b = <($($n,)*) as HasSuffStat<($($x,)*)>>::empty_suffstat(&d);
                sample.iter().skip(2).for_each(|x| stat_b.observe(x));
                let mut stat_c = <($($n,)*) as HasSuffStat<($($x,)*)>>::empty_suffstat(&d);
                sample.iter().take(2).for_each(|x| stat_c.observe(x));

                SuffStat::<($($x,)*)>::merge(&mut stat_b, stat_c);

                let state_array: [GaussianSuffStat; $len] = stat.clone().into();
                let state_b_array: [GaussianSuffStat; $len] = stat_b.into();

                for i in 0..$len {
                    assert_eq!(state_array[i].n(), state_b_array[i].n());
                    assert::close(state_array[i].mean(), state_b_array[i].mean(), 1e-10);
                    assert::close(state_array[i].sum_x_sq(), state_b_array[i].sum_x_sq(), 1e-10);
                }

                sample.iter().take(2).for_each(|x| stat.forget(x));
                //assert_eq!(SuffStat::<($($x,)*)>::n(&stat), 3);
            )+
        };
    }

    macro_rules! tuple_impls_test {
        ($($len:expr => ($($n:ty, $x:tt),*))+) => {
            #[test]
            fn sampleable() {
                tuple_sampleable_test!(
                    $(
                        $len => ($($n $x)*)
                    )+
                );
            }
            #[test]
            fn product_of_gaussians_are_independent() {
                tuple_density_test!(
                    $(
                        $len => ($($n $x)*)
                    )+
                );
            }

            #[test]
            fn support() {
                tuple_support_test!(
                    $(
                        $len => ($($n $x)*)
                    )+
                );
            }

            #[test]
            fn continuous_has_pdf() {
                tuple_continuous_test!(
                    $(
                        $len => ($($n $x)*)
                    )+
                );
            }

            #[test]
            fn mean_and_mode() {
                tuple_mean_median_mode_test!(
                    $(
                        $len => ($($n, $x),*)
                    )+
                );
            }

            #[test]
            fn cdf() {
                tuple_cdf_test!(
                    $(
                        $len => ($($n $x)*)
                    )+
                );
            }
            #[test]
            fn entropy() {
                tuple_entropy_test!(
                    $(
                        $len => ($($n $x)*)
                    )+
                );
            }

            #[cfg(feature = "experimental")]
            #[test]
            fn suffstat() {
                tuple_has_suffstat_test!(
                    $(
                        $len => ($($n $x)*)
                    )+
                );
            }
            /*
            tuple_conjugate_prior_test!(
                $(
                    $len => ($($n $t $x $f)*)
                )+
            );
            */
        };
    }

    tuple_impls_test!(
      //0 =>  ()
      1 =>  (Gaussian, f64)
      2 =>  (Gaussian, f64, Gaussian, f64)
      3 =>  (Gaussian, f64, Gaussian, f64, Gaussian, f64)
      4 =>  (Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64)
      5 =>  (Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64)
      6 =>  (Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64)
      7 =>  (Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64)
      8 =>  (Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64)
      9 =>  (Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64)
      10 => (Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64)
      11 => (Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64)
      12 => (Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64, Gaussian, f64)
    );

    #[cfg(feature = "experimental")]
    #[test]
    fn independent_product_gaussians_conjugate() {
        use crate::{
            data::DataOrSuffStat,
            dist::NormalInvGamma,
            traits::{ConjugatePrior, SuffStat},
        };

        let f = (Gaussian::standard(), Gaussian::standard());
        let h = (
            NormalInvGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
            NormalInvGamma::new_unchecked(2.0, 3.0, 3.0, 3.0),
        );

        assert_eq!(h.ln_f(&f), h.0.ln_f(&f.0) + h.1.ln_f(&f.1));

        let mut stat = h.empty_stat();
        stat.observe(&(1.0, 2.0));
        stat.observe(&(-1.0, 0.0));

        assert_eq!(stat.0.n(), 2);
        assert_eq!(stat.1.n(), 2);

        assert_eq!(stat.0.mean(), 0.0);
        assert_eq!(stat.1.mean(), 1.0);

        assert_eq!(stat.0.sum_sq_diff(), 2.0);
        assert_eq!(stat.1.sum_sq_diff(), 2.0);

        let posterior =
            h.posterior(&crate::data::DataOrSuffStat::SuffStat(&stat));

        assert_eq!(
            h.0.posterior(&DataOrSuffStat::SuffStat(&stat.0)),
            posterior.0
        );
        assert_eq!(
            h.1.posterior(&DataOrSuffStat::SuffStat(&stat.1)),
            posterior.1
        );
    }
}
