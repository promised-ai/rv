#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

use crate::data::DataOrSuffStat;
use crate::dist::VonMises;
use crate::traits::SuffStat;

/// VonMises sufficient statistic.
/// 
/// Holds number of observations and vector sum components
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct VonMisesSuffStat {
    /// Number of observations
    n: usize,
    /// cosine component of vector sum
    x: f64,
    /// sine compnent of vector sum
    y: f64,
}

impl VonMisesSuffStat {
    #[inline]
    pub fn new() -> Self {
        VonMisesSuffStat {
            n: 0,
            x: 0.0,
            y: 0.0,
        }
    }

    /// Create a suffucuent statitic without checking whether they are valid.
    #[inline]
    pub fn from_parts_unchecked(n: usize, x: f64, y: f64) -> Self {
        VonMisesSuffStat {n, x, y }
    }

    /// Get the number of observations
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the cosine component of the sample sum
    #[inline]
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Get the sine component of the sample sum
    #[inline]
    pub fn y(&self) -> f64 {
        self.y
    }
    
    // /// Magnitude of sum vector (never used alone, always with the prior params)
    // #[inline]
    // pub fn mag(&self) -> f64 {
    //     (self.x * self.x + self.y * self.y).sqrt()        
    // }

    // /// Angle of sum vector (never used alone: always with the prior params)
    // pub fn angle(&self) -> f64 {
    //     const PI: f64 = std::f64::consts::PI;
    //     let ang = self.y.atan2(self.x);
    //     // make ang ranges from [0, 2pi]
    //     if ang >= 0 {
    //         ang
    //     } else {
    //         ang + 2 * PI
    //     }
    // }
}

macro_rules! impl_VonMises_suffstat {
    ($kind:ty) => {
        impl<'a> From<&'a VonMisesSuffStat>
            for DataOrSuffStat<'a, $kind, VonMises>
        {
            fn from(stat: &'a VonMisesSuffStat) -> Self {
                DataOrSuffStat::SuffStat(stat)
            }
        }

        impl<'a> From<&'a Vec<$kind>> for DataOrSuffStat<'a, $kind, VonMises> {
            fn from(xs: &'a Vec<$kind>) -> Self {
                DataOrSuffStat::Data(xs)
            }
        }
        
        impl<'a> From<&'a [$kind]> for DataOrSuffStat<'a, $kind, VonMises> {
            fn from(xs: &'a [$kind]) -> Self {
                DataOrSuffStat::Data(xs)
            }
        }

        // impl From<&<Vec<&kind>> for VonMisesSuffStat {
        //     fn from(xs: &Vec<&kind>) -> Self {
        //         let mut stat = VonMisesSuffStat::new();
        //         stat.observe_many(xs);
        //         stat
        //     }
        // }

        // impl From<&[&kind]> for VonMisesSuffStat {
        //     fn from(xs: &[&kind]) -> Self {
        //         let mut stat = VonMisesSuffStat::new();
        //         stat.observe_many(xs);
        //         stat
        //     }
        // }

        impl SuffStat<$kind> for VonMisesSuffStat {
            fn n(&self) -> usize {
                self.n
            }

            fn observe(&mut self, a: &$kind) {
                let af = f64::from(*a);

                // TODO: af should be >=0 and <= 2.0*PI
                
                self.n += 1;
                self.x += af.cos();
                self.y += af.sin();
            }

            fn forget(&mut self, a: &$kind) {
                if self.n > 1 {
                    let af = f64::from(*a);

                    self.n -= 1;
                    self.x -= af.cos();
                    self.y -= af.sin();
                } else {
                    self.n = 0;
                    self.x = 0.0;
                    self.y = 0.0;
                }
            }
        }
    };
}

impl_VonMises_suffstat!(f32);
impl_VonMises_suffstat!(f64);

#[cfg(test)]
mod test {
    use super::*;
    const PI: f64 = std::f64::consts::PI;
    const EPSILON: f64 = f64::EPSILON;

    #[test]
    fn from_parts_unchecked() {
        let stat = VonMisesSuffStat::from_parts_unchecked(10, 1.9, 2.3);
        assert_eq!(stat.n(), 10);
        assert_eq!(stat.x(), 1.9);
        assert_eq!(stat.y(), 2.3);
    }

    #[test]
    fn suffstat_increments_correctly() {
        let afs: Vec<f64> = vec![0.0, PI/2.0, PI, 1.5*PI];
        let mut suffstat = VonMisesSuffStat::new();

        for af in afs {
            suffstat.observe(&af);
        }

        assert_eq!(suffstat.n(), 4);
        assert::close(suffstat.x(), 0.0, 1e-14);
        assert::close(suffstat.y(), 0.0, 1e-14);
    }

    #[test]
    fn suffstat_decrements_correctly() {
        let afs: Vec<f64> = vec![0.0, PI/2.0, PI, 1.5*PI];
        let mut suffstat = VonMisesSuffStat::new();

        for af in afs {
            suffstat.observe(&af);
        }

        suffstat.forget(&0.0);

        assert_eq!(suffstat.n(), 3);
        assert::close(suffstat.x(), -1.0, 1e-14);
        assert::close(suffstat.y(), 0.0, 1e-14);
    }

    // #[test]
    // fn magnitude_leq_n() {
    //     let afs: Vec<f64> = vec![PI/4.0, PI/4.0, PI/4.0, PI/4.0];
    //     let mut suffstat = VonMisesSuffStat::new();

    //     suffstat.observe_many(afs);

    //     let x = suffstat.x();
    //     let y = suffstat.y();
    //     let mag = (x*x + y*y).sqrt();
    //     assert::close(mag, suffstat.n as f64, 1e-14);
    // }

    // #[test]
    // fn catch_nonvalid_inputs() {}
}