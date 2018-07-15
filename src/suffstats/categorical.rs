use dist::categorical::CategoricalDatum;
use traits::SuffStat;

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

#[cfg(test)]
mod tests {
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
