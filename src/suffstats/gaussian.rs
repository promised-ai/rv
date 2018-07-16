use traits::SuffStat;

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

macro_rules! impl_suffstat {
    ($kind:ty) => {
        // TODO: store in a more nuerically stable form
        impl SuffStat<$kind> for GaussianSuffStat {
            fn observe(&mut self, x: &$kind) {
                let xf = *x as f64;
                self.n += 1;
                self.sum_x += xf;
                self.sum_x_sq += xf.powi(2);
            }

            fn forget(&mut self, x: &$kind) {
                if self.n > 1 {
                    let xf = *x as f64;
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

impl_suffstat!(f32);
impl_suffstat!(f64);
