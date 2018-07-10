extern crate rand;
extern crate special;

use self::rand::Rng;
use self::special::Beta as SBeta;
use self::special::Gamma as SGamma;
use self::rand::distributions::Gamma;
use traits::*;

pub struct Beta {
    alpha: f64,
    beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Beta { alpha: alpha, beta: beta }
    }

    pub fn uniform() -> Self {
        Beta::new(1.0, 1.0)
    }

    pub fn jeffreys() -> Self {
        Beta::new(0.5, 0.5)
    }
}

impl Default for Beta {
    fn default() -> Self {
        Beta::jeffreys()
    }
}

impl Rv for Beta {
    type DatumType = f64;

    fn ln_f(&self, x: &f64) -> f64 {
        let denom = self.alpha.ln_beta(self.beta);
        x.powf(self.alpha - 1.0) * (1.0 - *x).powf(self.beta - 1.0) - denom
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> f64 {
        let ga = Gamma::new(self.alpha, 1.0);
        let gb = Gamma::new(self.beta, 1.0);
        let a = rng.sample(ga);
        let b = rng.sample(gb);
        a / (a + b)
    }

    fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<f64> {
        let ga = Gamma::new(self.alpha, 1.0);
        let gb = Gamma::new(self.beta, 1.0);
        (0..n)
            .map(|_| {
                let a = rng.sample(ga);
                let b = rng.sample(gb);
                a / (a + b)
            })
            .collect()
    }

    #[inline]
    fn ln_normalizer(&self) -> f64 {
        0.0
    }
}

impl ContinuousDistr for Beta {}

impl Mean<f64> for Beta {
    fn mean(&self) -> Option<f64> {
        Some(self.alpha / (self.alpha + self.beta))
    }
}

impl Mode for Beta {
    fn mode(&self) -> Option<f64> {
       if self.beta > 1.0 {
           if self.alpha > 1.0 {
                Some((self.alpha - 1.0) / (self.alpha + self.beta - 2.0))
           } else if self.alpha == 1.0 {
               Some(0.0)
           } else {
               None
           }
       } else if self.beta == 1.0 {
           if self.alpha > 1.0 {
               Some(1.0)
           } else {
               None
           }
       } else {
           None
       }
    }
}

impl Variance<f64> for Beta {
    fn variance(&self) -> Option<f64> {
        let apb = self.alpha + self.beta;
        Some(self.alpha * self.beta / (apb * apb * (apb + 1.0)))
    }
}

impl Entropy for Beta {
    fn entropy(&self) -> f64 {
        let apb = self.alpha + self.beta;
        self.alpha.ln_beta(self.beta)
            - (self.alpha - 1.0) * self.alpha.digamma()
            - (self.beta - 1.0) * self.beta.digamma()
            + (apb + 2.0) * apb.digamma()
    }
}

impl Skewness for Beta {
    fn skewness(&self) -> Option<f64> {
        let apb = self.alpha + self.beta;
        let numer = 2.0*(self.beta - self.alpha) * (apb + 1.0).sqrt();
        let denom = self.alpha*self.beta*(apb + 2.0)*(apb + 3.0);
        Some(numer / denom)
    }
}

impl Kurtosis for Beta {
    fn kurtosis(&self) -> Option<f64> {
        let apb = self.alpha + self.beta;
        let amb = self.alpha - self.beta;
        let atb = self.alpha * self.beta;
        let numer = 6.0 * (amb * amb * (apb + 1.0) - atb*(apb + 2.0));
        let denom = atb * (apb + 2.0) * (apb + 3.0);
        Some(numer / denom)
    }
}
