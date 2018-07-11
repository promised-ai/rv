extern crate rand;
extern crate special;

use self::rand::distributions::Uniform;
use self::rand::Rng;
use traits::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> Self {
        Bernoulli { p: p }
    }

    pub fn uniform() -> Self {
        Bernoulli { p: 0.5 }
    }

    #[inline]
    pub fn q(&self) -> f64 {
        1.0 - self.p
    }
}

impl Default for Bernoulli {
    fn default() -> Self {
        Bernoulli::uniform()
    }
}

impl Rv for Bernoulli {
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

impl Support for Bernoulli {
    fn contains(&self, _x: &bool) -> bool {
        true
    }
}

impl DiscreteDistr for Bernoulli {
    fn pmf(&self, x: &bool) -> f64 {
        self.f(x)
    }

    fn ln_pmf(&self, x: &bool) -> f64 {
        self.ln_f(x)
    }
}

impl Cmf for Bernoulli {
    fn cmf(&self, x: &bool) -> f64 {
        if *x {
            1.0
        } else {
            self.p
        }
    }
}

impl Mean<f64> for Bernoulli {
    fn mean(&self) -> Option<f64> {
        Some(self.p)
    }
}

impl Median<f64> for Bernoulli {
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

impl Mode for Bernoulli {
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

impl Variance<f64> for Bernoulli {
    fn variance(&self) -> Option<f64> {
        Some(self.p * (1.0 - self.p))
    }
}

impl Entropy for Bernoulli {
    fn entropy(&self) -> f64 {
        let q = self.q();
        -q * q.ln() - self.p * self.p.ln()
    }
}

impl Skewness for Bernoulli {
    fn skewness(&self) -> Option<f64> {
        Some((1.0 - 2.0 * self.p) / (self.p * self.q()).sqrt())
    }
}

impl Kurtosis for Bernoulli {
    fn kurtosis(&self) -> Option<f64> {
        let q = self.q();
        Some((1.0 - 6.0 * self.p * q) / (self.p * q))
    }
}
