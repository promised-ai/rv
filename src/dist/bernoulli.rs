extern crate rand;
extern crate special;

use self::rand::distributions::Uniform;
use self::rand::Rng;
use traits::*;

/// Bernoulli distribution with success probability *p*
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> Self {
        Bernoulli { p: p }
    }

    /// A Bernoulli distribution with a 50% chance of success
    pub fn uniform() -> Self {
        Bernoulli { p: 0.5 }
    }

    /// The complement of `p`, i.e. `(1 - p)`.
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

impl Cdf for Bernoulli {
    fn cdf(&self, x: &bool) -> f64 {
        if *x {
            1.0
        } else {
            self.q()
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

#[cfg(test)]
mod tests {
    extern crate assert;
    use super::*;

    const TOL: f64 = 1E-12;

    #[test]
    fn new() {
        let b = Bernoulli::new(0.1);
        assert::close(b.p, 0.1, TOL);
    }

    #[test]
    fn uniform_p_should_be_one_half() {
        let b = Bernoulli::uniform();
        assert::close(b.p, 0.5, TOL);
    }

    #[test]
    fn q_should_be_the_compliment_of_p() {
        let b = Bernoulli::new(0.1);
        assert::close(b.q(), 0.9, TOL);
    }

    #[test]
    fn pmf_of_1_should_be_p() {
        let b1 = Bernoulli::new(0.1);
        assert::close(b1.pmf(&true), 0.1, TOL);

        let b2 = Bernoulli::new(0.85);
        assert::close(b2.pmf(&true), 0.85, TOL);
    }

    #[test]
    fn ln_pmf_of_1_should_be_ln_p() {
        let b1 = Bernoulli::new(0.1);
        assert::close(b1.ln_pmf(&true), 0.1_f64.ln(), TOL);

        let b2 = Bernoulli::new(0.85);
        assert::close(b2.ln_pmf(&true), 0.85_f64.ln(), TOL);
    }

    #[test]
    fn pmf_of_0_should_be_q() {
        let b1 = Bernoulli::new(0.1);
        assert::close(b1.pmf(&false), 0.9, TOL);

        let b2 = Bernoulli::new(0.85);
        assert::close(b2.pmf(&false), 0.15, TOL);
    }

    #[test]
    fn ln_pmf_of_0_should_be_ln_q() {
        let b1 = Bernoulli::new(0.1);
        assert::close(b1.ln_pmf(&false), 0.9_f64.ln(), TOL);

        let b2 = Bernoulli::new(0.85);
        assert::close(b2.ln_pmf(&false), 0.15_f64.ln(), TOL);
    }

    #[test]
    fn sample_should_draw_the_correct_number_of_samples() {
        let mut rng = rand::thread_rng();
        let n = 103;
        let xs = Bernoulli::uniform().sample(n, &mut rng);
        assert_eq!(xs.len(), n);
    }

    #[test]
    fn contains_both_true_and_false() {
        let b = Bernoulli::uniform();
        assert!(b.contains(&true));
        assert!(b.contains(&false));
    }

    #[test]
    fn cmf_of_zero_is_q() {
        let b = Bernoulli::new(0.1);
        assert::close(b.cdf(&false), 0.9, TOL);
    }

    #[test]
    fn cmf_of_one_is_one() {
        let b = Bernoulli::new(0.1);
        assert::close(b.cdf(&true), 1.0, TOL);
    }

    #[test]
    fn mean_is_p() {
        assert::close(Bernoulli::new(0.1).mean().unwrap(), 0.1, TOL);
        assert::close(Bernoulli::new(0.7).mean().unwrap(), 0.7, TOL);
    }

    #[test]
    fn median_for_low_p_is_zero() {
        assert::close(Bernoulli::new(0.1).median().unwrap(), 0.0, TOL);
        assert::close(Bernoulli::new(0.499).median().unwrap(), 0.0, TOL);
    }

    #[test]
    fn median_for_high_p_is_one() {
        assert::close(Bernoulli::new(0.9).median().unwrap(), 1.0, TOL);
        assert::close(Bernoulli::new(0.5001).median().unwrap(), 1.0, TOL);
    }

    #[test]
    fn median_for_p_one_half_is_one_half() {
        assert::close(Bernoulli::new(0.5).median().unwrap(), 0.5, TOL);
        assert::close(Bernoulli::uniform().median().unwrap(), 0.5, TOL);
    }

    #[test]
    fn mode_for_high_p_is_true() {
        assert!(Bernoulli::new(0.5001).mode().unwrap());
        assert!(Bernoulli::new(0.8).mode().unwrap());
    }

    #[test]
    fn mode_for_low_p_is_true() {
        assert!(!Bernoulli::new(0.4999).mode().unwrap());
        assert!(!Bernoulli::new(0.2).mode().unwrap());
    }

    #[test]
    fn mode_for_even_p_is_none() {
        assert!(Bernoulli::new(0.5).mode().is_none());
        assert!(Bernoulli::uniform().mode().is_none());
    }

    #[test]
    fn variance_for_uniform() {
        assert::close(Bernoulli::uniform().variance().unwrap(), 0.25, TOL);
    }

    #[test]
    fn variance() {
        assert::close(Bernoulli::new(0.1).variance().unwrap(), 0.09, TOL);
        assert::close(Bernoulli::new(0.9).variance().unwrap(), 0.09, TOL);
    }

    #[test]
    fn entropy() {
        let b1 = Bernoulli::new(0.1);
        let b2 = Bernoulli::new(0.9);
        assert::close(b1.entropy(), 0.3250829733914482, TOL);
        assert::close(b2.entropy(), 0.3250829733914482, TOL);
    }

    #[test]
    fn unifrom_entropy() {
        let b = Bernoulli::uniform();
        assert::close(b.entropy(), 0.6931471805599453, TOL);
    }

    #[test]
    fn uniform_skewness_should_be_zero() {
        let b = Bernoulli::uniform();
        assert::close(b.skewness().unwrap(), 0.0, TOL);
    }

    #[test]
    fn skewness() {
        let b = Bernoulli::new(0.3);
        assert::close(b.skewness().unwrap(), 0.8728715609439696, TOL);
    }

    #[test]
    fn uniform_kurtosis() {
        let b = Bernoulli::uniform();
        assert::close(b.kurtosis().unwrap(), -2.0, TOL);
    }
}
