extern crate rand;

use self::rand::Rng;

/// Random variable.
pub trait Rv {
    type DatumType;

    /// Un-normalized probability function
    fn f(&self, x: &Self::DatumType) -> f64 {
        self.ln_f(x).exp()
    }

    /// Un-normalized probability function
    fn ln_f(&self, x: &Self::DatumType) -> f64;

    /// The constant term in the PDF/PMF. Should not be a function of any of
    /// the parameters.
    fn normalizer(&self) -> f64 {
        self.ln_normalizer().exp()
    }

    /// The log of the constant term in the PDF/PMF. Should not be a function of
    /// any of the parameters.
    fn ln_normalizer(&self) -> f64;

    /// Single draw from the `Rv`
    fn draw<R: Rng>(&self, rng: &mut R) -> Self::DatumType;

    /// Multiple draws of the `Rv`
    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<Self::DatumType> {
        (0..n).map(|_| self.draw(&mut rng)).collect()
    }
}

/// Trait identefying the support of the distribution
pub trait Support: Rv {
    /// Returns `true` if `x` is in the support of the `Rv`
    fn contains(&self, x: &Self::DatumType) -> bool;
}

/// Trait describing continuous probability distributions
pub trait ContinuousDistr: Rv {
    /// The value of the Probability Density Function (PDF) at `x`
    fn pdf(&self, x: &Self::DatumType) -> f64 {
        self.ln_pdf(x).exp()
    }

    /// The value of the log Probability Density Function (PDF) at `x`
    fn ln_pdf(&self, x: &Self::DatumType) -> f64 {
        self.ln_f(x) - self.ln_normalizer()
    }
}

pub trait Cdf: Rv {
    /// The value of the Cumulative Density Function at `x`
    fn cdf(&self, x: &Self::DatumType) -> f64;

    /// Survival function
    fn sf(&self, x: &Self::DatumType) -> f64 {
        1.0 - self.cdf(x)
    }
}

pub trait InverseCdf: Rv {
    /// The value of the `x` at the given probability in the CDF
    fn invcdf(&self, p: &Self::DatumType) -> Self::DatumType;

    /// Alias for `invcdf`
    fn quantile(&self, p: &Self::DatumType) -> Self::DatumType {
        self.invcdf(p)
    }
}

pub trait DiscreteDistr: Rv {
    fn pmf(&self, x: &Self::DatumType) -> f64 {
        self.ln_pmf(x).exp()
    }

    fn ln_pmf(&self, x: &Self::DatumType) -> f64 {
        self.ln_f(x) - self.ln_normalizer()
    }
}

pub trait Mean<M>: Rv {
    fn mean(&self) -> Option<M>;
}

pub trait Median<M>: Rv {
    fn median(&self) -> Option<M>;
}

pub trait Mode: Rv {
    fn mode(&self) -> Option<Self::DatumType>;
}

pub trait Variance<V>: Rv {
    fn variance(&self) -> Option<V>;
}

pub trait Entropy: Rv {
    fn entropy(&self) -> f64;
}

pub trait Skewness: Rv {
    fn skewness(&self) -> Option<f64>;
}

pub trait Kurtosis: Rv {
    fn kurtosis(&self) -> Option<f64>;
}
