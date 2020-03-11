//! Pareto distribution over x in [shape, ∞)
#[cfg(feature = "serde1")]
use serde_derive::{Deserialize, Serialize};

use crate::impl_display;
use crate::traits::*;
use rand::Rng;
use std::f64;
use std::fmt;

/// [Pareto distribution](https://en.wikipedia.org/wiki/Pareto_distribution) Pareto(x_m, α)
/// over x in (x_m, ∞).
///
/// **NOTE**: The Pareto distribution is parameterized in terms of shape, α, and
/// scale, x_m.
///
/// ```math
///                α x_m^α
/// f(x|α, x_m) = ---------
///               x^(α + 1)
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Pareto {
    shape: f64,
    scale: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum ParetoError {
    /// Shape parameter is less than or equal to zero
    ShapeTooLow { shape: f64 },
    /// Shape parameter is infinite or NaN
    ShapeNotFinite { shape: f64 },
    /// Scale parameter is less than or equal to zero
    ScaleTooLow { scale: f64 },
    /// Scale parameter is infinite or NaN
    ScaleNotFinite { scale: f64 },
}

impl Pareto {
    /// Create a new `Pareto` distribution with shape (α) and scale (x_m).
    pub fn new(shape: f64, scale: f64) -> Result<Self, ParetoError> {
        if shape <= 0.0 {
            Err(ParetoError::ShapeTooLow { shape })
        } else if !shape.is_finite() {
            Err(ParetoError::ShapeNotFinite { shape })
        } else if scale <= 0.0 {
            Err(ParetoError::ScaleTooLow { scale })
        } else if !scale.is_finite() {
            Err(ParetoError::ScaleNotFinite { scale })
        } else {
            Ok(Pareto { shape, scale })
        }
    }

    /// Creates a new Pareto without checking whether the parameters are valid.
    #[inline]
    pub fn new_unchecked(shape: f64, scale: f64) -> Self {
        Pareto { shape, scale }
    }

    /// Get shape parameter
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::Pareto;
    /// let pareto = Pareto::new(1.0, 2.0).unwrap();
    /// assert_eq!(pareto.shape(), 1.0);
    /// ```
    #[inline]
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Set the shape parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::Pareto;
    /// let mut pareto = Pareto::new(2.0, 1.0).unwrap();
    /// assert_eq!(pareto.shape(), 2.0);
    ///
    /// pareto.set_shape(1.1).unwrap();
    /// assert_eq!(pareto.shape(), 1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Pareto;
    /// # let mut pareto = Pareto::new(2.0, 1.0).unwrap();
    /// assert!(pareto.set_shape(1.1).is_ok());
    /// assert!(pareto.set_shape(0.0).is_err());
    /// assert!(pareto.set_shape(-1.0).is_err());
    /// assert!(pareto.set_shape(std::f64::INFINITY).is_err());
    /// assert!(pareto.set_shape(std::f64::NEG_INFINITY).is_err());
    /// assert!(pareto.set_shape(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_shape(&mut self, shape: f64) -> Result<(), ParetoError> {
        if shape <= 0.0 {
            Err(ParetoError::ShapeTooLow { shape })
        } else if !shape.is_finite() {
            Err(ParetoError::ShapeNotFinite { shape })
        } else {
            self.set_shape_unchecked(shape);
            Ok(())
        }
    }

    /// Set the shape parameter without input validation
    #[inline]
    pub fn set_shape_unchecked(&mut self, shape: f64) {
        self.shape = shape;
    }

    /// Get scale parameter
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::dist::Pareto;
    /// let pareto = Pareto::new(1.0, 2.0).unwrap();
    /// assert_eq!(pareto.scale(), 2.0);
    /// ```
    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Set the scale parameter
    ///
    /// # Example
    ///
    /// ```rust
    /// use rv::dist::Pareto;
    /// let mut pareto = Pareto::new(2.0, 1.0).unwrap();
    /// assert_eq!(pareto.scale(), 1.0);
    ///
    /// pareto.set_scale(1.1).unwrap();
    /// assert_eq!(pareto.scale(), 1.1);
    /// ```
    ///
    /// Will error for invalid values
    ///
    /// ```rust
    /// # use rv::dist::Pareto;
    /// # let mut pareto = Pareto::new(2.0, 1.0).unwrap();
    /// assert!(pareto.set_scale(1.1).is_ok());
    /// assert!(pareto.set_scale(0.0).is_err());
    /// assert!(pareto.set_scale(-1.0).is_err());
    /// assert!(pareto.set_scale(std::f64::INFINITY).is_err());
    /// assert!(pareto.set_scale(std::f64::NEG_INFINITY).is_err());
    /// assert!(pareto.set_scale(std::f64::NAN).is_err());
    /// ```
    #[inline]
    pub fn set_scale(&mut self, scale: f64) -> Result<(), ParetoError> {
        if scale <= 0.0 {
            Err(ParetoError::ScaleTooLow { scale })
        } else if !scale.is_finite() {
            Err(ParetoError::ScaleNotFinite { scale })
        } else {
            self.set_scale_unchecked(scale);
            Ok(())
        }
    }

    /// Set the scale parameter without input validation
    #[inline]
    pub fn set_scale_unchecked(&mut self, scale: f64) {
        self.scale = scale;
    }
}

impl From<&Pareto> for String {
    fn from(pareto: &Pareto) -> String {
        format!("Pareto(xₘ: {}, α: {})", pareto.scale, pareto.shape)
    }
}

impl_display!(Pareto);

macro_rules! impl_traits {
    ($kind:ty) => {
        impl Rv<$kind> for Pareto {
            fn ln_f(&self, x: &$kind) -> f64 {
                // TODO: cache ln(shape) and ln(scale)
                self.shape.ln() + self.shape * self.scale.ln()
                    - (self.shape + 1.0) * f64::from(*x).ln()
            }

            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let p =
                    rand_distr::Pareto::new(self.scale, self.shape).unwrap();
                rng.sample(p) as $kind
            }
        }

        impl ContinuousDistr<$kind> for Pareto {}

        impl Support<$kind> for Pareto {
            fn supports(&self, x: &$kind) -> bool {
                x.is_finite() && *x >= self.scale as $kind
            }
        }

        impl Cdf<$kind> for Pareto {
            fn cdf(&self, x: &$kind) -> f64 {
                let xk = f64::from(*x);
                1.0 - (self.scale / xk).powf(self.shape)
            }
        }

        impl Mean<$kind> for Pareto {
            fn mean(&self) -> Option<$kind> {
                if self.shape <= 1.0 {
                    Some(f64::INFINITY as $kind)
                } else {
                    Some(
                        ((self.shape * self.scale) / (self.shape - 1.0))
                            as $kind,
                    )
                }
            }
        }

        impl Mode<$kind> for Pareto {
            fn mode(&self) -> Option<$kind> {
                Some(self.scale as $kind)
            }
        }
    };
}

impl Variance<f64> for Pareto {
    fn variance(&self) -> Option<f64> {
        if self.shape <= 2.0 {
            Some(f64::INFINITY)
        } else {
            Some(
                (self.scale * self.scale * self.shape)
                    / ((self.shape - 1.0)
                        * (self.shape - 1.0)
                        * (self.shape - 2.0)),
            )
        }
    }
}

impl Entropy for Pareto {
    fn entropy(&self) -> f64 {
        ((self.scale / self.shape) * (1.0 + 1.0 / self.shape).exp()).log10()
    }
}

impl Skewness for Pareto {
    fn skewness(&self) -> Option<f64> {
        if self.shape <= 3.0 {
            None
        } else {
            Some(
                ((2.0 * (1.0 + self.shape)) / (self.shape - 3.0))
                    * ((self.shape - 2.0) / self.shape).sqrt(),
            )
        }
    }
}

impl Kurtosis for Pareto {
    fn kurtosis(&self) -> Option<f64> {
        let s = self.shape;
        if s <= 4.0 {
            None
        } else {
            Some(
                6.0 * (s * s * s + s * s - 6.0 * s - 2.0)
                    / (s * (s - 3.0) * (s - 4.0)),
            )
        }
    }
}

impl_traits!(f32);
impl_traits!(f64);

impl std::error::Error for ParetoError {}

impl fmt::Display for ParetoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeTooLow { shape } => {
                write!(f, "rate ({}) must be greater than zero", shape)
            }
            Self::ShapeNotFinite { shape } => {
                write!(f, "non-finite rate: {}", shape)
            }
            Self::ScaleTooLow { scale } => {
                write!(f, "scale ({}) must be greater than zero", scale)
            }
            Self::ScaleNotFinite { scale } => {
                write!(f, "non-finite scale: {}", scale)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::{ks_test, linspace};
    use crate::test_basic_impls;
    use std::f64;

    const TOL: f64 = 1E-12;
    const KS_PVAL: f64 = 0.2;
    const N_TRIES: usize = 5;

    test_basic_impls!(Pareto::new(1.0, 2.0).unwrap());

    #[test]
    fn new() {
        let par = Pareto::new(1.0, 2.0).unwrap();
        assert::close(par.shape, 1.0, TOL);
        assert::close(par.scale, 2.0, TOL);
    }

    #[test]
    fn ln_pdf() {
        let par = Pareto::new(1.0, 1.0).unwrap();
        let xs: Vec<f64> = linspace(1.0, 10.0, 50);
        let known_values = vec![
            0.0,
            -0.33724542487159,
            -0.62574464256068,
            -0.87782608435141,
            -1.10166191675938,
            -1.30294896831875,
            -1.48581738023802,
            -1.65335714636894,
            -1.80794049497223,
            -1.95142830468991,
            -2.08530727004013,
            -2.21078395130698,
            -2.32885101447536,
            -2.44033498049183,
            -2.54593135162578,
            -2.64623091899672,
            -2.74173978158852,
            -2.83289479858116,
            -2.92007567073088,
            -3.00361449648347,
            -3.08380341088723,
            -3.1609007511217,
            -3.2351360770347,
            -3.30671429273787,
            -3.37581905575119,
            -3.44261561655489,
            -3.50725319906522,
            -3.56986700831531,
            -3.6305799332765,
            -3.68950399873713,
            -3.74674160934844,
            -3.80238662054703,
            -3.85652526448347,
            -3.9092369538942,
            -3.96059498272958,
            -4.01066713905223,
            -4.05951624306638,
            -4.1072006209919,
            -4.15377452374948,
            -4.19928849799471,
            -4.24378971586435,
            -4.28732226882826,
            -4.329927430236,
            -4.37164389047682,
            -4.41250796811123,
            -4.45255379986144,
            -4.49181351195122,
            -4.53031737494986,
            -4.56809394399027,
            -4.60517018598809,
        ];

        let generated_values: Vec<f64> =
            xs.iter().map(|x| par.ln_f(x)).collect();
        assert::close(known_values, generated_values, TOL);
    }

    #[test]
    fn cdf_shape_2() {
        let par = Pareto::new(2.0, 1.0).unwrap();
        let xs: Vec<f64> = linspace(1.0, 10.0, 50);
        let known_values = vec![
            0.,
            0.28626634958383,
            0.46513700155937,
            0.58431440443213,
            0.66768166089965,
            0.72827071072884,
            0.77368272221699,
            0.80859375,
            0.83600846936685,
            0.85792899408284,
            0.8757310698204,
            0.89038531775018,
            0.9025923972575,
            0.91286834083321,
            0.9216,
            0.92908199432892,
            0.93554189374211,
            0.94115772963435,
            0.94607039374677,
            0.95039256198347,
            0.95421521328731,
            0.9576124567474,
            0.96064515071547,
            0.96336364746094,
            0.96580989676041,
            0.96801907400501,
            0.97002085180237,
            0.97184040157628,
            0.97349918875068,
            0.97501560874089,
            0.9764054991598,
            0.97768255502677,
            0.97885866741805,
            0.97994420127635,
            0.98094822455862,
            0.98187869822485,
            0.98274263453342,
            0.98354622954415,
            0.98429497452267,
            0.98499375,
            0.9856469055063,
            0.98625832741924,
            0.98683149690943,
            0.98736953960104,
            0.98787526827421,
            0.98835121970153,
            0.98879968652184,
            0.98922274490089,
            0.98962227860357,
            0.99,
        ];
        let generated_values: Vec<f64> =
            xs.iter().map(|x| par.cdf(x)).collect();
        assert::close(known_values, generated_values, TOL);
    }

    #[test]
    fn cdf_scale_2() {
        let par = Pareto::new(1.0, 2.0).unwrap();
        let xs: Vec<f64> = linspace(2.0, 10.0, 50);
        let known_values = vec![
            0.,
            0.07547169811321,
            0.14035087719298,
            0.19672131147541,
            0.24615384615385,
            0.28985507246377,
            0.32876712328767,
            0.36363636363636,
            0.39506172839506,
            0.42352941176471,
            0.44943820224719,
            0.47311827956989,
            0.49484536082474,
            0.51485148514851,
            0.53333333333333,
            0.55045871559633,
            0.56637168141593,
            0.58119658119658,
            0.59504132231405,
            0.608,
            0.62015503875969,
            0.63157894736842,
            0.64233576642336,
            0.65248226950355,
            0.66206896551724,
            0.67114093959732,
            0.6797385620915,
            0.68789808917197,
            0.69565217391304,
            0.7030303030303,
            0.71005917159763,
            0.71676300578035,
            0.72316384180791,
            0.7292817679558,
            0.73513513513514,
            0.74074074074074,
            0.74611398963731,
            0.75126903553299,
            0.75621890547264,
            0.7609756097561,
            0.76555023923445,
            0.76995305164319,
            0.7741935483871,
            0.77828054298643,
            0.78222222222222,
            0.78602620087336,
            0.78969957081545,
            0.79324894514768,
            0.79668049792531,
            0.8,
        ];
        let generated_values: Vec<f64> =
            xs.iter().map(|x| par.cdf(x)).collect();
        assert::close(known_values, generated_values, TOL);
    }

    #[test]
    fn mean() {
        let m1: f64 = Pareto::new(0.5, 1.0).unwrap().mean().unwrap();
        let m2: f64 = Pareto::new(1.0, 1.0).unwrap().mean().unwrap();
        let m3: f64 = Pareto::new(3.0, 1.0).unwrap().mean().unwrap();
        let m4: f64 = Pareto::new(2.0, 0.1).unwrap().mean().unwrap();
        assert::close(m1, f64::INFINITY, TOL);
        assert::close(m2, f64::INFINITY, TOL);
        assert::close(m3, 1.5, TOL);
        assert::close(m4, 0.2, TOL);
    }

    #[test]
    fn mode() {
        let m1: f64 = Pareto::new(2.0, 2.0).unwrap().mode().unwrap();
        let m2: f64 = Pareto::new(1.0, 2.0).unwrap().mode().unwrap();
        let m3: f64 = Pareto::new(5.0, 1.0).unwrap().mode().unwrap();
        assert::close(m1, 2.0, TOL);
        assert::close(m2, 2.0, TOL);
        assert::close(m3, 1.0, TOL);
    }

    #[test]
    fn variance() {
        let v1: f64 = Pareto::new(1.0, 1.0).unwrap().variance().unwrap();
        let v2: f64 = Pareto::new(2.0, 1.0).unwrap().variance().unwrap();
        let v3: f64 = Pareto::new(3.0, 1.0).unwrap().variance().unwrap();
        let v4: f64 = Pareto::new(3.0, 2.0).unwrap().variance().unwrap();

        assert::close(v1, f64::INFINITY, TOL);
        assert::close(v2, f64::INFINITY, TOL);
        assert::close(v3, 0.75, TOL);
        assert::close(v4, 3.0, TOL);
    }

    #[test]
    fn skewness() {
        let s1 = Pareto::new(1.0, 1.0).unwrap().skewness();
        let s2 = Pareto::new(3.0, 1.0).unwrap().skewness();
        let s3 = Pareto::new(4.0, 1.0).unwrap().skewness();
        let s4 = Pareto::new(4.0, 2.0).unwrap().skewness();

        assert!(s1.is_none());
        assert!(s2.is_none());
        assert::close(s3.unwrap(), 7.0710678118654755, TOL);
        assert::close(s4.unwrap(), 7.0710678118654755, TOL);
    }

    #[test]
    fn kurtosis() {
        let k1 = Pareto::new(1.0, 1.0).unwrap().kurtosis();
        let k2 = Pareto::new(4.0, 1.0).unwrap().kurtosis();
        let k3 = Pareto::new(5.0, 1.0).unwrap().kurtosis();
        let k4 = Pareto::new(5.0, 2.0).unwrap().kurtosis();

        assert!(k1.is_none());
        assert!(k2.is_none());
        assert::close(k3.unwrap(), 70.8, TOL);
        assert::close(k4.unwrap(), 70.8, TOL);
    }

    #[test]
    fn entropy() {
        let par1 = Pareto::new(1.0, 1.0).unwrap();
        let par2 = Pareto::new(1.2, 3.4).unwrap();
        assert::close(par1.entropy(), 0.86858896380650363, TOL);
        assert::close(par2.entropy(), 1.2485042211505921, TOL);
    }

    #[test]
    fn draw_test() {
        let mut rng = rand::thread_rng();
        let par = Pareto::new(1.2, 3.4).unwrap();
        let cdf = |x: f64| par.cdf(&x);

        // test is flaky, try a few times
        let passes = (0..N_TRIES).fold(0, |acc, _| {
            let xs: Vec<f64> = par.sample(1000, &mut rng);
            let (_, p) = ks_test(&xs, cdf);
            if p > KS_PVAL {
                acc + 1
            } else {
                acc
            }
        });

        assert!(passes > 0);
    }
}
