#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "arraydist")]
use crate::nalgebra::{DMatrix, DVector};
use crate::traits::*;

/// Represents any Datum/Value, X, for which Rv<X> may be implemented on a
/// `Distribution`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum Datum {
    F64(f64),
    F32(f32),
    Bool(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    ISize(isize),
    USize(usize),
    Vec(Vec<f64>),
    #[cfg(feature = "arraydist")]
    DVector(DVector<f64>),
    #[cfg(feature = "arraydist")]
    DMatrix(DMatrix<f64>),
    Compound(Vec<Datum>),
}

pub trait RvDatum
where
    Self: Rv<<Self as RvDatum>::Support>,
{
    type Support: From<Datum> + Into<Datum>;
}

impl<Fx> HasDensity<Datum> for Fx
where
    Fx: RvDatum,
{
    fn ln_f(&self, x: &Datum) -> f64 {
        let y = <Self as RvDatum>::Support::from(x.clone());
        <Self as HasDensity<<Self as RvDatum>::Support>>::ln_f(self, &y)
    }
}

impl<Fx> Sampleable<Datum> for Fx
where
    Fx: RvDatum,
{
    fn draw<R: rand::Rng>(&self, rng: &mut R) -> Datum {
        let x =
            <Self as Sampleable<<Self as RvDatum>::Support>>::draw(self, rng);
        x.into()
    }

    fn sample<R: rand::Rng>(&self, n: usize, rng: &mut R) -> Vec<Datum> {
        <Self as Sampleable<<Self as RvDatum>::Support>>::sample(self, n, rng)
            .drain(..)
            .map(|x| x.into())
            .collect()
    }

    fn sample_stream<'r, R: rand::Rng>(
        &'r self,
        rng: &'r mut R,
    ) -> Box<dyn Iterator<Item = Datum> + 'r> {
        let iter =
            <Self as Sampleable<<Self as RvDatum>::Support>>::sample_stream(
                self, rng,
            )
            .map(|x| x.into());
        Box::new(iter)
    }
}

macro_rules! impl_rvdatum {
    ($dist: ident, $type: ty) => {
        #[cfg(feature = "datum")]
        impl RvDatum for $crate::dist::$dist {
            type Support = $type;
        }
    };
}

macro_rules! convert_datum {
    ($self:ty | $primary:ident, $( $variant:ident ),*) => (
        impl From<$self> for Datum {
            fn from(x: $self) -> Datum {
                Datum::$primary(x)
            }
        }

        impl From<Datum> for $self {
            fn from(datum: Datum) -> $self {
                match datum {
                    Datum::$primary(x) => x,
                    $(Datum::$variant(x) => x.into(),)*
                    Datum::Compound(mut xs) => {
                        if xs.len() == 1 {
                            xs.pop().unwrap().into()
                        } else {
                            panic!("failed")
                        }
                    }
                    _ => {
                        panic!("failed")
                    }
                }
            }
        }
    );
    ($self:ty | $primary:ident) => (
        convert_datum!($self | $primary, );
    )
}

convert_datum!(f32 | F32);
convert_datum!(f64 | F64, F32);
convert_datum!(bool | Bool);
convert_datum!(u8 | U8);
convert_datum!(u16 | U16, U8);
convert_datum!(u32 | U32, U8, U16);
convert_datum!(u64 | U64, U8, U16, U32);
convert_datum!(usize | USize, U8, U16);
convert_datum!(i8 | I8);
convert_datum!(i16 | I16, I8);
convert_datum!(i32 | I32, I16);
convert_datum!(i64 | I64, I16, I32, I8);
convert_datum!(isize | ISize, I8, I16);
convert_datum!(Vec<f64> | Vec);
#[cfg(feature = "arraydist")]
convert_datum!(DVector<f64> | DVector);
#[cfg(feature = "arraydist")]
convert_datum!(DMatrix<f64> | DMatrix);

impl From<Vec<Datum>> for Datum {
    fn from(xs: Vec<Datum>) -> Self {
        Datum::Compound(xs)
    }
}

impl From<Datum> for Vec<Datum> {
    fn from(x: Datum) -> Self {
        match x {
            Datum::Compound(xs) => xs,
            _ => panic!("invalid From type for Datum::Compound"),
        }
    }
}

impl_rvdatum!(Bernoulli, bool);
impl_rvdatum!(Beta, f64);
impl_rvdatum!(BetaBinomial, u32);
impl_rvdatum!(Binomial, u32);
impl_rvdatum!(Categorical, u32);
impl_rvdatum!(Cauchy, f64);
impl_rvdatum!(ChiSquared, f64);
impl_rvdatum!(Dirichlet, Vec<f64>);
// impl_rvdatum!(DiscreteUniform<u32>, u32);
impl_rvdatum!(Empirical, f64);
impl_rvdatum!(Exponential, f64);
impl_rvdatum!(Gamma, f64);
impl_rvdatum!(Gaussian, f64);
impl_rvdatum!(Geometric, u32);
impl_rvdatum!(Gev, f64);
impl_rvdatum!(InvChiSquared, f64);
impl_rvdatum!(InvGamma, f64);
impl_rvdatum!(InvGaussian, f64);
impl_rvdatum!(KsTwoAsymptotic, f64);
impl_rvdatum!(Kumaraswamy, f64);
impl_rvdatum!(Laplace, f64);
impl_rvdatum!(LogNormal, f64);
#[cfg(feature = "arraydist")]
impl_rvdatum!(MvGaussian, DVector<f64>);
impl_rvdatum!(NegBinomial, u32);
impl_rvdatum!(Pareto, f64);
impl_rvdatum!(Poisson, u32);
impl_rvdatum!(ScaledInvChiSquared, f64);
impl_rvdatum!(Skellam, i32);
impl_rvdatum!(StudentsT, f64);
impl_rvdatum!(SymmetricDirichlet, Vec<f64>);
impl_rvdatum!(Uniform, f64);
impl_rvdatum!(VonMises, f64);
#[cfg(feature = "arraydist")]
impl_rvdatum!(InvWishart, DMatrix<f64>);
