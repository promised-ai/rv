use nalgebra::base::constraint::{
    SameNumberOfColumns, SameNumberOfRows, ShapeConstraint,
};
use nalgebra::base::storage::Storage;
use nalgebra::base::Norm;
use nalgebra::{ComplexField, Dim, Matrix};
use num::Zero;

pub const E2METRIC: Euclidean2Norm = Euclidean2Norm {};

pub struct Euclidean2Norm;

impl<N: ComplexField> Norm<N> for Euclidean2Norm {
    #[inline]
    fn norm<R, C, S>(&self, m: &Matrix<N, R, C, S>) -> N::RealField
    where
        R: Dim,
        C: Dim,
        S: Storage<N, R, C>,
    {
        m.norm_squared()
    }
    #[inline]
    fn metric_distance<R1, C1, S1, R2, C2, S2>(
        &self,
        m1: &Matrix<N, R1, C1, S1>,
        m2: &Matrix<N, R2, C2, S2>,
    ) -> N::RealField
    where
        R1: Dim,
        C1: Dim,
        S1: Storage<N, R1, C1>,
        R2: Dim,
        C2: Dim,
        S2: Storage<N, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
    {
        m1.zip_fold(m2, N::RealField::zero(), |acc, a, b| {
            let diff = a - b;
            acc + diff.modulus_squared()
        })
    }
}

#[inline]
pub fn e2_norm<N, R1, C1, S1, R2, C2, S2>(
    m1: &Matrix<N, R1, C1, S1>,
    m2: &Matrix<N, R2, C2, S2>,
    scale: N,
) -> N::RealField
where
    N: ComplexField,
    R1: Dim,
    C1: Dim,
    S1: Storage<N, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: Storage<N, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    m1.zip_fold(m2, N::RealField::zero(), |acc, a, b| {
        let diff = (a - b) / scale.clone();
        acc + diff.modulus_squared()
    })
}
