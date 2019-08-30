//! Gaussian Processes
use nalgebra::base::constraint::{SameNumberOfColumns, ShapeConstraint};
use nalgebra::base::storage::Storage;
use nalgebra::base::EuclideanNorm;
use nalgebra::base::Norm;
use nalgebra::{DMatrix, DVector, Dim, Matrix};
use std::f64;
use std::ops::{Index, IndexMut};
use std::fmt;


#[derive(Clone, Debug)]
pub struct CovGrad {
    slices: Vec<DMatrix<f64>>
}

impl fmt::Display for CovGrad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.slices.iter().map(|s| {
            write!(f, "{}", s)
        }).collect()
    }
}

impl CovGrad {
    pub fn new(slices: &[DMatrix<f64>]) -> Self {
        Self {
            slices: slices.to_vec()
        }
    }

    pub fn component_mul(&self, other: &DMatrix<f64>) -> Self {
        let new_slices = self.slices.iter().map(|s| s.component_mul(other)).collect();
        Self {
            slices: new_slices
        }
    }

    pub fn left_mul(&self, other: &DMatrix<f64>) -> Self {
        let new_slices = self.slices.iter().map(|s| other * s).collect();
        Self {
            slices: new_slices
        }
    }

    pub fn right_mul(&self, other: &DMatrix<f64>) -> Self {
        let new_slices = self.slices.iter().map(|s| s * other).collect();
        Self {
            slices: new_slices
        }
    }

    pub fn relative_eq(&self, other: &CovGrad, rel: f64, abs: f64) -> bool {
        assert!(self.slices.len() == other.slices.len(), "Cannot compare dissimilarly shaped CovMats");
        self.slices.iter().zip(other.slices.iter())
            .map(|(a, b)| a.relative_eq(b, rel, abs))
            .all(|x| x)
    }

    pub fn concat_cols(&self, other: &Self) -> Self {
        let slices = [self.slices.clone(), other.slices.clone()].concat();
        Self {
            slices
        }
    }

    pub fn zeros(n: usize, m: usize) -> Self {
        Self {
            slices: (0..m).map(|_| DMatrix::zeros(n, n)).collect()
        }
    }

    /// Create a new CovMat from a sequence of column slices
    pub fn from_column_slices(n: usize, m: usize, slice: &[f64]) -> Self {
        assert_eq!(n * n * m, slice.len(), "An incorrect number of points were given");
        let mut slices = Vec::with_capacity(m);
        
        for k in 0..m {
            let start = n * n * k;
            let end = start + n * n;
            slices.push(DMatrix::from_column_slice(n, n, &slice[start..end]));
        }

        Self {
            slices
        }
    }

    /// Create a new CovMat from a sequence of row slices
    pub fn from_row_slices(n: usize, m: usize, slice: &[f64]) -> Self {
        assert_eq!(n * n * m, slice.len(), "An incorrect number of points were given");
        let mut slices = Vec::with_capacity(m);
        
        for k in 0..m {
            let start = n * n * k;
            let end = start + n * n;
            slices.push(DMatrix::from_row_slice(n, n, &slice[start..end]));
        }

        Self {
            slices
        }
    }

}

impl Index<usize> for CovGrad {
    type Output = DMatrix<f64>;

    fn index(&self, k: usize) -> &Self::Output {
        assert!(k < self.slices.len(), "The requested value was outside of available values");
        &self.slices[k]
    }
}

impl IndexMut<usize> for CovGrad {
    fn index_mut(&mut self, k: usize) -> &mut Self::Output {
        assert!(k < self.slices.len(), "The requested value was outside of available values");
        &mut self.slices[k]
    }
}


impl Index<(usize, usize, usize)> for CovGrad {
    type Output = f64;

    fn index(&self, (i, j, k): (usize, usize, usize)) -> &Self::Output {
        assert!(k < self.slices.len(), "The requested value was outside of available values");
        &self.slices[k][(i, j)]
    }
}

impl IndexMut<(usize, usize, usize)> for CovGrad {
    fn index_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut Self::Output {
        assert!(k < self.slices.len(), "The requested value was outside of available values");
        &mut self.slices[k][(i, j)]
    }
}


/// Kernel Function
pub trait Kernel: std::fmt::Debug + Clone + PartialEq
{
    // Returns the covariance matrix for two equal sized vectors
    fn covariance<R1, R2, C1, C2, S1, S2>(
        &self,
        x1: &Matrix<f64, R1, C1, S1>,
        x2: &Matrix<f64, R2, C2, S2>,
    ) -> DMatrix<f64>
    where
        R1: Dim,
        R2: Dim,
        C1: Dim,
        C2: Dim,
        S1: Storage<f64, R1, C1>,
        S2: Storage<f64, R2, C2>,
        ShapeConstraint: SameNumberOfColumns<C1, C2>;
    /// Reports if the given kernel function is stationary.
    fn is_stationary(&self) -> bool;
    /// Returns the diagnal of the kernel(x, x)
    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>;

    /// Return the corresponding parameter vector
    /// The parameters here are in a log-scale
    fn parameters(&self) -> DVector<f64>;

    /// Create a new kernel of the given type from the provided parameters.
    /// The parameters here are in a log-scale
    fn from_parameters(param_vec: &DVector<f64>) -> Self;

    /// Takes a sequence of parameters and consumes only the ones it needs
    /// to create itself.
    /// The parameters here are in a log-scale
    fn consume_parameters(param_vec: &DVector<f64>) -> (Self, DVector<f64>);

    /// Covariance and Gradient with the log-scaled hyper-parameters
    fn covariance_with_gradient<R, C, S>(
        &self, 
        x: &Matrix<f64, R, C, S>,
    ) -> (DMatrix<f64>, CovGrad)
        where
            R: Dim,
            C: Dim,
            S: Storage<f64, R, C>;

    fn add<B: Kernel>(self, other: B) -> AddKernel<Self, B> {
        AddKernel::new(self, other)
    }

    fn mul<B: Kernel>(self, other: B) -> ProductKernel<Self, B> {
        ProductKernel::new(self, other)
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct ConstantKernel {
    value: f64,
}

impl ConstantKernel {
    pub fn new(value: f64) -> Self {
        Self {
            value
        }
    }
}

impl Kernel for ConstantKernel {

    fn covariance<R1, R2, C1, C2, S1, S2>(
        &self,
        x1: &Matrix<f64, R1, C1, S1>,
        x2: &Matrix<f64, R2, C2, S2>,
    ) -> DMatrix<f64>
    where
        R1: Dim,
        R2: Dim,
        C1: Dim,
        C2: Dim,
        S1: Storage<f64, R1, C1>,
        S2: Storage<f64, R2, C2>,
        ShapeConstraint: SameNumberOfColumns<C1, C2>
    {
        DMatrix::from_element(x1.nrows(), x2.nrows(), self.value)
    }

    fn is_stationary(&self) -> bool {
        true
    }

    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>
    {
        DVector::from_element(x.ncols(), self.value)
    }

    fn parameters(&self) -> DVector<f64>
    {
        DVector::from_column_slice(&[self.value.ln()])
    }

    fn from_parameters(param_vec: &DVector<f64>) -> Self {
        Self {
            value: param_vec[0].exp()
        }
    }

    fn consume_parameters(param_vec: &DVector<f64>) -> (Self, DVector<f64>) {
        assert!(param_vec.len() >= 1, "ConstantKernel requires one parameter");
        let (cur, next) = param_vec.as_slice().split_at(1);
        let ck = Self::from_parameters(&DVector::from_column_slice(cur));
        (ck, DVector::from_column_slice(next))
    }

    fn covariance_with_gradient<R, C, S>(
        &self, 
        x: &Matrix<f64, R, C, S>,
    ) -> (DMatrix<f64>, CovGrad)
        where
            R: Dim,
            C: Dim,
            S: Storage<f64, R, C>
    {
        let cov = self.covariance(x, x);
        let grad = CovGrad::new(&[DMatrix::from_element(x.nrows(), x.nrows(), self.value)]);
        (cov, grad)
    }
}

/// Kernel representing the sum of two other kernels
#[derive(Debug, Clone, PartialEq)]
pub struct AddKernel<A, B>
    where
        A: Kernel,
        B: Kernel,
{
    a: A,
    b: B,
}

impl<A, B> AddKernel<A, B>
    where
        A: Kernel,
        B: Kernel,
{
    /// Construct a new Kernel from two other Kernels
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> Kernel for AddKernel<A, B>
    where
        A: Kernel,
        B: Kernel,
{
    fn is_stationary(&self) -> bool {
        self.a.is_stationary() && self.b.is_stationary()
    }

    fn covariance<R1, R2, C1, C2, S1, S2>(
        &self,
        x1: &Matrix<f64, R1, C1, S1>,
        x2: &Matrix<f64, R2, C2, S2>,
    ) -> DMatrix<f64>
    where
        R1: Dim,
        R2: Dim,
        C1: Dim,
        C2: Dim,
        S1: Storage<f64, R1, C1>,
        S2: Storage<f64, R2, C2>,
        ShapeConstraint: SameNumberOfColumns<C1, C2>

    {
        self.a.covariance(x1, x2) + self.b.covariance(x1, x2)
    }

    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C> 
    {
        let a = self.a.diag(x);
        let b = self.b.diag(x);
        a.zip_map(&b, |y1, y2| y1 + y2)
    }

    fn parameters(&self) -> DVector<f64> {
        let a = self.a.parameters();
        let b = self.b.parameters();

        DVector::from_column_slice(&[a.as_slice(), b.as_slice()].concat())
    }

    fn from_parameters(param_vec: &DVector<f64>) -> Self {
        let (a, b_params) = A::consume_parameters(param_vec);
        let b = B::from_parameters(&b_params);
        Self::new(a, b)
    }

    fn consume_parameters(param_vec: &DVector<f64>) -> (Self, DVector<f64>) {
        let (a, b_params) = A::consume_parameters(param_vec);
        let (b, left) = B::consume_parameters(&b_params);
        (Self::new(a, b), left)
    }

    fn covariance_with_gradient<R, C, S>(
        &self, 
        x: &Matrix<f64, R, C, S>,
    ) -> (DMatrix<f64>, CovGrad)
        where
            R: Dim,
            C: Dim,
            S: Storage<f64, R, C>
    {
        let (cov_a, grad_a) = self.a.covariance_with_gradient(x);
        let (cov_b, grad_b) = self.b.covariance_with_gradient(x);

        let new_cov = cov_a + cov_b;

        let new_grad = grad_a.concat_cols(&grad_b);
        (new_cov, new_grad)
    }
}

/// Kernel representing the product of two other kernels
#[derive(Clone, Debug, PartialEq)]
pub struct ProductKernel<A, B>
    where
        A: Kernel,
        B: Kernel,
{
    a: A,
    b: B,
}

impl<A, B> ProductKernel<A, B>
    where
        A: Kernel,
        B: Kernel,
{
    /// Construct a new Kernel from two other Kernels
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> Kernel for ProductKernel<A, B>
    where
        A: Kernel,
        B: Kernel,
{
    fn covariance<R1, R2, C1, C2, S1, S2>(
        &self,
        x1: &Matrix<f64, R1, C1, S1>,
        x2: &Matrix<f64, R2, C2, S2>,
    ) -> DMatrix<f64>
    where
        R1: Dim,
        R2: Dim,
        C1: Dim,
        C2: Dim,
        S1: Storage<f64, R1, C1>,
        S2: Storage<f64, R2, C2>,
        ShapeConstraint: SameNumberOfColumns<C1, C2>
    {
        let cov_a = self.a.covariance(x1, x2);
        let cov_b = self.b.covariance(x1, x2);
        cov_a.component_mul(&cov_b)
    }

    fn is_stationary(&self) -> bool {
        self.a.is_stationary() && self.b.is_stationary()
    }

    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C> 
    {
        let a = self.a.diag(x);
        let b = self.b.diag(x);
        a.zip_map(&b, |y1, y2| y1 * y2)
    }

    fn parameters(&self) -> DVector<f64> {
        let a = self.a.parameters();
        let b = self.b.parameters();

        DVector::from_column_slice(&[a.as_slice(), b.as_slice()].concat())
    }

    fn from_parameters(param_vec: &DVector<f64>) -> Self {
        let (a, b_params) = A::consume_parameters(param_vec);
        let b = B::from_parameters(&b_params);
        Self::new(a, b)
    }

    fn consume_parameters(param_vec: &DVector<f64>) -> (Self, DVector<f64>) {
        let (a, b_params) = A::consume_parameters(param_vec);
        let (b, left) = B::consume_parameters(&b_params);
        (Self::new(a, b), left)
    }

    fn covariance_with_gradient<R, C, S>(
        &self, 
        x: &Matrix<f64, R, C, S>,
    ) -> (DMatrix<f64>, CovGrad)
        where
            R: Dim,
            C: Dim,
            S: Storage<f64, R, C>
    {
        let (cov_a, grad_a) = self.a.covariance_with_gradient(x);
        let (cov_b, grad_b) = self.b.covariance_with_gradient(x);

        let new_cov = cov_a.component_mul(&cov_b);
        let new_grad = grad_a.component_mul(&cov_b)
            .concat_cols(
                &grad_b.component_mul(&cov_a)
            );

        (new_cov, new_grad)
    }
}

/// Radial-basis function (RBF) kernel
/// The distance metric here is L2 (Euclidean).
///
/// ```math
///     K(\mathbf{x}, \mathbf{x'}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x'}\|^2}{2\sigma^2}\right)
/// ```
///
/// # Parameters
/// * `l` - Length scale.
///
#[derive(Clone, Debug, PartialEq)]
pub struct RBFKernel {
    l: f64,
}

impl RBFKernel {
    pub fn new(l: f64) -> Self {
        Self { l }
    }
}

impl Kernel for RBFKernel {
    fn covariance<R1, R2, C1, C2, S1, S2>(
        &self,
        x1: &Matrix<f64, R1, C1, S1>,
        x2: &Matrix<f64, R2, C2, S2>,
    ) -> DMatrix<f64>
    where
        R1: Dim,
        R2: Dim,
        C1: Dim,
        C2: Dim,
        S1: Storage<f64, R1, C1>,
        S2: Storage<f64, R2, C2>,
        ShapeConstraint: SameNumberOfColumns<C1, C2>,
    {
        let m = x1.nrows();
        let n = x2.nrows();

        let mut dm: DMatrix<f64> = DMatrix::zeros(m, n);

        let metric = EuclideanNorm {};
        for i in 0..m {
            for j in 0..n {
                let d = metric.metric_distance(&x1.row(i), &x2.row(j)) / self.l;
                dm[(i, j)] = d * d;
            }
        }

        dm.map(|e| (-0.5 * e).exp())
    }

    fn is_stationary(&self) -> bool {
        true
    }

    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>,
    {
        DVector::repeat(x.nrows(), 1.0)
    }

    fn parameters(&self) -> DVector<f64> {
        DVector::from_column_slice(&[self.l.ln()])
    }

    fn consume_parameters(param_vec: &DVector<f64>) -> (Self, DVector<f64>) {
        assert!(param_vec.nrows() >= 1, "RBFKernel requires one parameters");
        let (cur, next) = param_vec.as_slice().split_at(1);
        let ck = Self::from_parameters(&DVector::from_column_slice(cur));
        (ck, DVector::from_column_slice(next))
    }

    fn from_parameters(param_vec: &DVector<f64>) -> Self {
        assert_eq!(param_vec.len(), 1, "The parameter vector for RBFKernel must be of length 1");
        Self::new(param_vec[0].exp())
    }

    fn covariance_with_gradient<R, C, S>(
        &self, 
        x: &Matrix<f64, R, C, S>,
    ) -> (DMatrix<f64>, CovGrad)
        where
            R: Dim,
            C: Dim,
            S: Storage<f64, R, C>,
    {
        let n = x.nrows();

        let metric = EuclideanNorm {};
        let mut dm = DMatrix::zeros(n, n);
        let mut grad = CovGrad::zeros(n, 1);

        for i in 0..n {
            for j in 0..i {
                // Save covariance
                let d = metric.metric_distance(&x.row(i), &x.row(j));
                let l2 = self.l.powi(2);
                let d2 = d * d;
                let exp_d2 = (-d2 / (2.0 * l2)).exp();
                let cov_ij = exp_d2;

                dm[(i, j)] = cov_ij;
                dm[(j, i)] = cov_ij;

                // Save gradient
                let dc_dl = d2 * cov_ij / l2;
                grad[(i, j, 0)] = dc_dl;
                grad[(j, i, 0)] = dc_dl;
            }
            dm[(i, i)] = 1.0;
        }

        (dm, grad)
    }
}

/// White Noise Kernel
#[derive(Clone, Debug, PartialEq)]
pub struct WhiteKernel {
    /// Level of the noise
    noise_level: f64,
}

impl WhiteKernel {
    pub fn new(noise_level: f64) -> Self {
        Self {
            noise_level
        }
    }
}

impl Kernel for WhiteKernel {
    fn covariance<R1, R2, C1, C2, S1, S2>(
        &self,
        x1: &Matrix<f64, R1, C1, S1>,
        _x2: &Matrix<f64, R2, C2, S2>,
    ) -> DMatrix<f64>
    where
        R1: Dim,
        R2: Dim,
        C1: Dim,
        C2: Dim,
        S1: Storage<f64, R1, C1>,
        S2: Storage<f64, R2, C2>,
        ShapeConstraint: SameNumberOfColumns<C1, C2>
    {
        let n = x1.nrows();
        DMatrix::from_diagonal_element(n, n, self.noise_level)
    }

    fn is_stationary(&self) -> bool {
        true
    }

    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>
    {
        let n = x.nrows();
        DVector::from_element(n, self.noise_level)
    }

    fn parameters(&self) -> DVector<f64> {
        DVector::from_column_slice(&[self.noise_level.ln()])
    }

    fn from_parameters(param_vec: &DVector<f64>) -> Self {
        assert_eq!(param_vec.nrows(), 1, "Only one parameter expected");
        Self::new(param_vec[0].exp())
    }

    fn consume_parameters(param_vec: &DVector<f64>) -> (Self, DVector<f64>) {
        assert!(param_vec.nrows() >= 1, "WhiteKernel requires one parameters");
        let (cur, next) = param_vec.as_slice().split_at(1);
        let ck = Self::from_parameters(&DVector::from_column_slice(cur));
        (ck, DVector::from_column_slice(next))
    }

    fn covariance_with_gradient<R, C, S>(
        &self, 
        x: &Matrix<f64, R, C, S>,
    ) -> (DMatrix<f64>, CovGrad)
        where
            R: Dim,
            C: Dim,
            S: Storage<f64, R, C>
    {
        let cov = self.covariance(x, x);
        let grad = CovGrad::new(&[DMatrix::from_diagonal_element(x.nrows(), x.nrows(), self.noise_level)]);
        (cov, grad)
    }
}

/// Rational Quadratic Kernel
///
/// # Parameters
/// `scale` -- Length scale
/// `mixture` -- Mixture Scale
#[derive(Clone, Debug, PartialEq)]
pub struct RationalQuadratic {
    scale: f64,
    mixture: f64,
}

impl RationalQuadratic {
    pub fn new(scale: f64, mixture: f64) -> Self {
        Self {
            scale,
            mixture
        }
    }
}

impl Kernel for RationalQuadratic
{
    fn covariance<R1, R2, C1, C2, S1, S2>(
        &self,
        x1: &Matrix<f64, R1, C1, S1>,
        x2: &Matrix<f64, R2, C2, S2>,
    ) -> DMatrix<f64>
    where
        R1: Dim,
        R2: Dim,
        C1: Dim,
        C2: Dim,
        S1: Storage<f64, R1, C1>,
        S2: Storage<f64, R2, C2>,
        ShapeConstraint: SameNumberOfColumns<C1, C2>
    {
        let metric = EuclideanNorm {};
        let d = 2.0 * self.scale * self.scale * self.mixture;
        DMatrix::from_fn(x1.nrows(), x2.nrows(), |i, j| {
            let t = metric.metric_distance(&x1.row(i), &x2.row(j)).powi(2) / d;
            (1.0 + t).powf(-self.mixture)
        })
    }

    fn is_stationary(&self) -> bool {
        true
    }

    fn diag<R, C, S>(&self, x: &Matrix<f64, R, C, S>) -> DVector<f64>
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C> 
    {
        DVector::repeat(x.len(), 1.0)
    }

    fn parameters(&self) -> DVector<f64> {
        DVector::from_column_slice(&[self.scale.ln(), self.mixture.ln()])
    }

    fn from_parameters(param_vec: &DVector<f64>) -> Self {
        assert_eq!(param_vec.nrows(), 2, "");
        let scale = param_vec[0].exp();
        let mixture = param_vec[1].exp();
        Self::new(scale, mixture)
    }

    fn consume_parameters(param_vec: &DVector<f64>) -> (Self, DVector<f64>) {
        assert!(param_vec.nrows() >= 2, "RationalQuadratic requires two parameters");
        let (cur, next) = param_vec.as_slice().split_at(2);
        let ck = Self::from_parameters(&DVector::from_column_slice(cur));
        (ck, DVector::from_column_slice(next))
    }

    fn covariance_with_gradient<R, C, S>(
        &self, 
        x: &Matrix<f64, R, C, S>,
    ) -> (DMatrix<f64>, CovGrad)
        where
            R: Dim,
            C: Dim,
            S: Storage<f64, R, C>
    {
        let n = x.nrows();
        let mut cov = DMatrix::zeros(n, n);
        let mut grad = CovGrad::zeros(n, 2);
        let d = 2.0 * self.mixture * self.scale.powi(2);
        let metric = EuclideanNorm {};
        for i in 0..n {
            // off-diagnols
            for j in 0..i {
                let d2 = metric.metric_distance(&x.row(i), &x.row(j)).powi(2);
                let temp = d2 / d;
                let base = 1.0 + temp;
                let k = base.powf(-self.mixture);
                cov[(i, j)] = k;
                cov[(j, i)] = k;

                let dk_dl = d2 * k / (self.scale.powi(2) * base);
                let dk_da = k * (-self.mixture * base.ln() + d2 / (2.0 * self.scale.powi(2) * base));

                grad[(i, j, 0)] = dk_dl;
                grad[(j, i, 0)] = dk_dl;
                grad[(j, i, 1)] = dk_da;
                grad[(i, j, 1)] = dk_da; 
            }
            // diag
            cov[(i, i)] = 1.0;
        }
        (cov, grad)
    }
}

/// Exp Sine^2 Kernel
/// k(x_i, x_j) = exp(-2 (sin(pi / periodicity * d(x_i, x_j)) / length_scale) ^ 2)
#[derive(Clone, Debug, PartialEq)]
pub struct ExpSineSquaredKernel {
    periodicity: f64,
    length_scale: f64,
}

impl ExpSineSquaredKernel {
    /// Create a new ExpSineSquaredKernel
    pub fn new(periodicity: f64, length_scale: f64) -> Self {
        Self {
            periodicity,
            length_scale,
        }
    }
}

/*
impl Kernel for ExpSineSquaredKernel
{
    fn covariance(&self, x1: &DMatrix<f64>, x2: &DMatrix<f64>) -> DMatrix<f64> {
        assert!(x1.len() == x2.len());
        let diff = x1 - x2;
        let diff_t = diff.clone().transpose();
        let sinarg = f64::consts::PI * (diff * &diff_t) / self.periodicity;
        sinarg.map(|e| (2.0 * (e.sin().powi(2)) / (self.length_scale * self.length_scale)).exp())
    }

    fn is_stationary(&self) -> bool {
        true
    }

    fn diag(&self, x: &DMatrix<f64>) -> DVector<f64> {
        DVector::repeat(x.len(), 1.0)
    }

    fn parameters(&self) -> DVector<f64> {
        DVector::from_column_slice(&[self.periodicity, self.length_scale])
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbf_simple() {
        let kernel = RBFKernel::new(1.0);
        assert::close(kernel.parameters()[0], 0.0, 1E-10);
        assert_eq!(kernel,
            RBFKernel::from_parameters(&DVector::from_column_slice(&[0.0]))
        );
        assert!(kernel.is_stationary());
    }

    #[test]
    fn rbf_1d() {
        let xs = DVector::from_column_slice(&[0.0, 1.0, 2.0, 3.0]);
        let kernel = RBFKernel::new(1.0);

        let cov = kernel.covariance(&xs, &xs);
        let expected_cov = DMatrix::from_column_slice(
            4,
            4,
            &[
                1., 0.60653066, 0.13533528, 0.011109, 0.60653066, 1.,
                0.60653066, 0.13533528, 0.13533528, 0.60653066, 1., 0.60653066,
                0.011109, 0.13533528, 0.60653066, 1.,
            ],
        );

        assert!(expected_cov.relative_eq(&cov, 1E-8, 1E-8));
        let expected_diag = DVector::from_column_slice(&[1., 1., 1., 1.]);
        assert_eq!(kernel.diag(&xs), expected_diag);
    }

    #[test]
    fn rbf_2d() {
        use nalgebra::Matrix4x2;

        let kernel = RBFKernel::new(1.0);
        let xs =
            Matrix4x2::from_column_slice(&[0., 1., 2., 3., 4., 5., 6., 7.]);
        let expected_cov = DMatrix::from_column_slice(
            4,
            4,
            &[
                1.00000000e+00,
                3.67879441e-01,
                1.83156389e-02,
                1.23409804e-04,
                3.67879441e-01,
                1.00000000e+00,
                3.67879441e-01,
                1.83156389e-02,
                1.83156389e-02,
                3.67879441e-01,
                1.00000000e+00,
                3.67879441e-01,
                1.23409804e-04,
                1.83156389e-02,
                3.67879441e-01,
                1.00000000e+00,
            ],
        );

        let cov = kernel.covariance(&xs, &xs);
        assert!(expected_cov.relative_eq(&cov, 1E-8, 1E-8));
    }

    #[test]
    fn rbf_different_sizes() {
        use nalgebra::Matrix5x1;
        let kernel = RBFKernel::new(1.0);

        let x1 = Matrix5x1::from_column_slice(&[-4., -3., -2., -1., 1.]);
        let x2 = DMatrix::from_column_slice(
            10,
            1,
            &[-5., -4., -3., -2., -1., 0., 1., 2., 3., 4.],
        );

        let cov = kernel.covariance(&x1, &x2);
        let expected_cov = DMatrix::from_row_slice(
            5,
            10,
            &[
                6.06530660e-01,
                1.00000000e+00,
                6.06530660e-01,
                1.35335283e-01,
                1.11089965e-02,
                3.35462628e-04,
                3.72665317e-06,
                1.52299797e-08,
                2.28973485e-11,
                1.26641655e-14,
                1.35335283e-01,
                6.06530660e-01,
                1.00000000e+00,
                6.06530660e-01,
                1.35335283e-01,
                1.11089965e-02,
                3.35462628e-04,
                3.72665317e-06,
                1.52299797e-08,
                2.28973485e-11,
                1.11089965e-02,
                1.35335283e-01,
                6.06530660e-01,
                1.00000000e+00,
                6.06530660e-01,
                1.35335283e-01,
                1.11089965e-02,
                3.35462628e-04,
                3.72665317e-06,
                1.52299797e-08,
                3.35462628e-04,
                1.11089965e-02,
                1.35335283e-01,
                6.06530660e-01,
                1.00000000e+00,
                6.06530660e-01,
                1.35335283e-01,
                1.11089965e-02,
                3.35462628e-04,
                3.72665317e-06,
                1.52299797e-08,
                3.72665317e-06,
                3.35462628e-04,
                1.11089965e-02,
                1.35335283e-01,
                6.06530660e-01,
                1.00000000e+00,
                6.06530660e-01,
                1.35335283e-01,
                1.11089965e-02,
            ],
        );
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
    }

    #[test]
    fn rbf_gradient() {
        const E: f64 = std::f64::consts::E;
        let x = DMatrix::from_row_slice(2, 2, &[
            1.0, 2.0,
            3.0, 4.0
        ]);
        let r = RBFKernel::new(1.0);
        let (cov, grad) = r.covariance_with_gradient(&x);

        let expected_cov = DMatrix::from_row_slice(2, 2, &[
          1.0, 1.0 / E.powi(4),
          1.0 / E.powi(4) , 1.0,
        ]);

        let expected_grad = CovGrad::from_column_slices(2, 1, &[
            0.0,
            8.0 / E.powi(4),
            8.0 / E.powi(4),
            0.0,
        ]);
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));

        let r = RBFKernel::new(4.0);
        let (cov, grad) = r.covariance_with_gradient(&x);

        let expected_cov = DMatrix::from_row_slice(2, 2, &[
          1.0, 1.0 / E.powf(1.0 / 4.0),
          1.0 / E.powf(1.0 / 4.0), 1.0,
        ]);

        let expected_grad = CovGrad::from_column_slices(2, 1, &[
            0.0,
            (1.0 / 2.0) / E.powf(0.25),
            (1.0 / 2.0) / E.powf(0.25),
            0.0,
        ]);

        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));
    }

    #[test]
    fn constant_kernel() {
        let kernel = ConstantKernel::new(3.0);
        assert::close(kernel.parameters()[0], 3.0_f64.ln(), 1E-10);

        assert!(
            kernel.parameters().relative_eq(
                &ConstantKernel::from_parameters(
                    &DVector::from_column_slice(&[3.0_f64.ln()])
                ).parameters(),
                1E-10,
                1E-10
            )
        );

        let x = DMatrix::from_column_slice(2, 2, &[1.0, 3.0, 2.0, 4.0]);
        let y = DMatrix::from_column_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);

        let (cov, grad) = kernel.covariance_with_gradient(&x);

        let expected_cov = DMatrix::from_row_slice(2, 2, &[
            3.0, 3.0,
            3.0, 3.0
        ]);

        let expected_grad = CovGrad::from_row_slices(2, 1, &[
            3.0, 3.0,
            3.0, 3.0,
        ]);
        
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));

        let cov = kernel.covariance(&x, &y);
        let expected_cov = DMatrix::from_row_slice(2, 2, &[
            3.0, 3.0,
            3.0, 3.0
        ]);
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
    }

    #[test]
    fn rational_quadratic() {
        let kernel = RationalQuadratic::new(3.0, 5.0);
        assert::close(kernel.parameters()[0], 3.0_f64.ln(), 1E-10);
        assert::close(kernel.parameters()[1], 5.0_f64.ln(), 1E-10);
        
        assert!(
            kernel.parameters().relative_eq(
                &RationalQuadratic::from_parameters(
                    &DVector::from_column_slice(&[3.0_f64.ln(), 5.0_f64.ln()])
                ).parameters(),
                1E-10,
                1E-10
            )
        );

        let x = DMatrix::from_row_slice(2, 2, &[
            1.0, 2.0,
            3.0, 4.0
        ]);
        let y = DMatrix::from_row_slice(2, 2, &[
            5.0, 7.0,
            6.0, 8.0
        ]);

        let cov = kernel.covariance(&x, &y);
        let expected_cov = DMatrix::from_row_slice(2, 2, &[
            5_904_900_000.0 / 38_579_489_651.0, 5_904_900_000.0 / 78_502_725_751.0,
            5_904_900_000.0 / 11_592_740_742.0, 1_889_568.0 / 6_436_343.0
        ]);
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));

        let (cov, grad) = kernel.covariance_with_gradient(&x);

        let expected_cov = DMatrix::from_row_slice(2, 2, &[
           1.0,                       184528125.0 / 282475249.0,
           184528125.0 / 282475249.0, 1.0
        ]);

        let eg_l = 0.53326868;
        let eg_a = -0.01151411;
        let expected_grad = CovGrad::from_row_slices(2, 2, &[
            0.0, eg_l,
            eg_l, 0.0,
            0.0, eg_a,
            eg_a, 0.0
        ]);
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));
    }


    #[test]
    fn white_kernel() {
        let kernel = WhiteKernel::new(3.0);

        assert::close(kernel.parameters()[0], 3.0_f64.ln(), 1E-10);
        
        assert!(
            kernel.parameters().relative_eq(
                &WhiteKernel::from_parameters(
                    &DVector::from_column_slice(&[3.0_f64.ln()])
                ).parameters(),
                1E-10,
                1E-10
            )
        );


        let x = DMatrix::from_row_slice(2, 2, &[
            1.0, 2.0,
            3.0, 4.0
        ]);
        let y = DMatrix::from_row_slice(2, 2, &[
            5.0, 7.0,
            6.0, 8.0
        ]);

        let cov = kernel.covariance(&x, &y);
        let expected_cov = DMatrix::from_row_slice(2, 2, &[
            3.0, 0.0,
            0.0, 3.0,
        ]);
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));

        let (cov, grad) = kernel.covariance_with_gradient(&x);

        let expected_cov = DMatrix::from_row_slice(2, 2, &[
            3.0, 0.0,
            0.0, 3.0,
        ]);

        let expected_grad = CovGrad::from_row_slices(2, 1, &[
            3.0, 0.0,
            0.0, 3.0,
        ]);
        assert!(cov.relative_eq(&expected_cov, 1E-8, 1E-8));
        assert!(grad.relative_eq(&expected_grad, 1E-8, 1E-8));
    }

    #[test]
    fn product_kernel() {
        let kernel = ProductKernel::new(
            ConstantKernel::new(3.0),
            WhiteKernel::new(1.0)
        );
        let x = DMatrix::from_row_slice(2, 2, &[
            1.0, 2.0,
            3.0, 4.0
        ]);

        let expected_cov = DMatrix::from_row_slice(2, 2, &[
            3.0, 0.0,
            0.0, 3.0
        ]);

        let expected_grad = CovGrad::new(&[
            DMatrix::from_row_slice(2, 2, &[
                3.0, 0.0,
                0.0, 3.0
            ]),
            DMatrix::from_row_slice(2, 2, &[
                3.0, 0.0,
                0.0, 3.0
            ]),
        ]);

        let (cov, grad) = kernel.covariance_with_gradient(&x);
        println!("cov = {}, grad = {}", cov, grad);

        assert!(cov.relative_eq(&expected_cov, 1E-7, 1E-7));
        assert!(grad.relative_eq(&expected_grad, 1E-7, 1E-7));
    }
}
