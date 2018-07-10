


/// Example:
/// ```
/// let nig = NormalInverseGamma::default();
///
/// let cpnt = Component {
///     distribution: Gaussian::standard(),
///     prior: &nig,
///     obs: GaussianSuffStats::new(),
/// }
///
/// cpnt.observe(2.0);
///
/// println!("{}", cpnt.obs())  // { n: 1, sumx: 2.0, sumxsq: 4.0 }
/// ```
pub struct Component<T, D, Pr, 'pr> {
   distribution:  D,
   prior: &'pr Pr,
   obs: Vec<T>
}


