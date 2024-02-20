use rand::Rng;

/// Random variable
///
/// Contains the minimal functionality that a random object must have to be
/// useful: a function defining the un-normalized density/mass at a point,
/// and functions to draw samples from the distribution.
pub trait Process<SampleSpace, ObservationSpace> {
    /// Probability function
    ///
    /// # Example
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::*;
    ///
    /// let g = Gaussian::standard();
    /// assert!(g.f(&0.0_f64) > g.f(&0.1_f64));
    /// assert!(g.f(&0.0_f64) > g.f(&-0.1_f64));
    /// ```
    fn f(&self, x: &ObservationSpace) -> f64 {
        self.ln_f(x).exp()
    }

    /// Probability function
    ///
    /// # Example
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::*;
    ///
    /// let g = Gaussian::standard();
    /// assert!(g.ln_f(&0.0_f64) > g.ln_f(&0.1_f64));
    /// assert!(g.ln_f(&0.0_f64) > g.ln_f(&-0.1_f64));
    /// ```
    fn ln_f(&self, x: &ObservationSpace) -> f64;

    /// Single draw from the `Rv`
    ///
    /// # Example
    ///
    /// Flip a coin
    ///
    /// ```
    /// use rv::dist::Bernoulli;
    /// use rv::traits::*;
    ///
    /// let b = Bernoulli::uniform();
    /// let mut rng = rand::thread_rng();
    /// let x: bool = b.draw(&mut rng); // could be true, could be false.
    /// ```
    fn draw<R: Rng>(&self, rng: &mut R) -> SampleSpace;

    /// Multiple draws of the `Rv`
    ///
    /// # Example
    ///
    /// Flip a lot of coins
    ///
    /// ```
    /// use rv::dist::Bernoulli;
    /// use rv::traits::*;
    ///
    /// let b = Bernoulli::uniform();
    /// let mut rng = rand::thread_rng();
    /// let xs: Vec<bool> = b.sample(22, &mut rng);
    ///
    /// assert_eq!(xs.len(), 22);
    /// ```
    ///
    /// Estimate Gaussian mean
    ///
    /// ```
    /// use rv::dist::Gaussian;
    /// use rv::traits::*;
    ///
    /// let gauss = Gaussian::standard();
    /// let mut rng = rand::thread_rng();
    /// let xs: Vec<f64> = gauss.sample(100_000, &mut rng);
    ///
    /// assert::close(xs.iter().sum::<f64>()/100_000.0, 0.0, 1e-2);
    /// ```
    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<SampleSpace> {
        (0..n).map(|_| self.draw(&mut rng)).collect()
    }

    /// Create a never-ending iterator of samples
    ///
    /// # Example
    ///
    /// Estimate the mean of a Gamma distribution
    ///
    /// ```
    /// use rv::traits::*;
    /// use rv::dist::Gamma;
    ///
    /// let mut rng = rand::thread_rng();
    ///
    /// let gamma = Gamma::new(2.0, 1.0).unwrap();
    ///
    /// let n = 1_000_000_usize;
    /// let mean = <Gamma as Rv<f64>>::sample_stream(&gamma, &mut rng)
    ///     .take(n)
    ///     .sum::<f64>() / n as f64;;
    ///
    /// assert::close(mean, 2.0, 1e-2);
    /// ```
    fn sample_stream<'r, R: Rng>(
        &'r self,
        mut rng: &'r mut R,
    ) -> Box<dyn Iterator<Item = SampleSpace> + 'r> {
        Box::new(std::iter::repeat_with(move || self.draw(&mut rng)))
    }
}

// impl<T, X> Process<X, X> for T
// where
//     T: Rv<X>,
// {
//     fn f(&self, x: &X) -> f64 {
//         self.f(x)
//     }

//     fn ln_f(&self, x: &X) -> f64 {
//         self.ln_f(x)
//     }

//     fn draw<R: Rng>(&self, rng: &mut R) -> X {
//         self.draw(rng)
//     }
// }
