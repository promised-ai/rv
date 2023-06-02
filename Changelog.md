# Changelog

## 0.16.0
- Fix bug in `InvGaussianSuffStat`
- Add `ln_f_stat` method to `HasSuffStat` trait. This method computes the log
    likelihood of the data represented by the statistic.
- Un-implement `HasSuffStat<f64>` for `InverseGamma`
- Add `InvGammaSuffStat`
- Add `BetaSuffStat`
- Cache ln(weights) for `Mixture` to improve `ln_f` computation performance

## 0.15.1
- Fix bug in `Empirical::draw` that causes an `index out of bounds error` when
    the last last bin is drawn.

## 0.15.0
- All structs and enums use `snake_case` naming.
- Dependency updates (thanks @bsull).
- Entropy impl for Bernoulli Mixuture.
- Corrected NormalInverseGamma Posterior.
- Corrected NormalInverseWishart Posterior.
- Added `Kernel::reparameterize`.
- Improve CI.

## 0.14.4
- re-export nalgebra (flag `arraydist`)
- add `reparameterize` method to `process::gaussian::Kernel`, which will
    replace `from_parmeters` in 0.15.0
- Implement `Error` for `NormalInvGammaError` and `NormalInvChiSquaredError`

## 0.14.3
- Remove dependency on peroxide

## 0.14.2
- Fix some incorrect documentation
- Pass `--all-features` to docs.rs builds

## 0.14.1
- Fix bug in computation of entropies for continuous mixture models 

## 0.14.0
- Fix typo: `CategoricalError::EmptyWights` to `CategoricalError::EmptyWeights`
- Allow `Categorical::from_ln_weights` to accept zero (-Inf) weights
- Update `lru` crate to 0.7 due to security vulnerability in 0.6.6

## 0.13.1
- When using `dist::Gamma` to draw a `dist::Poisson`, the Poisson rate
    parameter is shifted to `f64::EPSILON` in the event of underflow.

## 0.13.0
- Clippy lints
- Use `&'a [T]` instead of `&'a Vec<T>` in `DataOrSuffStat`
- Use `&[T]` instead of `&Vec<T>` in `GewekeTester`
- Renamed `BesselIvError` variants
- Implement `Rv<usize>` for `Poisson`
- Additional caching of expensive computations
- Estimate Gaussian mixture entropy with Gauss–Legendre quadrature for a 10x
    speedup.
- Place `ndarray` dependent distributions behind `arraydist` flag.
- Fix bug the caused weights to be computed incorrectly when using
    `Mixture::combine` where one or more of the inputs was an empty mixture

## 0.12.0
- Added Inverse X^2 distribution
- Added Scaled Inverse X^2 distribution (`InvChiSquared`)
- Added Inverse Gaussian distribution
- Added Inverse Gaussian sufficient statistic
- Added Normal Inverse X^2 (`NormalInvChiSquared`) distribution as prior for
    Gaussian
- Added Inverse X^2 distribution (`InvChiSquared`)
- Implemented `From` instead of `Into` for sufficient statistic converters to
    allow the more explicit/ergonomic `From` conversions.
- Improved some error messages in `NormalGamma`
- Caching of a normalizing constant in `BetaBinomial`
- Lots of testing verifying proper conjugate prior behavior
- Added Gaussian prior geweke tests and example
- More robust dpgmm example -- can use different priors
- Version updates
- Misc styling and warning fixes

## 0.11.1
- Added Normal Inverse Gamma (`NormalInvGamma`) distribution as prior for
    Gaussian

## 0.11.0
- Added `_with_cache` variants to `ConjugatePrior` `ln_m` and `ln_pp` methods
    for use cases where these methods are called many times in different data
    without changing the underlying distribution. This also adds two more
    associated types to the trait (`LnMCache` and `LnPpCache`), and one method
    each to compute the caches.
- Remove all references to `serde_derive` crate
- Gaussian process improvements including new kernels

## 0.10.5
- Added Gaussian processes

## 0.10.4
- Updated math in `NormalGamma` and `GaussianSuffStat` to reduce rounding errors

## 0.10.2
- Categorical datum converters work like standard usize-to-boolean casting where
    0 is false, and anything greater than 0 is true.

## 0.10.1
- Add `from_parts_unchecked` method for constructing sufficient statistic since
    there was no way of manipulating statistic's fields or creating them
    manually.

## 0.10.0
- Implement Poisson Mode
- Store `GaussianSuffStat` in a more numerically stable way. The extra numerical
    stability comes at the cost of slower `observe` and `forget`. Serialization
    on `GaussianSuffStat` has changed.

## 0.9.3
- Entropy for Gaussian mixtures more robust to models with dispersed
    low-variance components
- Categorical::new_unchecked is now public

## 0.9.2
- Benchmarks
- implement `Entropy` and `KlDivergence` for `Poisson`
- Implement `QuadBounds` and `Entropy` for `Mixture<Poisson>`
- Implement `Mean` for mixtures of distributions that implement `Mean<f64>` or
    `Mean<f32>`
- `Rv` trait has `sample_stream` method that produces a never-ending iterator of
    random numbers.

## 0.9.1
- Remove printlns

## 0.9.0
- Refactor errors to provide more information
- Errors implement the `std::error::Error` trait
- Fixed a bug with `rv::dist::Mixture` that would compute means incorrectly
- `MvGaussian::dims` renamed to `MvGaussian::ndims`
- More aggressive caching of log likelihood terms
- Cached terms no longer unnecessarily computed on `Clone`
- Remove dependency on `getset`
- Setters do input validation unless prefixed by `_unchecked`.
- feature `serde_support` renamed to `serde1`

## 0.8.3
- Added Negative Binomial distribution
- Added Skellam Distribution
- Add `try_quad` function for numerical integration of functions that return
    `Result<f64, E>`.

## 0.8.2
- Remove dependency on `quadrature` crate in favor of hand-rolled adaptive
    Simpson's rule, which handles multimodal distributions better.
