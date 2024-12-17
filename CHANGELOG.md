# Changelog

## [0.19.0] - 2024-06-24

### Changed
- `NormalInvChiSquared`, `NormalGamma`, and `NormalInvGamme` `PpCache` for Gaussian conjugate analysis changed. `ln_pp_with_cache` is much faster.

## [0.18.0] - 2024-06-24

### Added
- Add log1pexp and logaddexp
- Add LogSumExp trait with logsumexp method. This way we can make applying it a little more generic, similar to how sum works.
- Propagate these functions across crate

### Removed
- Removed logsumexp function taking a slice argument

## [0.17.0] - 2024-06-24

### Added
- experimental module with stick_breaking_process submodule
- ConvergentSequence trait and sorted_uniforms function
- Parameterized trait for distributions
- Process trait generalizing Rv
- HasDensity and Sampleable traits split from Rv
- Implementations of ConjugatePrior<X, Bernoulli> for UnitPowerLaw
- Implementations of Parameterized, HasDensity and Sampleable for various distributions
- Tests for new trait implementations
- Chad Scherrer to authors

### Changed
- Updated dependencies including ahash, aho-corasick, anyhow, argmin, argmin-math, autocfg, bit-vec
- Refactored ConjugatePrior with changes to cache names
- Updated import statements to use wildcards for rv::traits module
- Changed random number generator in examples to Xoshiro256Plus
- Renamed LnMCache and LnPpCache to MCache and PpCache in conjugate prior implementations
- Updated code examples in documentation
- Made minor stylistic changes suggested by Clippy
- Updated test profile for proptest with opt-level 3

### Removed
- datum module and related code
- Distribution enum and ProductDistribution struct
- Rv trait implementation for various distributions

### Fixed
- Various typos in comments and documentation
- Incorrect usages of std::f64 constants


## [0.16.5] - 2024-03-14
- Moved repository to GitHub.

## [0.16.4] - 2024-01-31
- Bump min rust version to 1.72
- Fix bug in `dist::Beta` that sometimes resulted in invalid cache setting
  `alpha`or `beta`.
- Add `set_a` and `set_b` methods to `dist::Uniform`

## [0.16.3] - 2024-01-23
- Implement `Variance<f64>`for mixture models whose component distribution type
  also implements `Variance<f64>` and `Mean<f64>`.
- Implement `Variance<f32>`for mixture models whose component distribution type
  also implements `Variance<f32>` and `Mean<f32>`.

## [0.16.2] - 2024-01-02
- Fix edge case in `misc::logsumexp` that would return `NaN` if the first value
  in the slice was `-inf`.

## [0.16.1] - 2023-12-14
- Performance improvements in `misc::logsumexp`

## [0.16.0] - 2023-06-07
- Fix bug in `InvGaussianSuffStat`
- Add `ln_f_stat` method to `HasSuffStat` trait. This method computes the log
    likelihood of the data represented by the statistic.
- Un-implement `HasSuffStat<f64>` for `InverseGamma`
- Add `InvGammaSuffStat`
- Add `BetaSuffStat`
- Cache ln(weights) for `Mixture` to improve `ln_f` computation performance

## [0.15.1] - 2023-05-22
- Fix bug in `Empirical::draw` that causes an `index out of bounds error` when
    the last last bin is drawn.

## [0.15.0] - 2023-03-07
- All structs and enums use `snake_case` naming.
- Dependency updates (thanks @bsull).
- Entropy impl for Bernoulli Mixuture.
- Corrected NormalInverseGamma Posterior.
- Corrected NormalInverseWishart Posterior.
- Added `Kernel::reparameterize`.
- Improve CI.

## [0.14.4] - 2022-02-16
- re-export nalgebra (flag `arraydist`)
- add `reparameterize` method to `process::gaussian::Kernel`, which will
    replace `from_parmeters` in 0.15.0
- Implement `Error` for `NormalInvGammaError` and `NormalInvChiSquaredError`

## [0.14.3] - 2022-02-16
- Remove dependency on peroxide

## [0.14.2] - 2022-01-31
- Fix some incorrect documentation
- Pass `--all-features` to docs.rs builds

## [0.14.1] - 2022-01-21
- Fix bug in computation of entropies for continuous mixture models 

## [0.14.0] - 2022-01-05
- Fix typo: `CategoricalError::EmptyWights` to `CategoricalError::EmptyWeights`
- Allow `Categorical::from_ln_weights` to accept zero (-Inf) weights
- Update `lru` crate to 0.7 due to security vulnerability in 0.6.6

## [0.13.1] - 2021-08-14
- When using `dist::Gamma` to draw a `dist::Poisson`, the Poisson rate
    parameter is shifted to `f64::EPSILON` in the event of underflow.

## [0.13.0] - 2021-07-15
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

## [0.12.0] - 2021-05-05
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

## [0.11.1] - 2021-02-12
- Added Normal Inverse Gamma (`NormalInvGamma`) distribution as prior for
    Gaussian

## [0.11.0] - 2020-11-20
- Added `_with_cache` variants to `ConjugatePrior` `ln_m` and `ln_pp` methods
    for use cases where these methods are called many times in different data
    without changing the underlying distribution. This also adds two more
    associated types to the trait (`LnMCache` and `LnPpCache`), and one method
    each to compute the caches.
- Remove all references to `serde_derive` crate
- Gaussian process improvements including new kernels

## [0.10.4] - 2020-10-09
- Added Gaussian processes

## [0.10.3] - 2020-10-07
- Updated math in `NormalGamma` and `GaussianSuffStat` to reduce rounding errors

## [0.10.2] - 2020-09-07
- Categorical datum converters work like standard usize-to-boolean casting where
    0 is false, and anything greater than 0 is true.

## [0.10.1] - 2020-08-10
- Add `from_parts_unchecked` method for constructing sufficient statistic since
    there was no way of manipulating statistic's fields or creating them
    manually.

## [0.10.0] - 2020-06-02
- Implement Poisson Mode
- Store `GaussianSuffStat` in a more numerically stable way. The extra numerical
    stability comes at the cost of slower `observe` and `forget`. Serialization
    on `GaussianSuffStat` has changed.

## [0.9.3] - 2020-05-30
- Entropy for Gaussian mixtures more robust to models with dispersed
    low-variance components
- Categorical::new_unchecked is now public

## [0.9.2] - 2020-05-23
- Benchmarks
- implement `Entropy` and `KlDivergence` for `Poisson`
- Implement `QuadBounds` and `Entropy` for `Mixture<Poisson>`
- Implement `Mean` for mixtures of distributions that implement `Mean<f64>` or
    `Mean<f32>`
- `Rv` trait has `sample_stream` method that produces a never-ending iterator of
    random numbers.

## [0.9.1] - 2020-05-16
- Remove printlns

## [0.9.0] - 2020-05-16
- Refactor errors to provide more information
- Errors implement the `std::error::Error` trait
- Fixed a bug with `rv::dist::Mixture` that would compute means incorrectly
- `MvGaussian::dims` renamed to `MvGaussian::ndims`
- More aggressive caching of log likelihood terms
- Cached terms no longer unnecessarily computed on `Clone`
- Remove dependency on `getset`
- Setters do input validation unless prefixed by `_unchecked`.
- feature `serde_support` renamed to `serde1`

## [0.8.3] - 2020-05-05
- Added Negative Binomial distribution
- Added Skellam Distribution
- Add `try_quad` function for numerical integration of functions that return
    `Result<f64, E>`.

## [0.8.2] - 2020-02-12
- Remove dependency on `quadrature` crate in favor of hand-rolled adaptive
    Simpson's rule, which handles multimodal distributions better.

[0.18.0]: https://github.com/promise-ai/rv/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/promise-ai/rv/compare/v0.16.5...v0.17.0
[0.16.5]: https://github.com/promise-ai/rv/compare/v0.16.4...v0.16.5
[0.16.4]: https://github.com/promise-ai/rv/compare/v0.16.3...v0.16.4
[0.16.3]: https://github.com/promise-ai/rv/compare/v0.16.2...v0.16.3
[0.16.2]: https://github.com/promise-ai/rv/compare/v0.16.1...v0.16.2
[0.16.1]: https://github.com/promise-ai/rv/compare/v0.16.0...v0.16.1
[0.16.0]: https://github.com/promise-ai/rv/compare/v0.15.1...v0.16.0
[0.15.1]: https://github.com/promise-ai/rv/compare/v0.15.0...v0.15.1
[0.15.0]: https://github.com/promise-ai/rv/compare/v0.14.4...v0.15.0
[0.14.4]: https://github.com/promise-ai/rv/compare/v0.14.3...v0.14.4
[0.14.3]: https://github.com/promise-ai/rv/compare/v0.14.2...v0.14.3
[0.14.2]: https://github.com/promise-ai/rv/compare/v0.14.1...v0.14.2
[0.14.1]: https://github.com/promise-ai/rv/compare/v0.14.0...v0.14.1
[0.14.0]: https://github.com/promise-ai/rv/compare/v0.13.1...v0.14.0
[0.13.1]: https://github.com/promise-ai/rv/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/promise-ai/rv/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/promise-ai/rv/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/promise-ai/rv/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/promise-ai/rv/compare/v0.10.4...v0.11.0
[0.10.4]: https://github.com/promise-ai/rv/compare/v0.10.3...v0.10.4
[0.10.3]: https://github.com/promise-ai/rv/compare/v0.10.2...v0.10.3
[0.10.2]: https://github.com/promise-ai/rv/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/promise-ai/rv/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/promise-ai/rv/compare/v0.9.3...v0.10.0
[0.9.3]: https://github.com/promise-ai/rv/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/promise-ai/rv/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/promise-ai/rv/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/promise-ai/rv/compare/v0.8.3...v0.9.0
[0.8.3]: https://github.com/promise-ai/rv/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/promise-ai/rv/release/tag/v0.8.2
