# Changelog

## [0.19.0] - 2025-10-03

### Added
- Added product distributions expressible via tuples. (Note: Sufficient statistics are locked behind the `experimental` option since their behavior is not certain.)
- Added `ScaledPrior` and `ShiftedPrior` wrappers.


### Changed
- Updated to Rust 2024 Edition
- Updated `rand`, `rand_xoshiro`, and `rand_distr` deps.
- `NormalInvChiSquared`, `NormalGamma`, and `NormalInvGamme` `PpCache` for Gaussian conjugate analysis changed. `ln_pp_with_cache` is much faster.
- `Gamma` `PpCache` for Poisson conjugate analysis has been optimized. `ln_pp_with_cache` is faster.
- Fixed some typos.
- GitHub CI uses nightly since experimental features require nightly rust.
- `OnceLock` removed on distribution initialization in favor of uniform initialization cost over threads.
- Reworked `extract_stat` internals.
- `CDVM` was cleaned up, errors expanded, and parameters made more clear.
- Von Mises were sped up.
- `LogSumExp` was fixed when encountering -inf.
- `NormalInvGamma::ln_pp` was sped up.
- `NIX` `PpCache` was sped up by 20%.
- `NiX` Posterior predictive has its speed improved.

## [0.18.1] - 2025-02-28

### Fixed
- `LogSumExp` issues with `-Inf` entries

## [0.18.0] - 2024-12-09

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

### Changed
- Moved repository to GitHub.

## [0.16.4] - 2024-01-31
### Changed
- Bump min rust version to 1.72
- Fix bug in `dist::Beta` that sometimes resulted in invalid cache setting
  `alpha`or `beta`.
- Add `set_a` and `set_b` methods to `dist::Uniform`

## [0.16.3] - 2024-01-23

### Added
- Implement `Variance<f64>`for mixture models whose component distribution type
  also implements `Variance<f64>` and `Mean<f64>`.
- Implement `Variance<f32>`for mixture models whose component distribution type
  also implements `Variance<f32>` and `Mean<f32>`.

## [0.16.2] - 2024-01-02

### Changed
- Fix edge case in `misc::logsumexp` that would return `NaN` if the first value
  in the slice was `-inf`.

## [0.16.1] - 2023-12-14

### Changed
- Performance improvements in `misc::logsumexp`

## [0.16.0] - 2023-06-07

### Changed
- Fix bug in `InvGaussianSuffStat`
- Cache ln(weights) for `Mixture` to improve `ln_f` computation performance

### Added
- Add `ln_f_stat` method to `HasSuffStat` trait. This method computes the log
    likelihood of the data represented by the statistic.
- Add `InvGammaSuffStat`
- Add `BetaSuffStat`

### Removed
- Un-implement `HasSuffStat<f64>` for `InverseGamma`

## [0.15.1] - 2023-05-22

### Changed
- Fix bug in `Empirical::draw` that causes an `index out of bounds error` when
    the last last bin is drawn.

## [0.15.0] - 2023-03-07

### Changed
- All structs and enums use `snake_case` naming.
- Dependency updates (thanks @bsull).
- Entropy impl for Bernoulli Mixuture.
- Corrected NormalInverseGamma Posterior.
- Corrected NormalInverseWishart Posterior.
- Added `Kernel::reparameterize`.
- Improve CI.

## [0.14.4] - 2022-02-16

### Changed
- re-export nalgebra (flag `arraydist`)
- add `reparameterize` method to `process::gaussian::Kernel`, which will
    replace `from_parmeters` in 0.15.0
- Implement `Error` for `NormalInvGammaError` and `NormalInvChiSquaredError`

## [0.14.3] - 2022-02-16

### Removed
- Remove dependency on peroxide

## [0.14.2] - 2022-01-31

### Changed
- Fix some incorrect documentation
- Pass `--all-features` to docs.rs builds

## [0.14.1] - 2022-01-21

### Changed
- Fix bug in computation of entropies for continuous mixture models 

## [0.14.0] - 2022-01-05

### Changed
- Fix typo: `CategoricalError::EmptyWights` to `CategoricalError::EmptyWeights`
- Allow `Categorical::from_ln_weights` to accept zero (-Inf) weights
- Update `lru` crate to 0.7 due to security vulnerability in 0.6.6

## [0.13.1] - 2021-08-14

### Changed
- When using `dist::Gamma` to draw a `dist::Poisson`, the Poisson rate
    parameter is shifted to `f64::EPSILON` in the event of underflow.

## [0.13.0] - 2021-07-15

### Changed
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

### Added
- Added Inverse X^2 distribution
- Added Scaled Inverse X^2 distribution (`InvChiSquared`)
- Added Inverse Gaussian distribution
- Added Inverse Gaussian sufficient statistic
- Added Normal Inverse X^2 (`NormalInvChiSquared`) distribution as prior for
    Gaussian
- Added Inverse X^2 distribution (`InvChiSquared`)
- Implemented `From` instead of `Into` for sufficient statistic converters to
    allow the more explicit/ergonomic `From` conversions.
- Added Gaussian prior geweke tests and example
- More robust dpgmm example -- can use different priors

### Changed
- Improved some error messages in `NormalGamma`
- Caching of a normalizing constant in `BetaBinomial`
- Lots of testing verifying proper conjugate prior behavior
- Version updates
- Misc styling and warning fixes

## [0.11.1] - 2021-02-12

### Added
- Added Normal Inverse Gamma (`NormalInvGamma`) distribution as prior for
    Gaussian

## [0.11.0] - 2020-11-20

### Changed
- Gaussian process improvements including new kernels

### Added
- Added `_with_cache` variants to `ConjugatePrior` `ln_m` and `ln_pp` methods
    for use cases where these methods are called many times in different data
    without changing the underlying distribution. This also adds two more
    associated types to the trait (`LnMCache` and `LnPpCache`), and one method
    each to compute the caches.

### Removed
- Remove all references to `serde_derive` crate

## [0.10.4] - 2020-10-09

### Added
- Added Gaussian processes

## [0.10.3] - 2020-10-07
### Changed
- Updated math in `NormalGamma` and `GaussianSuffStat` to reduce rounding errors

## [0.10.2] - 2020-09-07
### Changed
- Categorical datum converters work like standard usize-to-boolean casting where
    0 is false, and anything greater than 0 is true.

## [0.10.1] - 2020-08-10
### Changed
- Add `from_parts_unchecked` method for constructing sufficient statistic since
    there was no way of manipulating statistic's fields or creating them
    manually.

## [0.10.0] - 2020-06-02

### Changed
- Implement Poisson Mode
- Store `GaussianSuffStat` in a more numerically stable way. The extra numerical
    stability comes at the cost of slower `observe` and `forget`. Serialization
    on `GaussianSuffStat` has changed.

## [0.9.3] - 2020-05-30

### Changed
- Entropy for Gaussian mixtures more robust to models with dispersed
    low-variance components
- Categorical::new_unchecked is now public

## [0.9.2] - 2020-05-23

### Changed
- Benchmarks

### Added
- implement `Entropy` and `KlDivergence` for `Poisson`
- Implement `QuadBounds` and `Entropy` for `Mixture<Poisson>`
- Implement `Mean` for mixtures of distributions that implement `Mean<f64>` or
    `Mean<f32>`
- `Rv` trait has `sample_stream` method that produces a never-ending iterator of
    random numbers.

## [0.9.1] - 2020-05-16

### Removed
- Remove printlns

## [0.9.0] - 2020-05-16

### Changed
- Refactor errors to provide more information
- Errors implement the `std::error::Error` trait
- Fixed a bug with `rv::dist::Mixture` that would compute means incorrectly
- `MvGaussian::dims` renamed to `MvGaussian::ndims`
- More aggressive caching of log likelihood terms
- Cached terms no longer unnecessarily computed on `Clone`
- Setters do input validation unless prefixed by `_unchecked`.
- feature `serde_support` renamed to `serde1`

### Removed
- Remove dependency on `getset`

## [0.8.3] - 2020-05-05

### Added
- Added Negative Binomial distribution
- Added Skellam Distribution
- Add `try_quad` function for numerical integration of functions that return
    `Result<f64, E>`.

## [0.8.2] - 2020-02-12

### Removed
- Remove dependency on `quadrature` crate in favor of hand-rolled adaptive
    Simpson's rule, which handles multimodal distributions better.

[0.19.0]: https://github.com/promise-ai/rv/compare/v0.18.1...v0.19.0
[0.18.1]: https://github.com/promise-ai/rv/compare/v0.18.0...v0.18.1
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
