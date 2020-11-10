# Changelog

## 0.11.0
- Added `_with_cache` variants to `ConjucatePrior` `ln_m` and `ln_pp` methods
    for use cases where these methods are called many times in different data
    without changing the underlying distribution. This also adds two more
    associated types to the trait (`LnMCache` and `LnPpCache`), and one method
    each to compute the caches.
- Remove all references to `serde_derive` crate

## 0.10.5
- Added Gaussian processes
-
## 0.10.4
- Updated math in `NormalGamma` and `GaussianSuffStat` to reduce rounding errors

## 0.10.3
- Updated dependencies to get past yanked dependency issue.

## 0.10.2
- Categorical datum converters work like standard usize-to-boolean casting where 0 is false, and anything greater than 0 is true.

## 0.10.1
- Add `from_parts_unchecked` method for constructing sufficient statistic since there was no way of manipulating statistic's fields or creating them manually.

## 0.10.0
- Implement Poisson Mode
- Store `GaussianSuffStat` in a more numerically stable way. The extra numerical stability comes at the cost of slower `observe` and `forget`. Serialization on `GaussianSuffStat` has changed.

## 0.9.3
- Entropy for Gaussian mixtures more robust to models with dispersed low-variance components
- Categorical::new_unchecked is now public

## 0.9.2
- Benchmarks
- implement `Entropy` and `KlDivergence` for `Poisson`
- Implement `QuadBounds` and `Entropy` for `Mixture<Poisson>`
- Implement `Mean` for mixtures of distributions that implement `Mean<f64>` or `Mean<f32>`
- `Rv` trait has `sample_stream` method that produces a never-ending iterator of random numbers.

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
- Add `try_quad` function for numerical integration of functions that return `Result<f64, E>`.

## 0.8.2
- Remove dependency on `quadrature` crate in favor of hand-rolled adaptive Simpson's rule, which handles multimodal distributions better.
