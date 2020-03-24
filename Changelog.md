# Changelog

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
