# Changelog

## 0.8.3
- Added Negative Binomial distribution
- Added Skellam Distribution
- Add `try_quad` function for numerical integration of functions that return `Result<f64, E>`.

## 0.8.2
- Remove dependency on `quadrature` crate in favor of hand-rolled adaptive Simpson's rule, which handles multimodal distributions better.
