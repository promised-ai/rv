#[cfg(test)]
use approx::RelativeEq;

// tests that Clone, Debug, and PartialEq are implemented for a distribution
// Tests that partial eq is not sensitive to OnceCell initialization, which
// often happens in ln_f is called
#[macro_export]
macro_rules! test_basic_impls {
    ([continuous] $fx: expr) => {
        test_basic_impls!($fx, 0.5_f64);
    };
    ([categorical] $fx: expr) => {
        test_basic_impls!($fx, 0_usize);
    };
    ([count] $fx: expr) => {
        test_basic_impls!($fx, 3_u32);
    };
    ([binary] $fx: expr) => {
        test_basic_impls!($fx, true);
    };
    ($fx: expr, $x: expr) => {
        #[test]
        fn should_impl_debug_clone_and_partialeq() {
            // make the expression a thing. If we don't do this, calling $fx
            // reconstructs the distribution which means we don't do caching
            let fx = $fx;

            // clone a copy of fn before any computation of cached values is
            // done
            let fx2 = fx.clone();
            assert_eq!($fx, fx2);

            // Computing ln_f normally initializes all cached values
            let y1 = fx.ln_f(&$x);
            let y2 = fx.ln_f(&$x);
            assert_eq!(y1, y2);

            // check the fx == fx2 despite fx having its cached values initalized
            assert_eq!(fx2, fx);

            // Make sure Debug is implemented for fx
            let _s1 = format!("{:?}", fx);
        }
    };
}

#[cfg(test)]
/// Assert Relative Eq for sequences
pub fn relative_eq<T, I>(
    left: I,
    right: I,
    epsilon: T::Epsilon,
    max_relative: T::Epsilon,
) -> bool
where
    T: RelativeEq,
    T::Epsilon: Copy,
    I: IntoIterator<Item = T>,
    <I as IntoIterator>::IntoIter: ExactSizeIterator,
{
    let a = left.into_iter();
    let b = right.into_iter();

    if a.len() != b.len() {
        return false;
    }

    a.zip(b)
        .all(|(a, b)| a.relative_eq(&b, epsilon, max_relative))
}
