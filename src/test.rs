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
            let fx2 = $fx.clone();
            assert_eq!($fx, fx2);
            let _f = $fx.ln_f(&$x);
            assert_eq!(fx2, $fx);
            let _s1 = format!("{:?}", $fx);
        }
    };
}
