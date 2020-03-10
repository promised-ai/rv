// tests that Clone, Debug, and PartialEq are implemented for a distribution
#[macro_export]
macro_rules! test_basic_impls {
    ($fx: expr) => {
        #[test]
        fn should_impl_debug_clone_and_partialeq() {
            assert_eq!($fx, $fx.clone());
            let _s1 = format!("{:?}", $fx);
        }
    };
}
