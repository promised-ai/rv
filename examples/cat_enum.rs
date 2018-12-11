/// Using an enum as a categorical variable.
///
/// In a future where proc_macro is stable, we can offload all of this to a
/// custom derive to have code that looks like this:
///
/// ```
/// #[derive(CategoricalEnum)]
/// enum Color {
///     Red = 0,
///     Blue = 1,
///     Green = 2,
/// }
/// ```
extern crate rand;
extern crate rv;
extern crate num;

use num::FromPrimitive;
use std::convert::From;
use rv::prelude::*;

/// We have to assign values 0, ..., n-1 to the enum values so they map to
/// indices in the categorical weights
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
enum Color {
    Red = 0,
    Blue = 1,
    Green = 2,
}

/// Then we implement a couple of traits for our `enum` so it satisfies the
/// requirements to be a `CategoricalDatum`.
///
/// We must implement `From<Color>` for `usize` so we can convert the color to
/// a vector index.
impl From<Color> for usize {
    fn from(color: Color) -> usize {
        color as usize
    }
}

/// We must implement `FromPrimitive` for color, so we can convert from a
/// `usize` index into a `Color` on Rust stable. We must, at minimum, implement
/// `from_i64` and `from_u64` -- the rest of the trait methods have defaults.
impl FromPrimitive for Color {
    fn from_i64(x: i64) -> Option<Self> {
        match x {
            0 => Some(Color::Red),
            1 => Some(Color::Blue),
            2 => Some(Color::Green),
            _ => None,
        }
    }

    fn from_u64(x: u64) -> Option<Self> {
        match x {
            0 => Some(Color::Red),
            1 => Some(Color::Blue),
            2 => Some(Color::Green),
            _ => None,
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    let ctgrl = Categorical::new(&vec![0.25, 0.25, 0.5]).unwrap();

    // Compute the PMF, P(Red).
    let p = ctgrl.pmf(&Color::Red);
    println!("p({:?}) = {}", Color::Red, p);

    // Take 10 draws from {Red, Blue, Green} according to the distribution.
    let xs: Vec<Color> = ctgrl.sample(10, &mut rng);
    println!("xs: {:?}", xs);
}
