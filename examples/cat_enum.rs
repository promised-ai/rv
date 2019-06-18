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
use rv::data::CategoricalDatum;
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

/// Then we implement the CatgoricalDatum trait for Color which has methods to
/// convert to and from a usize index.
impl CategoricalDatum for Color {
    fn into_usize(&self) -> usize {
        *self as usize
    }

    fn from_usize(n: usize) -> Self {
        match n {
            0 => Color::Red,
            1 => Color::Blue,
            2 => Color::Green,
            _ => panic!("Cannot convert {} to Color", n),
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
