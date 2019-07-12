#[cfg(feature = "serde_support")]
use serde_derive::{Deserialize, Serialize};

use crate::impl_display;
use crate::misc::vec_to_string;
use crate::result;

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Partition {
    /// The assignment of the n items to partitions 0, ..., k-1
    z: Vec<usize>,
    /// The number of items assigned to each partition
    counts: Vec<usize>,
}

impl Default for Partition {
    fn default() -> Self {
        Partition::new()
    }
}

impl From<&Partition> for String {
    fn from(part: &Partition) -> String {
        let mut out = String::new();
        out.push_str(
            format!("Partition (n: {}, k: {})\n", part.len(), part.k())
                .as_str(),
        );
        out.push_str(
            format!("  assignment: {}\n", vec_to_string(&part.z, 15)).as_str(),
        );
        out.push_str(
            format!("  counts: {}\n", vec_to_string(&part.counts, part.k()))
                .as_str(),
        );
        out
    }
}

impl_display!(Partition);

impl Partition {
    /// Empty partition
    pub fn new() -> Partition {
        Partition {
            z: vec![],
            counts: vec![],
        }
    }

    pub fn new_unchecked(z: Vec<usize>, counts: Vec<usize>) -> Self {
        Partition { z, counts }
    }

    pub fn z(&self) -> &Vec<usize> {
        &self.z
    }

    pub fn z_mut(&mut self) -> &mut Vec<usize> {
        &mut self.z
    }

    pub fn counts(&self) -> &Vec<usize> {
        &self.counts
    }

    pub fn counts_mut(&mut self) -> &mut Vec<usize> {
        &mut self.counts
    }

    /// Create a `Partition` with a given assignment, `z`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rv::data::Partition;
    /// let z1 = vec![0, 1, 2, 3, 1, 2];
    /// let part = Partition::from_z(z1).unwrap();
    ///
    /// assert_eq!(*part.z(), vec![0, 1, 2, 3, 1, 2]);
    /// assert_eq!(*part.counts(), vec![1, 2, 2, 1]);
    ///
    /// // Invalid z because k=4 is empty. All partitions must be occupied.
    /// let z2 = vec![0, 1, 2, 3, 1, 5];
    /// assert!(Partition::from_z(z2).is_err());
    /// ```
    pub fn from_z(z: Vec<usize>) -> result::Result<Self> {
        // TODO: integrate NoneError into output instead of using expect
        let k = *z.iter().max().expect("empty z") + 1;
        let mut counts: Vec<usize> = vec![0; k];
        z.iter().for_each(|&zi| counts[zi] += 1);

        if counts.iter().all(|&ct| ct > 0) {
            let part = Partition { z, counts };
            Ok(part)
        } else {
            let err_kind = result::ErrorKind::InvalidParameterError;
            Err(result::Error::new(err_kind, "Unoccupied partition(s)"))
        }
    }

    /// Remove the item at index `ix`
    ///
    /// # Example
    ///
    /// ```
    /// # use rv::data::Partition;
    /// let mut part = Partition::from_z(vec![0, 1, 0, 2]).unwrap();
    /// part.remove(1).expect("Could not remove");
    ///
    /// assert_eq!(*part.z(), vec![0, 0, 1]);
    /// assert_eq!(*part.counts(), vec![2, 1]);
    /// ```
    pub fn remove(&mut self, ix: usize) -> result::Result<()> {
        // Panics  on index error panics.
        let zi = self.z.remove(ix);
        if self.counts[zi] == 1 {
            let _ct = self.counts.remove(zi);
            // ensure canonical order
            self.z.iter_mut().for_each(|zj| {
                if *zj > zi {
                    *zj -= 1
                }
            });
            Ok(())
        } else {
            self.counts[zi] -= 1;
            Ok(())
        }
    }

    /// Append a new item assigned to partition `zi`
    ///
    /// # Example
    ///
    /// ``` rust
    /// # use rv::data::Partition;
    /// let mut part = Partition::from_z(vec![0, 1, 0, 2]).unwrap();
    /// part.append(3).expect("Could not append");
    ///
    /// assert_eq!(*part.z(), vec![0, 1, 0, 2, 3]);
    /// assert_eq!(*part.counts(), vec![2, 1, 1, 1]);
    /// ```
    pub fn append(&mut self, zi: usize) -> result::Result<()> {
        let k = self.k();
        if zi > k {
            let err_kind = result::ErrorKind::InvalidParameterError;
            let err = result::Error::new(err_kind, "zi higher than k");
            Err(err)
        } else {
            self.z.push(zi);
            if zi == k {
                self.counts.push(1);
            } else {
                self.counts[zi] += 1;
            }
            Ok(())
        }
    }

    /// Returns the number of partitions, k.
    ///
    /// # Example
    ///
    /// ``` rust
    /// # use rv::data::Partition;
    /// let part = Partition::from_z(vec![0, 1, 0, 2]).unwrap();
    ///
    /// assert_eq!(part.k(), 3);
    /// assert_eq!(*part.counts(), vec![2, 1, 1]);
    /// ```
    pub fn k(&self) -> usize {
        self.counts.len()
    }

    /// Returns the number items
    pub fn len(&self) -> usize {
        self.z.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the partition weights (normalized counts)
    ///
    /// # Example
    ///
    /// ``` rust
    /// # use rv::data::Partition;
    /// let part = Partition::from_z(vec![0, 1, 0, 2]).unwrap();
    /// let weights = part.weights();
    ///
    /// assert_eq!(weights, vec![0.5, 0.25, 0.25]);
    /// ```
    pub fn weights(&self) -> Vec<f64> {
        let n = self.len() as f64;
        self.counts.iter().map(|&ct| (ct as f64) / n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let part = Partition::from_z(vec![0, 1, 0, 2]).unwrap();

        assert_eq!(part.k(), 3);
        assert_eq!(part.counts, vec![2, 1, 1]);
    }
}
