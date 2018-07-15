use std::io;

pub struct Partition {
    /// The assignment of the n items to partitions 0, ..., k-1
    pub z: Vec<usize>,
    /// The number of items assigned to each partition
    pub counts: Vec<usize>,
}

impl Partition {
    /// Empty partition
    pub fn new() -> Partition {
        Partition {
            z: vec![],
            counts: vec![],
        }
    }

    /// Create a `Partition` with a given assignment, `z`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate rv;
    /// # use rv::partition::Partition;
    /// #
    /// let z1 = vec![0, 1, 2, 3, 1, 2];
    /// let part = Partition::from_z(z1).unwrap();
    ///
    /// assert_eq!(part.z, vec![0, 1, 2, 3, 1, 2]);
    /// assert_eq!(part.counts, vec![1, 2, 2, 1]);
    ///
    /// // Invalid z because k=4 is empty. All partitions must be occupied.
    /// let z2 = vec![0, 1, 2, 3, 1, 5];
    /// assert!(Partition::from_z(z2).is_err());
    /// ```
    pub fn from_z(z: Vec<usize>) -> io::Result<Self> {
        // TODO: integrate NoneError into output instead of using expect
        let k = *z.iter().max().expect("empty z") + 1;
        let mut counts: Vec<usize> = vec![0; k];
        z.iter().for_each(|&zi| counts[zi] += 1);

        if counts.iter().all(|&ct| ct > 0) {
            let part = Partition {
                z: z,
                counts: counts,
            };
            Ok(part)
        } else {
            let err_kind = io::ErrorKind::InvalidInput;
            Err(io::Error::new(err_kind, "Unoccupied partition(s)"))
        }
    }

    /// Remove the item at index `ix`
    ///
    /// # Example
    ///
    /// ``` rust
    /// # extern crate rv;
    /// # use rv::partition::Partition;
    /// #
    /// let mut part = Partition::from_z(vec![0, 1, 0, 2]).unwrap();
    /// part.remove(1).expect("Could not remove");
    ///
    /// assert_eq!(part.z, vec![0, 0, 1]);
    /// assert_eq!(part.counts, vec![2, 1]);
    /// ```
    pub fn remove(&mut self, ix: usize) -> io::Result<()> {
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
    /// # extern crate rv;
    /// # use rv::partition::Partition;
    /// #
    /// let mut part = Partition::from_z(vec![0, 1, 0, 2]).unwrap();
    /// part.append(3).expect("Could not append");
    ///
    /// assert_eq!(part.z, vec![0, 1, 0, 2, 3]);
    /// assert_eq!(part.counts, vec![2, 1, 1, 1]);
    /// ```
    pub fn append(&mut self, zi: usize) -> io::Result<()> {
        let k = self.k();
        if zi > k {
            let err_kind = io::ErrorKind::InvalidInput;
            let err = io::Error::new(err_kind, "zi higher than k");
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
    /// # extern crate rv;
    /// # use rv::partition::Partition;
    /// #
    /// let part = Partition::from_z(vec![0, 1, 0, 2]).unwrap();
    ///
    /// assert_eq!(part.k(), 3);
    /// assert_eq!(part.counts, vec![2, 1, 1]);
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
    /// # extern crate rv;
    /// # use rv::partition::Partition;
    /// #
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
