extern crate rand;
extern crate rv;

use rand::Rng;
use rv::data::Partition;
use rv::dist::{Crp, Gaussian, NormalGamma};
use rv::misc::ln_pflip;
use rv::traits::*;
use rv::ConjugateModel;
use std::sync::Arc;

// Save keystrokes!
type GaussComponent = ConjugateModel<f64, Gaussian, NormalGamma>;

// Infinite mixture (CRP) model of univariate Gaussians
struct Dpgmm {
    // The data
    xs: Vec<f64>,
    // Keeps track of the data IDs as they're removed and replaced
    ixs: Vec<usize>,
    // The prior on the partition of data
    crp: Crp,
    // The current partition
    partition: Partition,
    // The Prior on each of the components. Component means are from a Gaussian
    // (Normal) distribution, and the precisions (reciprocal of the variance)
    // is from a gamma distribution.
    prior: Arc<NormalGamma>,
    // A vector of univariate normals with the conjugate Normal Gamma prior.
    components: Vec<GaussComponent>,
}

impl Dpgmm {
    // Draws a Dpgmm from the prior
    fn new<R: Rng>(
        xs: Vec<f64>,
        prior: NormalGamma,
        alpha: f64,
        mut rng: &mut R,
    ) -> Self {
        let n = xs.len();

        // Partition prior
        let crp = Crp::new(alpha, n).expect("Invalid params");

        // Initial partition drawn from the prior
        let partition = crp.draw(&mut rng);

        // Put the prior in a reference counter
        let prior_arc = Arc::new(prior);

        // Create an empty component for each partition. Gaussian::default()
        // is used as a template; The parameters don't matter.
        let mut components: Vec<GaussComponent> = (0..partition.k())
            .map(|_| {
                ConjugateModel::new(&Gaussian::default(), prior_arc.clone())
            }).collect();

        // Given the data to their respective components by having them observe
        // their data.
        xs.iter()
            .zip(partition.z.iter())
            .for_each(|(xi, &zi)| components[zi].observe(xi));

        Dpgmm {
            xs: xs,
            ixs: (0..n).collect(),
            crp: crp,
            partition: partition,
            prior: prior_arc,
            components: components,
        }
    }

    // Number of data
    fn n(&self) -> usize {
        self.xs.len()
    }

    /// Remove and return the datum at index `ix`. Return the datum and its
    /// index.
    fn remove(&mut self, pos: usize) -> (f64, usize) {
        let x = self.xs.remove(pos);
        let ix = self.ixs.remove(pos);
        let zi = self.partition.z[pos];

        let is_singleton = self.partition.counts[zi] == 1;
        self.partition.remove(pos).expect("could not remove");

        // If x was in a component by itself, remove that component; otherwise
        // have that component forget it.
        if is_singleton {
            let _cj = self.components.remove(zi);
        } else {
            self.components[zi].forget(&x);
        }

        (x, ix)
    }

    // For a datum `x` with index `ix`, assigns `x` to a partition
    // probabilistically according to the DPGMM. The datum is appended to the
    // end of `xs` and the assignment, `z`.
    fn insert<R: Rng>(&mut self, x: f64, ix: usize, mut rng: &mut R) {
        let mut ln_weights: Vec<f64> = self
            .partition
            .counts
            .iter()
            .zip(self.components.iter())
            .map(|(&w, cj)| (w as f64).ln() + cj.ln_pp(&x))  // nk * p(xi|xk)
            .collect();

        let mut ctmp: GaussComponent =
            ConjugateModel::new(&Gaussian::default(), self.prior.clone());

        // probability of being in a new category -- α * p(xi)
        ln_weights.push(self.crp.alpha.ln() + ctmp.ln_pp(&x));

        // Draws a new assignment in proportion with the weights
        let zi = ln_pflip(&ln_weights, 1, false, &mut rng)[0];

        // Here is where we re-insert the data back into xs, ixs, and the
        // partition.
        if zi == self.partition.k() {
            // If we've created a singleton, we must push a new component
            ctmp.observe(&x);
            self.components.push(ctmp);
        }

        // Push x, ix, and zi to the end of the list
        self.components[zi].observe(&x);
        self.xs.push(x);
        self.ixs.push(ix);
        self.partition.append(zi).expect("Could not append");
    }

    // reassigns a the datum at the position `pos`
    fn step<R: Rng>(&mut self, pos: usize, mut rng: &mut R) {
        let (x, ix) = self.remove(pos);
        self.insert(x, ix, &mut rng);
    }

    // Reassigns each datum in random order
    fn scan<R: Rng>(&mut self, mut rng: &mut R) {
        let mut positions: Vec<usize> = (0..self.n()).collect();
        rng.shuffle(&mut positions);
        positions.iter().for_each(|&pos| self.step(pos, &mut rng));
    }

    // Run the DPGMM for `iters` iterations
    fn run<R: Rng>(&mut self, iters: usize, mut rng: &mut R) {
        (0..iters).for_each(|_| self.scan(&mut rng));
        self.sort() // restore data/assignment order
    }

    // The data get shuffled as a result of the removal/insertion process, so we
    // need to re-sort the data by their indices to ensure the data and the
    // assignment are in the same order they were when they were passed in
    fn sort(&mut self) {
        let mut xs: Vec<f64> = vec![0.0; self.n()];
        let mut z: Vec<usize> = vec![0; self.n()];
        self.ixs.iter().enumerate().for_each(|(pos, &ix)| {
            xs[ix] = self.xs[pos];
            z[ix] = self.partition.z[pos];
        });
        self.partition.z = z;
        self.xs = xs;
        self.ixs = (0..self.n()).collect();
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    // Generate 100 data from two Gaussians. The Gaussians are far enought apart
    // that the DPGMM should separate them.
    let mut xs: Vec<f64> =
        Gaussian::new(-3.0, 1.0).unwrap().sample(50, &mut rng);
    let mut ys: Vec<f64> =
        Gaussian::new(3.0, 1.0).unwrap().sample(50, &mut rng);
    xs.append(&mut ys);

    // Parameters are more or less arbitrary. The only thing we need to worry
    // about is scale.
    let prior = NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap();

    // Draw a DPGMM from the prior
    let mut dpgmm = Dpgmm::new(xs, prior, 1.0, &mut rng);

    // .. and run it
    dpgmm.run(200, &mut rng);

    // there should be two categories, the first half belong to one category,
    // and the second half belong to the other. Something like
    // [0, 0, 0, 0, ...,0, 1, ..., 1, 1, 1, 1] -- subject to some noise,
    // because we don't actually know how many components there are.
    println!("{:?}", dpgmm.partition.z);
}
