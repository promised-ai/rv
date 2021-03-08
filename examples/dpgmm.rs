// Dirichlet Process Mixture Model
// -------------------------------
//
// In this example, we're going to build a Dirichlet Process Mixture Model
// (DPMM). In a typical mixture model, we assume we know the number of
// copmonents and learn the parameters for each component that best fit the
// data. For example, we might use a 2-component model to fit to bi-modal data.
// The DPMM uses a probabilistic process -- the Diriclet Process -- to describe
// how data are assigned to components, and does inference on the parameters of
// that process as well as the component parameters. The DPMM weighs simplicity
// (prefer fewer componets) with explanation.
//
// Below, we implement the collapsed Gibbs algorithm for sampling from s DPMM.
// The code is generic to any type of mixture as long as it has a conjugate
// prior.
//
// References
// ----------
//
// Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process
//     mixture models. Journal of computational and graphical statistics, 9(2),
//     249-265.
//
// Rasmussen, C. E. (1999, December). The infinite Gaussian mixture model. In
//     NIPS (Vol. 12, pp. 554-560).

use rand::seq::SliceRandom;
use rand::Rng;
use rv::data::Partition;
use rv::dist::{Crp, Gaussian, NormalInvGamma};
use rv::misc::ln_pflip;
use rv::traits::*;
use rv::ConjugateModel;
use std::sync::Arc;

// Infinite mixture (CRP) model
//
// This code is general to any type of mixture as long as it has a conjugate
// prior
struct Dpmm<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
{
    // The data
    xs: Vec<X>,
    // Keeps track of the data IDs as they're removed and replaced
    ixs: Vec<usize>,
    // The prior on the partition of data
    crp: Crp,
    // The current partition
    partition: Partition,
    // The Prior on each of the components.
    prior: Arc<Pr>,
    // A vector of component models with conjugate priors
    components: Vec<ConjugateModel<X, Fx, Pr>>,
}

impl<X, Fx, Pr> Dpmm<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
{
    // Draws a Dpmm from the prior
    fn new<R: Rng>(xs: Vec<X>, prior: Pr, alpha: f64, rng: &mut R) -> Self {
        let n = xs.len();

        // Partition prior
        let crp = Crp::new(alpha, n).expect("Invalid params");

        // Initial partition drawn from the prior
        let partition = crp.draw(rng);

        // Put the prior in a reference counter
        let prior_arc = Arc::new(prior);

        // Create an empty component for each partition. Drawing component
        // models is used as a template; The parameters don't matter because we
        // marginalize them away through the magic of conjugate priors.
        let mut components: Vec<ConjugateModel<X, Fx, Pr>> = (0..partition.k())
            .map(|_| {
                ConjugateModel::new(&prior_arc.draw(rng), prior_arc.clone())
            })
            .collect();

        // Given the data to their respective components by having them observe
        // their data.
        xs.iter()
            .zip(partition.z().iter())
            .for_each(|(xi, &zi)| components[zi].observe(xi));

        Dpmm {
            xs,
            ixs: (0..n).collect(),
            crp,
            partition,
            prior: prior_arc,
            components,
        }
    }

    // Number of data
    fn n(&self) -> usize {
        self.xs.len()
    }

    /// Remove and return the datum at index `ix`. Return the datum and its
    /// index.
    fn remove(&mut self, pos: usize) -> (X, usize) {
        let x = self.xs.remove(pos);
        let ix = self.ixs.remove(pos);
        let zi = self.partition.z()[pos];

        let is_singleton = self.partition.counts()[zi] == 1;
        self.partition.remove(pos).expect("could not remove");

        // If x was in a component by itself, remove that component; otherwise
        // have that component forget it.
        if is_singleton {
            let _cj = self.components.remove(zi);
        } else {
            self.components[zi].forget(&x);
            assert!(self.components[zi].n() > 0);
        }

        (x, ix)
    }

    // For a datum `x` with index `ix`, assigns `x` to a partition
    // probabilistically according to the DPGMM. The datum is appended to the
    // end of `xs` and the assignment, `z`.
    fn insert<R: Rng>(&mut self, x: X, ix: usize, rng: &mut R) {
        let mut ln_weights: Vec<f64> = self
            .partition
            .counts()
            .iter()
            .zip(self.components.iter())
            .map(|(&w, cj)| (w as f64).ln() + cj.ln_pp(&x)) // nk * p(xi|xk)
            .collect();

        let mut ctmp: ConjugateModel<X, Fx, Pr> =
            ConjugateModel::new(&self.prior.draw(rng), self.prior.clone());

        // probability of being in a new category -- α * p(xi)
        ln_weights.push(self.crp.alpha().ln() + ctmp.ln_pp(&x));

        // Draws a new assignment in proportion with the weights
        let zi = ln_pflip(&ln_weights, 1, false, rng)[0];

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
    fn step<R: Rng>(&mut self, pos: usize, rng: &mut R) {
        let (x, ix) = self.remove(pos);
        self.insert(x, ix, rng);
    }

    // Reassigns each datum in random order
    fn scan<R: Rng>(&mut self, rng: &mut R) {
        let mut positions: Vec<usize> = (0..self.n()).collect();
        positions.shuffle(rng);
        positions.iter().for_each(|&pos| self.step(pos, rng));
    }

    // Run the DPGMM for `iters` iterations
    fn run<R: Rng>(&mut self, iters: usize, rng: &mut R) {
        (0..iters).for_each(|_| self.scan(rng));
        self.sort() // restore data/assignment order
    }

    // The data get shuffled as a result of the removal/insertion process, so we
    // need to re-sort the data by their indices to ensure the data and the
    // assignment are in the same order they were when they were passed in
    fn sort(&mut self) {
        // This will at most do n swaps, but I feel like there's probably some
        // really obvious way to do better. Oh well... I'm an ML guy, not an
        // algorithms guy.
        for i in 0..self.n() {
            while self.ixs[i] != i {
                let j = self.ixs[i];
                self.ixs.swap(i, j);
                self.partition.z_mut().swap(i, j);
                self.xs.swap(i, j);
            }
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    // Generate 100 data from two Gaussians. The Gaussians are far enough apart
    // that the DPGMM should separate them.
    let mut xs: Vec<f64> =
        Gaussian::new(-3.0, 1.0).unwrap().sample(50, &mut rng);
    let mut ys: Vec<f64> =
        Gaussian::new(3.0, 1.0).unwrap().sample(50, &mut rng);
    xs.append(&mut ys);

    // Parameters are more or less arbitrary. The only thing we need to worry
    // about is scale.
    let prior = NormalInvGamma::new(0.0, 1.0, 1.0, 1.0).unwrap();

    // Draw a DPGMM from the prior
    let mut dpgmm = Dpmm::new(xs, prior, 1.0, &mut rng);

    // .. and run it
    dpgmm.run(200, &mut rng);

    // there should be two categories, the first half belong to one category,
    // and the second half belong to the other. Something like
    // [0, 0, 0, 0, ...,0, 1, ..., 1, 1, 1, 1] -- subject to some noise,
    // because we don't actually know how many components there are.
    let mut zs_a = dpgmm.partition.z().clone();
    let zs_b = zs_a.split_off(50);
    println!("{:?}", zs_a);
    println!("{:?}", zs_b);
}
