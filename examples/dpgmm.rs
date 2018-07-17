extern crate rand;
extern crate rv;

use rand::Rng;
use rv::dist::{Crp, Gaussian};
use rv::model::ConjugateModel;
use rv::partition::Partition;
use rv::priors::ng::NormalGamma;
use rv::traits::*;
use rv::utils::ln_pflip;

type GaussComponent<'pr> = ConjugateModel<'pr, f64, Gaussian, NormalGamma>;

// Infinite mixture (CRP) model of univariate Gaussians
struct Dpgmm<'pr> {
    // Keeps track of the data IDs as they're removed and replaced
    xs: Vec<f64>,
    ixs: Vec<usize>,
    crp: Crp,
    partition: Partition,
    prior: &'pr NormalGamma,
    components: Vec<GaussComponent<'pr>>,
}

impl<'pr> Dpgmm<'pr> {
    // Draws a Dpgmm from the prior
    fn new<R: Rng>(
        xs: Vec<f64>,
        prior: &'pr NormalGamma,
        alpha: f64,
        mut rng: &mut R,
    ) -> Self {
        let n = xs.len();
        let crp = Crp::new(alpha, n);
        let partition = crp.draw(&mut rng);
        let mut components: Vec<GaussComponent> = (0..partition.k())
            .map(|_| ConjugateModel::new(&Gaussian::default(), prior))
            .collect();

        xs.iter()
            .zip(partition.z.iter())
            .for_each(|(xi, &zi)| components[zi].observe(xi));

        Dpgmm {
            xs: xs,
            ixs: (0..n).collect(),
            crp: crp,
            partition: partition,
            prior: &prior,
            components: components,
        }
    }

    fn n(&self) -> usize {
        self.xs.len()
    }

    /// Remove and return the datum at index `ix`
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

    fn insert(&mut self, x: f64, ix: usize, k: usize) {
        if k == self.partition.k() {
            let mut c: GaussComponent =
                ConjugateModel::new(&Gaussian::default(), &self.prior);
            c.observe(&x);
            self.components.push(c);
        }
        self.components[k].observe(&x);
        self.xs.push(x);
        self.ixs.push(ix);
        self.partition.append(k).expect("Could not append");
    }

    fn step<R: Rng>(&mut self, pos: usize, mut rng: &mut R) {
        let (x, ix) = self.remove(pos);
        let mut ln_weights: Vec<f64> = self
            .partition
            .counts
            .iter()
            .zip(self.components.iter())
            .map(|(&w, cj)| (w as f64).ln() + cj.ln_pp(&x))
            .collect();

        let ctmp: GaussComponent<'pr> =
            ConjugateModel::new(&Gaussian::default(), &self.prior);

        ln_weights.push(self.crp.alpha.ln() + ctmp.ln_pp(&x));

        let k = ln_pflip(&ln_weights, 1, false, &mut rng)[0];
        self.insert(x, ix, k)
    }

    fn scan<R: Rng>(&mut self, mut rng: &mut R) {
        let mut positions: Vec<usize> = (0..self.n()).collect();
        rng.shuffle(&mut positions);
        positions.iter().for_each(|&pos| self.step(pos, &mut rng));
    }

    fn run<R: Rng>(&mut self, iters: usize, mut rng: &mut R) {
        (0..iters).for_each(|_| self.scan(&mut rng));
    }

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

    let mut xs: Vec<f64> = Gaussian::new(-3.0, 1.0).sample(50, &mut rng);
    let mut ys: Vec<f64> = Gaussian::new(3.0, 1.0).sample(50, &mut rng);
    xs.append(&mut ys);

    let prior = NormalGamma::new(0.0, 1.0, 1.0, 1.0);

    let mut dpgmm = Dpgmm::new(xs, &prior, 1.0, &mut rng);

    dpgmm.run(200, &mut rng);
    dpgmm.sort();

    // there should be two categories, the first half belong to one category,
    // and the second half belong to the other.
    println!("{:?}", dpgmm.partition.z);
}
