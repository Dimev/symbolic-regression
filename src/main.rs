#![feature(portable_simd)]
#![feature(array_chunks)]
#![feature(if_let_guard)]
#![feature(iter_intersperse)]

pub mod formula;
mod regression;
mod tracer;

use clap::Parser;
use regression::{Pair, Regressor};
use tracer::Params;

#[derive(Parser)]
struct Args {
    /// Number of samples to take for the path tracer
    #[arg(short, long, default_value_t = 1024)]
    rays: usize,

    /// Population size for the evolutionary algorithm
    #[arg(short, long, default_value_t = 10_000)]
    size: usize,

    /// number of steps to run the evolutionary algorithm for
    #[arg(short, long, default_value_t = 100)]
    iterations: usize,

    /// Penalty to use for formula size, bigger is worse
    #[arg(short, long, default_value_t = 0.5)]
    penalty: f32,
}

fn main() {
    let args = Args::parse();
    let ins = Params::new(args.rays);
    let res = ins.trace();
    let mut regres = Regressor::new(
        &["dot-up"],
        &[&ins.dot_up],
        &res.emissive,
        args.size,
        args.penalty,
    );

    for i in 0..args.iterations {
        // regress
        regres.step();

        // print out the current fitness
        println!(
            "Epoch {}, Best is {} with formula {}",
            i,
            regres.get_population()[0].score,
            regres.get_population()[0].formula
        );
    }

    // show the most promising formulas
    for Pair { score, formula } in regres.get_population().iter().take(5) {
        println!("Score: {:.4}, Formula: {}", score, formula);
    }
}
