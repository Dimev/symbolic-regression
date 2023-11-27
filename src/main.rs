#![feature(portable_simd)]

mod tracer;
mod regression;

use clap::Parser;
use regression::Regressor;
use tracer::Params;

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value_t = 10_000)]
    num_samples: usize,
}



fn main() {
    let args = Args::parse();
    let ins = Params::new(args.num_samples);
    let res = ins.trace();
    let regres = Regressor::new(&[&ins.dot_up], &res.emissive);
}
