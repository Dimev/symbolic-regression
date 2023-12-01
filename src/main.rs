#![feature(portable_simd)]
#![feature(array_chunks)]
#![feature(if_let_guard)]
#![feature(iter_intersperse)]

mod tracer;
mod regression;
mod formula;

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
