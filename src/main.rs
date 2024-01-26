#![feature(portable_simd)]
#![feature(array_chunks)]
#![feature(if_let_guard)]
#![feature(iter_intersperse)]

mod formula;
mod regression;
mod evolve;

use std::{fs::read_to_string, path::PathBuf, time::Instant};

use clap::Parser;
use regression::{Pair, Regressor};

#[derive(Parser)]
struct Args {
    /// Path to the csv file to train on
    /// Last column is the column to use as result
    path: PathBuf,

    /// number of steps to run the evolutionary algorithm for
    #[arg(short, long, default_value_t = 100)]
    iterations: usize,

    /// Penalty to use for formula size, bigger is worse
    #[arg(short, long, default_value_t = 0.5)]
    penalty: f32,

    /// Population size to use for the regression
    #[arg(short, long, default_value_t = 128)]
    size: usize,

    /// Interval at which to report progress in seconds, if at all
    #[arg(long)]
    progress: Option<f32>,

    /// initial formulas, deliminated by commas
    #[arg(short, long)]
    formulas: Option<String>,

    /// Seed for the mutations
    #[arg(long)]
    seed: Option<u64>,

    /// Number of threads to use, num physical CPU's by default
    #[arg(short, long)]
    jobs: Option<usize>,
}

fn main() {
    let args = Args::parse();

    // set the number of threads
    if let Some(jobs) = args.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()
            .expect("Failed to set the number of threads");
    }

    // seed
    if let Some(seed) = args.seed {
        fastrand::seed(seed)
    }

    // load and parse the csv file
    let file = read_to_string(args.path).expect("Failed to read file!");
    let mut csv = file.lines();

    // first line is the names of the parameters
    let names = csv
        .next()
        .expect("No header found in the file!")
        .split(',')
        .collect::<Vec<&str>>();

    // last column is the target to train for
    let (target_name, param_names) = names.split_last().expect("File has less than two columns!");

    // load the rest of the rows
    let mut columns: Vec<Vec<f32>> = std::iter::repeat_with(|| Vec::new())
        .take(names.len())
        .collect();

    for (index, row) in csv.enumerate() {
        // skip empty line
        if row.trim() == "" {
            println!("skipped row {}, as it was empty", index + 2);
            continue;
        }

        // parse the numbers on the row
        let nums = row.split(',').map(|x| {
            x.trim()
                .parse::<f32>()
                .expect(&format!("Could not parse f32 on line {}", index + 2))
        });

        // add them to the row
        for (num, column) in nums.zip(&mut columns) {
            column.push(num);
        }

        // check all columns still have the same number of values
        assert!(
            columns.iter().all(|x| x.len() == columns[0].len()),
            "line {} does not have enough values!",
            index + 2
        );
    }

    println!("{} inputs loaded", columns[0].len());

    // append extra columns so we have a multiple of 64
    let columns = columns
        .iter_mut()
        .map(|x| {
            // new size, padded to multiples of 64
            let size = x
                .len()
                .checked_next_multiple_of(64)
                .expect("Input did not have any rows!");

            // resize the array
            x.resize(size, x[0]);
            x.as_slice()
        })
        .collect::<Vec<&[f32]>>();

    // split off the end
    let (targets, params) = columns
        .split_last()
        .expect("File has less than two columns!");

    // show what we are regressing
    println!(
        "Finding formula f to fit {} = f({}) as best as possible",
        target_name,
        param_names.join(", ")
    );

    // make initial functions
    let funcs = if let Some(formulas) = args.formulas {
        formulas
            .split(',')
            .map(str::to_string)
            .collect::<Vec<String>>()
    } else {
        Vec::new()
    };

    // make regressor
    let mut regres = Regressor::new(
        &param_names,
        params,
        &targets,
        funcs
            .iter()
            .map(|x| x.as_str())
            .collect::<Vec<&str>>()
            .as_slice(),
        args.size,
        args.penalty,
    );

    // start measuring
    let mut last_report = Instant::now();

    // regress
    for i in 0..args.iterations {
        regres.step();

        // print out the current fitness
        if args
            .progress
            .filter(|x| last_report.elapsed().as_secs_f32() > *x)
            .is_some()
        {
            println!(
                "Epoch {}, best score {:.4} with formula `{}`",
                i,
                regres.get_population()[0].score,
                regres.get_population()[0].formula,
            );

            last_report = Instant::now();
        }
    }

    // show the most promising formulas
    println!("Found the following formulas, in reverse polish notation:");
    for (i, Pair { score, formula }) in regres.get_population().iter().enumerate().take(5) {
        println!("{}. Formula `{}` with score {:.4}", i + 1, formula, score);
    }
}
