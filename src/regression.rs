use rayon::prelude::*;

use crate::formula::Formula;

/// Pair for the score and a formula
pub struct Pair<'a> {
    /// How good this formula is
    pub score: f32,

    /// The formula
    pub formula: Formula<'a>,
}

/// Symbolic regressor
pub struct Regressor<'a> {
    /// Input params
    x: Vec<&'a [f32]>,

    /// Outputs
    y: &'a [f32],

    /// How many formulas to keep
    population: usize,

    /// all formulas
    formulas: Vec<Pair<'a>>,

    /// How much to punish large formulas
    size_penalty: f32,
}

impl<'a> Regressor<'a> {
    pub fn new(
        names: &[&'a str],
        inputs: &[&'a [f32]],
        outputs: &'a [f32],
        formulas: &[&'a str],
        population_size: usize,
        size_penalty: f32,
    ) -> Self {
        let formulas = formulas
            .into_iter()
            .map(|x| Pair {
                score: 0.0,
                formula: Formula::from_str(names, x)
                    .expect(&format!("Formula {} failed to parse", x)),
            })
            .chain(std::iter::repeat_with(|| Pair {
                score: 0.0,
                formula: Formula::new(names),
            }))
            .take(population_size)
            .collect();

        Self {
            x: inputs.into_iter().copied().collect(),
            y: outputs,
            population: population_size,
            size_penalty,
            formulas,
        }
    }

    /// get the population
    pub fn get_population(&self) -> &[Pair<'a>] {
        &self.formulas
    }

    /// Run a single step of the regressor
    pub fn step(&mut self) {
        // take the best samples
        self.formulas.truncate((self.population / 10).max(1));

        // where the population ends
        let best_end = self.formulas.len();

        // either mutate or reproduce the best
        while self.formulas.len() < self.population {
            if fastrand::bool() {
                // specimen to mutate
                let mut specimen = fastrand::choice(self.formulas[..best_end].iter())
                    .unwrap_or(&self.formulas[0])
                    .formula
                    .clone();
                specimen.mutate();
                self.formulas.push(Pair {
                    score: 0.0,
                    formula: specimen,
                });
            } else {
                let mut specimen = fastrand::choice(self.formulas[..best_end].iter())
                    .unwrap_or(&self.formulas[0])
                    .formula
                    .clone();

                let other = &fastrand::choice(self.formulas[..best_end].iter())
                    .unwrap_or(&self.formulas[0])
                    .formula;

                specimen.reproduce(&other);

                self.formulas.push(Pair {
                    score: 0.0,
                    formula: specimen,
                });
            }
        }

        // evaluate population
        self.formulas.par_iter_mut().for_each(|pair| {
            pair.score = (pair
                .formula
                .eval_many(self.y.len(), &self.x)
                .into_iter()
                .zip(self.y)
                .map(|(result, target)| (result - target) * (result - target))
                .sum::<f32>()
                / self.y.len() as f32
                + pair.formula.size() * pair.formula.size() * self.size_penalty)
                .min(f32::INFINITY);
        });

        // sort population, best to worst
        self.formulas.sort_by(|l, r| l.score.total_cmp(&r.score));
    }
}
