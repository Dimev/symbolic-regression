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
        population_size: usize,
        size_penalty: f32,
    ) -> Self {
        Self {
            x: inputs.into_iter().copied().collect(),
            y: outputs,
            population: population_size,
            size_penalty,
            formulas: vec![Pair {
                score: 0.0,
                formula: Formula::new(names),
            }],
        }
    }

    /// get the population
    pub fn get_population(&self) -> &[Pair<'a>] {
        &self.formulas
    }

    /// Run a single step of the regressor
    pub fn step(&mut self) {
        // generate new mutations
        let new_formulas = self
            .formulas
            .iter()
            .flat_map(|x| x.formula.mutate().into_iter())
            .collect::<Vec<Formula>>();

        // evaluate new population
        self.formulas = new_formulas
            .into_par_iter()
            .filter_map(|formula| {
                let score = formula
                    .eval_many(self.y.len(), &self.x)
                    .into_iter()
                    .zip(self.y)
                    .map(|(result, target)| (result - target) * (result - target))
                    .sum::<f32>()
                    / self.y.len() as f32
                    + formula.size() * formula.size() * self.size_penalty;

                // filter it out if it results in a nan
                if score.is_nan() {
                    None
                } else {
                    Some(Pair { score, formula })
                }
            })
            .collect::<Vec<Pair>>();

        // sort population, best to worst
        self.formulas.sort_by(|l, r| l.score.total_cmp(&r.score));

        // prune population size
        self.formulas.truncate(self.population);
    }
}
