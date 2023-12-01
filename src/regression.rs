use crate::formula::Formula;

pub struct Regressor<'a> {
	/// Input params
	x: Vec<&'a [f32]>,

	/// Outputs
	y: &'a [f32],

	/// current population of formulas
	population: Vec<Formula<'a>>,
}

impl<'a> Regressor<'a> {
	pub fn new(inputs: &[&'a [f32]], outputs: &'a [f32]) -> Self {
		Self {
			x: inputs.into_iter().copied().collect(),
			y: outputs,
			population: Vec::new(),
		}
	}
}
