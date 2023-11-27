pub struct Regressor<'a> {
	/// Input params
	x: Vec<&'a [f32]>,

	/// Outputs
	y: &'a [f32],
}

impl<'a> Regressor<'a> {
	pub fn new(inputs: &[&'a [f32]], outputs: &'a [f32]) -> Self {
		Self {
			x: inputs.into_iter().copied().collect(),
			y: outputs,
		}
	}
}
