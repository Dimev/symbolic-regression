use crate::formula::Formula;

/// Node for monte carlo tree search
struct Node<'a> {
    /// Formula of this node
    formula: Formula<'a>,

    /// fitness of this node (w_i)
    fitness: f32,

    /// number of times this node has been checked (n_i)
    checked: usize,

    /// number of times this branch has been checked (N_i)
    checked_total: usize,

    /// child nodes
    children: Vec<usize>,

    /// parent node
    parent: usize,
}

/// Symbolic regressor
pub struct Regressor<'a> {
    /// Input params
    x: Vec<&'a [f32]>,

    /// Outputs
    y: &'a [f32],

    /// all nodes
    nodes: Vec<Node<'a>>,
}

impl<'a> Regressor<'a> {
    pub fn new(names: &[&str], inputs: &[&'a [f32]], outputs: &'a [f32]) -> Self {
        Self {
            x: inputs.into_iter().copied().collect(),
            y: outputs,
            nodes: vec![Node {
                formula: Formula::new(names),
                children: Vec::new(),
                fitness: 0.0,
                checked: 0,
                checked_total: 0,
                parent: 0,
            }],
        }
    }

    /// Run a single step of the regressor
    pub fn step(&mut self) {
        // find the best node that has not been explored yet
        let mut cur = 0;
        while self.nodes[cur].children.len() > 0 {
            // get the one with the highest score
            cur = self.nodes[cur]
                .children
                .iter()
                .copied()
                .max_by_key(|x| 1)
                .unwrap();
        }

        // Generate mutations
        let mutations = self.nodes[cur].formula.mutate(fastrand::u64(..));

        // Apply gradient descent

        // evaluate the formulas
        let fitness = 0.0;

        // propagate results back
        while cur != 0 {
            // propagate results
            // TODO: just pick the lowest of fitness * times simulated?
            // fitness is the *lowest* fitness encountered in the entire tree
            // fitness here is how big the error is

            // go up one node
            cur = self.nodes[cur].parent;
        }
    }
}
