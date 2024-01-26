use std::ops::RangeInclusive;

use crate::formula::{Formula, Op, BINOPS, UNOPS};

impl<'a> Formula<'a> {
    fn delete_op_range(&mut self, range: RangeInclusive<usize>) {
        // delete the range
        for _ in range.clone() {
            self.operations.remove(*range.start());
        }
    }

    fn random_op_idx(&self, mut f: impl FnMut(Op) -> bool) -> Option<usize> {
        // find all operations that match
        let indices = (0..self.operations.len())
            .filter(|x| f(self.operations[*x]))
            .collect::<Vec<usize>>();

        // select a random one
        fastrand::choice(indices)
    }

    fn operation_range(&self, index: usize) -> RangeInclusive<usize> {
        // where it starts
        let mut start = index;

        // how many ends we need to skip
        let mut ends = 0;

        // work backwards
        while start > 0 {
            // if the current operation is binary, we can skip one end
            if self.operations[start].is_binop() {
                ends += 1;
            }
            // if the operation does not take from the stack, and we can't skip it, stop
            if self.operations[start].is_value() && ends == 0 {
                break;
            }
            // if it is an operation that does not take from the stack, we remove one end
            else if self.operations[start].is_value() {
                ends -= 1;
            }

            // go back one
            start -= 1;
        }

        // return the found range
        start..=index
    }

    /// reblace a 0 with a const
    fn change_zero(&mut self) {
        if let Some(idx) = self.random_op_idx(|x| x == Op::Zero) {
            self.operations[idx] = Op::Const(0.0);
        }
    }

    /// reblace a 1 with a const
    fn change_one(&mut self) {
        if let Some(idx) = self.random_op_idx(|x| x == Op::One) {
            self.operations[idx] = Op::Const(1.0);
        }
    }

    /// reblace a pi with a const
    fn change_pi(&mut self) {
        if let Some(idx) = self.random_op_idx(|x| x == Op::Pi) {
            self.operations[idx] = Op::Const(std::f32::consts::PI);
        }
    }

    /// round a number
    fn round_number(&mut self) {
        if let Some(idx) = self.random_op_idx(Op::is_const) {
            self.operations[idx] = Op::Const(self.operations[idx].get_const_value().round());
        }
    }

    /// modify a constant
    fn modify_number(&mut self) {
        // range to modify
        let power = fastrand::i32(-5..=5);

        // modify it
        if let Some(idx) = self.random_op_idx(Op::is_const) {
            self.operations[idx] = Op::Const(
                self.operations[idx].get_const_value()
                    + fastrand::f32() * (10.0 as f32).powi(power),
            );
        }
    }

    /// replace a constant
    fn replace_const(&mut self) {
        if let Some(idx) = self.random_op_idx(Op::is_const) {
            self.operations[idx] = fastrand::choice([Op::Zero, Op::One, Op::Pi]).unwrap_or(Op::One);
        }
    }

    /// insert a unary operation
    fn insert_unary(&mut self) {
        self.operations.insert(
            fastrand::usize(1..=self.operations.len()),
            fastrand::choice(UNOPS).unwrap_or(UNOPS[0]),
        )
    }

    /// replace a unary operation
    fn replace_unary(&mut self) {
        if let Some(idx) = self.random_op_idx(Op::is_unop) {
            self.operations[idx] = fastrand::choice(UNOPS).unwrap_or(UNOPS[0]);
        }
    }

    /// remove unary operation
    fn remove_unary(&mut self) {
        if let Some(idx) = self.random_op_idx(Op::is_unop) {
            self.operations.remove(idx);
        }
    }

    /// replace a binary operation
    fn replace_binop(&mut self) {
        if let Some(idx) = self.random_op_idx(Op::is_binop) {
            self.operations[idx] = fastrand::choice(BINOPS).unwrap_or(BINOPS[0]);
        }
    }

    /// insert a binary operation with the previous tree on the right side
    fn insert_binop_right(&mut self) {
        let idx = fastrand::usize(1..=self.operations.len());
        let op = fastrand::choice(BINOPS).unwrap_or(BINOPS[0]);
        let val = fastrand::choice(
            [
                Op::Var(fastrand::usize(0..self.names.len())),
                fastrand::choice([Op::Zero, Op::One, Op::Pi, Op::Const(0.0)]).unwrap_or(Op::Zero),
            ]
            .into_iter(),
        )
        .unwrap_or(Op::Zero);

        self.operations.insert(idx, op);
        self.operations.insert(idx, val);
    }

    /// insert a binary operation with the previous tree on the left side
    fn insert_binop_left(&mut self) {
        // TODO: left size, basically find where it starts
        let idx = fastrand::usize(1..=self.operations.len());
        let op = fastrand::choice(BINOPS).unwrap_or(BINOPS[0]);
        let val = fastrand::choice(
            [
                Op::Var(fastrand::usize(0..self.names.len())),
                fastrand::choice([Op::Zero, Op::One, Op::Pi, Op::Const(0.0)]).unwrap_or(Op::Zero),
            ]
            .into_iter(),
        )
        .unwrap_or(Op::Zero);

        self.operations.insert(idx, op);
        self.operations.insert(idx, val);
    }

    /// remove a binop with it's right side
    fn remove_binop_right(&mut self) {
        // remove the operation
        if let Some(idx) = self.random_op_idx(Op::is_binop) {
            self.operations.remove(idx);
            self.delete_op_range(self.operation_range(idx - 1));
        }
    }

    /// remove a binop with it's left side
    fn remove_binop_left(&mut self) {
        // TODO: left
        // remove the operation
        if let Some(idx) = self.random_op_idx(Op::is_binop) {
            self.operations.remove(idx);
            self.delete_op_range(self.operation_range(idx - 1));
        }
    }

    /// swap arguments of a binary operation
    fn swap_args(&mut self) {
        if let Some(idx) = self.random_op_idx(Op::is_non_commutative) {
            // right range
            let right = self.operation_range(idx - 1);

            // left range
            let left = self.operation_range(right.start() - 1);

            // copy the ranges
            let right_ops = &self.operations[right.clone()].to_vec();
            let left_ops = &self.operations[left.clone()].to_vec();

            // stitch them together
            let swapped_ops = right_ops.into_iter().chain(left_ops.into_iter()).copied();

            // write them to the original range
            for (op, new) in self.operations[*(left.start())..=*(right.end())]
                .iter_mut()
                .zip(swapped_ops.into_iter())
            {
                *op = new;
            }
        }
    }

    pub fn mutate(&mut self) {
        fastrand::choice([
            |_: &mut Formula<'a>| {},
            Self::change_zero,
            Self::change_one,
            Self::change_pi,
            Self::round_number,
            Self::modify_number,
            Self::replace_const,
            Self::insert_unary,
            Self::replace_unary,
            Self::remove_unary,
            Self::replace_binop,
            Self::insert_binop_right,
            Self::insert_binop_left,
            Self::remove_binop_right,
            Self::remove_binop_left,
            Self::swap_args,
        ])
        .unwrap_or(|_| {})(self);
    }

    pub fn reproduce(&mut self, other: &Self) {
        // find a chunk for ourselves
        let own_branch = self.operation_range(fastrand::usize(0..self.operations.len()));

        // find the chunk in the other
        let other_branch = other.operation_range(fastrand::usize(0..other.operations.len()));

        // take the other's chunk and replace our own with it
        // first delete our own
        for _ in own_branch.clone() {
            self.operations.remove(*own_branch.start());
        }

        // insert the other
        for i in other_branch.rev() {
            self.operations
                .insert(*own_branch.start(), other.operations[i]);
        }
    }
}
