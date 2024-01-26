use std::{borrow::Cow, fmt::Display, ops::RangeInclusive, simd::f32x64};

/// An operation in a formula
#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum Op {
    /// Read a variable
    Var(usize),

    /// Push a constant
    Const(f32),

    /// push constant Pi
    Pi,

    /// Push constant 1
    One,

    /// Push constant 0
    Zero,

    /// Addition
    Add,

    /// Subtraction
    Sub,

    /// Multiplication
    Mul,

    /// Division
    Div,

    /// Negation
    Neg,

    /// Power
    Pow,

    /// Exponential
    Exp,

    /// Root
    Sqrt,

    /// Logarithm
    Log,

    /// sin
    Sin,

    /// Cos
    Cos,

    /// Tan
    Tan,
}

impl Op {
    /// is this a const?
    fn is_const(self) -> bool {
        match self {
            Op::Const(_) => true,
            _ => false,
        }
    }

    // is this a value (pushes on the stack)
    fn is_value(self) -> bool {
        match self {
            Op::Var(_) | Op::Const(_) | Op::One | Op::Zero | Op::Pi => true,
            _ => false,
        }
    }

    /// is this a unop?
    fn is_unop(self) -> bool {
        UNOPS.contains(&self)
    }

    /// is this a binop
    fn is_binop(self) -> bool {
        BINOPS.contains(&self)
    }

    /// Get a const value, or 0 if nothing
    fn get_const_value(self) -> f32 {
        match self {
            Op::Const(v) => v,
            _ => 0.0,
        }
    }

    /// get a non-commutative binary operation
    fn is_non_commutative(self) -> bool {
        [Op::Pow, Op::Sub, Op::Div].contains(&self)
    }
}

/// all unary operations
const UNOPS: [Op; 7] = [
    Op::Neg,
    Op::Exp,
    Op::Sqrt,
    Op::Log,
    Op::Sin,
    Op::Cos,
    Op::Tan,
];

/// All binary operations
const BINOPS: [Op; 5] = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Pow];

/// A single formula
#[derive(Clone)]
pub struct Formula<'a> {
    operations: Vec<Op>,
    names: Vec<&'a str>,
}

impl<'a> Display for Formula<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.operations
                .iter()
                .map(|x| match x {
                    Op::Var(n) => self.names[*n].to_string(),
                    Op::Const(x) => format!("{x}"),
                    Op::Pi => "pi".to_string(),
                    Op::One => "1".to_string(),
                    Op::Zero => "0".to_string(),
                    Op::Add => "+".to_string(),
                    Op::Sub => "-".to_string(),
                    Op::Mul => "*".to_string(),
                    Op::Div => "/".to_string(),
                    Op::Neg => "neg".to_string(),
                    Op::Pow => "pow".to_string(),
                    Op::Exp => "exp".to_string(),
                    Op::Sqrt => "sqrt".to_string(),
                    Op::Log => "log".to_string(),
                    Op::Sin => "sin".to_string(),
                    Op::Cos => "cos".to_string(),
                    Op::Tan => "tan".to_string(),
                })
                .intersperse(" ".to_string())
                .collect::<String>()
        )
    }
}

impl<'a> Formula<'a> {
    pub fn from_str(names: &[&'a str], s: &str) -> Option<Self> {
        let ops = s
            .split_whitespace()
            .map(|w| match w {
                "pi" => Op::Pi,
                "1" => Op::One,
                "0" => Op::Zero,
                "+" => Op::Add,
                "-" => Op::Sub,
                "*" => Op::Mul,
                "/" => Op::Div,
                "neg" => Op::Neg,
                "pow" => Op::Pow,
                "exp" => Op::Exp,
                "sqrt" => Op::Sqrt,
                "log" => Op::Log,
                "sin" => Op::Sin,
                "cos" => Op::Cos,
                "tan" => Op::Tan,
                x if let Ok(n) = x.parse() => Op::Const(n),
                x if let Some(n) = names.iter().position(|n| n == &x) => Op::Var(n),
                x => panic!("Unexpected token {x}"),
            })
            .collect::<Vec<Op>>();

        // check if it's valid
        assert!(
            ops.iter().fold(0 as isize, |acc, op| if op.is_binop() {
                acc - 1
            } else if op.is_value() {
                acc + 1
            } else {
                acc
            }) == 1,
            "Function does not return 1 value on the stack!"
        );

        Some(Self {
            names: names.to_vec(),
            operations: ops,
        })
    }

    pub fn new(names: &[&'a str]) -> Self {
        Self {
            names: names.to_vec(),
            operations: std::iter::once(Op::Zero)
                .chain(
                    (0..names.len()).flat_map(|idx| {
                        [Op::Var(idx), Op::Const(1.0), Op::Mul, Op::Add].into_iter()
                    }),
                )
                .collect(),
        }
    }

    pub fn size(&self) -> f32 {
        self.operations.len() as f32
    }

    fn replace_op(&self, index: usize, op: Op) -> Self {
        let mut clone = self.clone();
        clone.operations[index] = op;
        clone
    }

    fn insert_op(&self, index: usize, op: Op) -> Self {
        let mut clone = self.clone();
        clone.operations.insert(index, op);
        clone
    }

    fn delete_op(&self, index: usize) -> Self {
        let mut clone = self.clone();
        clone.operations.remove(index);
        clone
    }

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
    fn replace_binary(&mut self) {
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

    pub fn mutate(&self) -> Vec<Self> {
        // all mutations below are applied on one of the operations in the formula
        let mut mutations = Vec::with_capacity(22);

        mutations
    }

    pub fn eval(&self, args: &[f32]) -> f32 {
        assert_eq!(args.len(), self.names.len());
        let mut stack = Vec::with_capacity(self.operations.len());

        for op in &self.operations {
            match op {
                Op::Var(n) => stack.push(args[*n]),
                Op::Const(x) => stack.push(*x),
                Op::Pi => stack.push(std::f32::consts::PI),
                Op::One => stack.push(1.0),
                Op::Zero => stack.push(0.0),
                Op::Add => {
                    let r = stack.pop().unwrap();
                    let l = stack.last_mut().unwrap();
                    *l = *l + r;
                }
                Op::Sub => {
                    let r = stack.pop().unwrap();
                    let l = stack.last_mut().unwrap();
                    *l = *l - r;
                }
                Op::Mul => {
                    let r = stack.pop().unwrap();
                    let l = stack.last_mut().unwrap();
                    *l = *l * r;
                }
                Op::Div => {
                    let r = stack.pop().unwrap();
                    let l = stack.last_mut().unwrap();
                    *l = *l / r;
                }
                Op::Pow => {
                    let r = stack.pop().unwrap();
                    let l = stack.last_mut().unwrap();
                    *l = l.powf(r);
                }
                Op::Neg => {
                    let l = stack.last_mut().unwrap();
                    *l = -*l;
                }
                Op::Exp => {
                    let l = stack.last_mut().unwrap();
                    *l = l.exp();
                }
                Op::Sqrt => {
                    let l = stack.last_mut().unwrap();
                    *l = l.sqrt();
                }
                Op::Log => {
                    let l = stack.last_mut().unwrap();
                    *l = l.log10();
                }
                Op::Sin => {
                    let l = stack.last_mut().unwrap();
                    *l = l.sin();
                }
                Op::Cos => {
                    let l = stack.last_mut().unwrap();
                    *l = l.cos();
                }
                Op::Tan => {
                    let l = stack.last_mut().unwrap();
                    *l = l.tan();
                }
            }
        }

        assert!(stack.len() == 1);
        stack[0]
    }

    pub fn eval_many(&self, n: usize, args: &[&[f32]]) -> Vec<f32> {
        assert_eq!(
            args.len(),
            self.names.len(),
            "Number of arguments does not match the number of bound names in the function"
        );
        assert!(
            args.iter().all(|x| x.len() == n),
            "Arguments are not the same lenght"
        );
        assert!(
            n % 64 == 0 && n > 0,
            "Argument count is not a multiple of 64"
        );

        let mut stack = Vec::with_capacity(self.operations.len());

        for op in &self.operations {
            match op {
                Op::Var(n) => stack.push(Cow::from(args[*n])),
                Op::Const(x) => stack.push(Cow::from_iter(std::iter::repeat(*x).take(n))),
                Op::Pi => stack.push(Cow::from_iter(
                    std::iter::repeat(std::f32::consts::PI).take(n),
                )),
                Op::One => stack.push(Cow::from_iter(std::iter::repeat(1.0_f32).take(n))),
                Op::Zero => stack.push(Cow::from_iter(std::iter::repeat(0.0_f32).take(n))),
                Op::Add => {
                    let r = stack.pop().unwrap();
                    let l = stack.last_mut().unwrap();
                    let rt = r.array_chunks().map(|x| f32x64::from_array(*x));
                    l.to_mut().array_chunks_mut().zip(rt).for_each(|(l, r)| {
                        (f32x64::from_array(*l) + r)
                            .to_array()
                            .iter()
                            .zip(0..)
                            .for_each(|(x, i)| l[i] = *x)
                    });
                }
                Op::Sub => {
                    let r = stack.pop().unwrap();
                    let l = stack.last_mut().unwrap();
                    let rt = r.array_chunks().map(|x| f32x64::from_array(*x));
                    l.to_mut().array_chunks_mut().zip(rt).for_each(|(l, r)| {
                        (f32x64::from_array(*l) - r)
                            .to_array()
                            .iter()
                            .zip(0..)
                            .for_each(|(x, i)| l[i] = *x)
                    });
                }
                Op::Mul => {
                    let r = stack.pop().unwrap();
                    let l = stack.last_mut().unwrap();
                    let rt = r.array_chunks().map(|x| f32x64::from_array(*x));
                    l.to_mut().array_chunks_mut().zip(rt).for_each(|(l, r)| {
                        (f32x64::from_array(*l) * r)
                            .to_array()
                            .iter()
                            .zip(0..)
                            .for_each(|(x, i)| l[i] = *x)
                    });
                }
                Op::Div => {
                    let r = stack.pop().unwrap();
                    let l = stack.last_mut().unwrap();
                    let rt = r.array_chunks().map(|x| f32x64::from_array(*x));
                    l.to_mut().array_chunks_mut().zip(rt).for_each(|(l, r)| {
                        (f32x64::from_array(*l) / r)
                            .to_array()
                            .iter()
                            .zip(0..)
                            .for_each(|(x, i)| l[i] = *x)
                    });
                }
                Op::Pow => {
                    let r = stack.pop().unwrap();
                    let l = stack.last_mut().unwrap();
                    l.to_mut()
                        .iter_mut()
                        .zip(r.iter())
                        .for_each(|(l, r)| *l = l.powf(*r));
                }
                Op::Neg => {
                    let l = stack.last_mut().unwrap();
                    l.to_mut().iter_mut().for_each(|x| *x = -*x);
                }
                Op::Exp => {
                    let l = stack.last_mut().unwrap();
                    l.to_mut().iter_mut().for_each(|x| *x = x.exp());
                }
                Op::Sqrt => {
                    let l = stack.last_mut().unwrap();
                    l.to_mut().iter_mut().for_each(|x| *x = x.sqrt());
                }
                Op::Log => {
                    let l = stack.last_mut().unwrap();
                    l.to_mut().iter_mut().for_each(|x| *x = x.log10());
                }
                Op::Sin => {
                    let l = stack.last_mut().unwrap();
                    l.to_mut().iter_mut().for_each(|x| *x = x.sin());
                }
                Op::Cos => {
                    let l = stack.last_mut().unwrap();
                    l.to_mut().iter_mut().for_each(|x| *x = x.cos());
                }
                Op::Tan => {
                    let l = stack.last_mut().unwrap();
                    l.to_mut().iter_mut().for_each(|x| *x = x.tan());
                }
            }
        }

        assert!(stack.len() == 1);
        stack[0].to_vec()
    }
}

#[test]
fn eval_1() {
    let f = Formula::from_str(&[], "1 1 +").unwrap();
    assert_eq!(f.eval(&[]), 2.0);
}

#[test]
fn eval_2() {
    let f = Formula::from_str(&[], "0 cos 1 +").unwrap();
    assert_eq!(f.eval(&[]), 2.0);
}

#[test]
fn eval_3() {
    let f = Formula::from_str(&[], "0 exp 1 *").unwrap();
    assert_eq!(f.eval(&[]), 1.0);
}

#[test]
fn eval_4() {
    let f = Formula::from_str(&[], "pi 0 +").unwrap();
    assert_eq!(f.eval(&[]), std::f32::consts::PI);
}

#[test]
fn eval_5() {
    let f = Formula::from_str(&[], "10 log 2 /").unwrap();
    assert_eq!(f.eval(&[]), 0.5);
}

#[test]
fn eval_6() {
    let f = Formula::from_str(&[], "3 2 pow log 4 +").unwrap();
    assert_eq!(f.eval(&[]), 3.0_f32.powf(2.0).log10() + 4.0);
}

#[test]
fn eval_7() {
    let f = Formula::from_str(&[], "6 2 / 3 exp 4 * -").unwrap();
    assert_eq!(f.eval(&[]), (6.0 / 2.0) - (3.0_f32.exp() * 4.0));
}

#[test]
fn eval_8() {
    let f = Formula::from_str(&[], "1 neg").unwrap();
    assert_eq!(f.eval(&[]), -1.0);
}

#[test]
fn eval_names() {
    let f = Formula::from_str(&["x1", "b"], "x1 b +").unwrap();
    assert_eq!(f.eval(&[1.0, 2.0]), 3.0);
}

#[test]
fn print_formula() {
    let f = Formula::from_str(&["x1", "x2", "x'"], "x1 x2 + x' *").unwrap();
    assert_eq!(format!("{f}"), "x1 x2 + x' *");
}

#[test]
fn eval_simd() {
    let f = Formula::from_str(&["x1", "b", "U"], "x1 b + U -").unwrap();
    assert_eq!(
        f.eval_many(512, &[&[1.0; 512], &[2.0; 512], &[3.0; 512]]),
        vec![0.0; 512]
    );
}

#[test]
fn eval_new() {
    let f = Formula::new(&["x1", "x2", "x3"]);
    assert_eq!(f.eval(&[1.0, 2.0, 3.0]), 6.0);
}

#[test]
fn eval_simd_a_lot() {
    const N: usize = 10_000_000;

    let f = Formula::from_str(
        &["x1", "x2", "x3", "x4", "x5"],
        "x1 x2 + x3 - x4 * x5 / exp log sqrt neg sin",
    )
    .unwrap();

    let x1 = vec![1.0; N];
    let x2 = vec![2.0; N];
    let x3 = vec![3.0; N];
    let x4 = vec![4.0; N];
    let x5 = vec![5.0; N];

    assert_eq!(f.eval_many(N, &[&x1, &x2, &x3, &x4, &x5]).len(), N);
}
