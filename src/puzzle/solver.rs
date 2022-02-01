use std::ops::{Add, BitAnd, BitOr, BitXor, Bound, Not, RangeBounds, Sub};

use crate::csp::{Assignment, Domain};
use crate::csp_repr::BoolExpr as CSPBoolExpr;
use crate::csp_repr::BoolVar as CSPBoolVar;
use crate::csp_repr::IntExpr as CSPIntExpr;
use crate::csp_repr::IntVar as CSPIntVar;
use crate::integration::IntegratedSolver;
use crate::integration::Model as IntegratedModel;

pub trait Convertible<T> {
    fn convert(self) -> T;
}

impl<T> Convertible<T> for T {
    fn convert(self) -> T {
        self
    }
}

impl Convertible<CSPBoolExpr> for bool {
    fn convert(self) -> CSPBoolExpr {
        CSPBoolExpr::Const(self)
    }
}

impl Convertible<CSPBoolExpr> for CSPBoolVar {
    fn convert(self) -> CSPBoolExpr {
        self.expr()
    }
}

impl Convertible<CSPIntExpr> for i32 {
    fn convert(self) -> CSPIntExpr {
        CSPIntExpr::Const(self)
    }
}

impl Convertible<CSPIntExpr> for CSPIntVar {
    fn convert(self) -> CSPIntExpr {
        self.expr()
    }
}

#[derive(Clone)]
pub struct Array0DImpl<T> {
    data: T,
}

impl<T> Convertible<T> for Array0DImpl<T> {
    fn convert(self) -> T {
        self.data.convert()
    }
}

#[derive(Clone)]
pub struct Array1DImpl<T> {
    data: Vec<T>,
}

#[derive(Clone)]
pub struct Array2DImpl<T> {
    shape: (usize, usize),
    data: Vec<T>,
}

impl<T> IntoIterator for Array0DImpl<T> {
    type Item = T;
    type IntoIter = std::iter::Once<T>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self.data)
    }
}

impl<T> IntoIterator for Array1DImpl<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<T> IntoIterator for Array2DImpl<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

pub trait Unary<X, T> {
    type Output;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X) -> T;
}

impl<A, X, T> Unary<X, T> for Array0DImpl<A>
where
    A: Convertible<X>,
{
    type Output = Array0DImpl<T>;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X) -> T,
    {
        Array0DImpl {
            data: func(self.data.convert()),
        }
    }
}

impl<A, X, T> Unary<X, T> for Array1DImpl<A>
where
    A: Convertible<X>,
{
    type Output = Array1DImpl<T>;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X) -> T,
    {
        Array1DImpl {
            data: self.data.into_iter().map(|x| func(x.convert())).collect(),
        }
    }
}

impl<A, X, T> Unary<X, T> for Array2DImpl<A>
where
    A: Convertible<X>,
{
    type Output = Array2DImpl<T>;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X) -> T,
    {
        Array2DImpl {
            shape: self.shape,
            data: self.data.into_iter().map(|x| func(x.convert())).collect(),
        }
    }
}

pub trait PropagateBinary<X, Y, T> {
    type Output;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X, Y) -> T;
}

impl<A, B, X, Y, T> PropagateBinary<X, Y, T> for (Array0DImpl<A>, Array0DImpl<B>)
where
    A: Convertible<X>,
    B: Convertible<Y>,
{
    type Output = Array0DImpl<T>;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X, Y) -> T,
    {
        Array0DImpl {
            data: func(self.0.data.convert(), self.1.data.convert()),
        }
    }
}

impl<A, B, X, Y, T> PropagateBinary<X, Y, T> for (Array1DImpl<A>, Array0DImpl<B>)
where
    A: Convertible<X>,
    B: Convertible<Y>,
    Y: Clone,
{
    type Output = Array1DImpl<T>;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X, Y) -> T,
    {
        let rhs = self.1.data.convert();
        Array1DImpl {
            data: self
                .0
                .data
                .into_iter()
                .map(|lhs| func(lhs.convert(), rhs.clone()))
                .collect(),
        }
    }
}

impl<A, B, X, Y, T> PropagateBinary<X, Y, T> for (Array0DImpl<A>, Array1DImpl<B>)
where
    A: Convertible<X>,
    B: Convertible<Y>,
    X: Clone,
{
    type Output = Array1DImpl<T>;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X, Y) -> T,
    {
        let lhs = self.0.data.convert();
        Array1DImpl {
            data: self
                .1
                .data
                .into_iter()
                .map(|rhs| func(lhs.clone(), rhs.convert()))
                .collect(),
        }
    }
}

impl<A, B, X, Y, T> PropagateBinary<X, Y, T> for (Array1DImpl<A>, Array1DImpl<B>)
where
    A: Convertible<X>,
    B: Convertible<Y>,
{
    type Output = Array1DImpl<T>;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X, Y) -> T,
    {
        Array1DImpl {
            data: self
                .0
                .data
                .into_iter()
                .zip(self.1.data.into_iter())
                .map(|(lhs, rhs)| func(lhs.convert(), rhs.convert()))
                .collect(),
        }
    }
}

impl<A, B, X, Y, T> PropagateBinary<X, Y, T> for (Array2DImpl<A>, Array0DImpl<B>)
where
    A: Convertible<X>,
    B: Convertible<Y>,
    Y: Clone,
{
    type Output = Array2DImpl<T>;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X, Y) -> T,
    {
        let rhs = self.1.data.convert();
        Array2DImpl {
            shape: self.0.shape,
            data: self
                .0
                .data
                .into_iter()
                .map(|lhs| func(lhs.convert(), rhs.clone()))
                .collect(),
        }
    }
}

impl<A, B, X, Y, T> PropagateBinary<X, Y, T> for (Array0DImpl<A>, Array2DImpl<B>)
where
    A: Convertible<X>,
    B: Convertible<Y>,
    X: Clone,
{
    type Output = Array2DImpl<T>;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X, Y) -> T,
    {
        let lhs = self.0.data.convert();
        Array2DImpl {
            shape: self.1.shape,
            data: self
                .1
                .data
                .into_iter()
                .map(|rhs| func(lhs.clone(), rhs.convert()))
                .collect(),
        }
    }
}

impl<A, B, X, Y, T> PropagateBinary<X, Y, T> for (Array2DImpl<A>, Array2DImpl<B>)
where
    A: Convertible<X>,
    B: Convertible<Y>,
{
    type Output = Array2DImpl<T>;

    fn generate<F>(self, func: F) -> Self::Output
    where
        F: Fn(X, Y) -> T,
    {
        assert_eq!(self.0.shape, self.1.shape);
        Array2DImpl {
            shape: self.0.shape,
            data: self
                .0
                .data
                .into_iter()
                .zip(self.1.data.into_iter())
                .map(|(lhs, rhs)| func(lhs.convert(), rhs.convert()))
                .collect(),
        }
    }
}

#[derive(Clone)]
pub struct Value<T>(T);

impl<T> IntoIterator for Value<T>
where
    T: IntoIterator,
{
    type Item = <T as IntoIterator>::Item;
    type IntoIter = <T as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T> IntoIterator for &Value<T>
where
    T: Clone + IntoIterator,
{
    type Item = <T as IntoIterator>::Item;
    type IntoIter = <T as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.clone().into_iter()
    }
}

pub trait IntoValue {
    type Output;

    fn into_value(self) -> Value<Self::Output>;
}

impl<T> IntoValue for Value<T> {
    type Output = T;

    fn into_value(self) -> Value<Self::Output> {
        self
    }
}

impl IntoValue for i32 {
    type Output = Array0DImpl<CSPIntExpr>;

    fn into_value(self) -> Value<Self::Output> {
        Value(Array0DImpl {
            data: self.convert(),
        })
    }
}

impl<X> Not for Value<X>
where
    X: Unary<CSPBoolExpr, CSPBoolExpr>,
{
    type Output = Value<<X as Unary<CSPBoolExpr, CSPBoolExpr>>::Output>;

    fn not(self) -> Self::Output {
        Value(self.0.generate(|x| !x))
    }
}

macro_rules! binary_op {
    ($trait_name:ident, $trait_func:ident, $input_type:ty, $output_type:ty, $gen:expr) => {
        impl<X, Y> $trait_name<Value<Y>> for Value<X>
        where
            (X, Y): PropagateBinary<$input_type, $input_type, $output_type>,
        {
            type Output =
                Value<<(X, Y) as PropagateBinary<$input_type, $input_type, $output_type>>::Output>;

            fn $trait_func(self, rhs: Value<Y>) -> Self::Output {
                Value((self.0, rhs.0).generate($gen))
            }
        }

        impl<X, Y> $trait_name<Value<Y>> for &Value<X>
        where
            (X, Y): PropagateBinary<$input_type, $input_type, $output_type>,
            X: Clone,
        {
            type Output =
                Value<<(X, Y) as PropagateBinary<$input_type, $input_type, $output_type>>::Output>;

            fn $trait_func(self, rhs: Value<Y>) -> Self::Output {
                Value((self.0.clone(), rhs.0).generate($gen))
            }
        }

        impl<X, Y> $trait_name<&Value<Y>> for Value<X>
        where
            (X, Y): PropagateBinary<$input_type, $input_type, $output_type>,
            Y: Clone,
        {
            type Output =
                Value<<(X, Y) as PropagateBinary<$input_type, $input_type, $output_type>>::Output>;

            fn $trait_func(self, rhs: &Value<Y>) -> Self::Output {
                Value((self.0, rhs.0.clone()).generate($gen))
            }
        }

        impl<X, Y> $trait_name<&Value<Y>> for &Value<X>
        where
            (X, Y): PropagateBinary<$input_type, $input_type, $output_type>,
            X: Clone,
            Y: Clone,
        {
            type Output =
                Value<<(X, Y) as PropagateBinary<$input_type, $input_type, $output_type>>::Output>;

            fn $trait_func(self, rhs: &Value<Y>) -> Self::Output {
                Value((self.0.clone(), rhs.0.clone()).generate($gen))
            }
        }
    };
}

binary_op!(BitAnd, bitand, CSPBoolExpr, CSPBoolExpr, |x, y| x & y);
binary_op!(BitOr, bitor, CSPBoolExpr, CSPBoolExpr, |x, y| x | y);
binary_op!(BitXor, bitxor, CSPBoolExpr, CSPBoolExpr, |x, y| x ^ y);
binary_op!(Add, add, CSPIntExpr, CSPIntExpr, |x, y| x + y);
binary_op!(Sub, sub, CSPIntExpr, CSPIntExpr, |x, y| x - y);

macro_rules! comparator {
    ($func_name:ident) => {
        impl<X> Value<X> {
            pub fn $func_name<Y>(
                self,
                rhs: Y,
            ) -> Value<
                <(X, <Y as IntoValue>::Output) as PropagateBinary<
                    CSPIntExpr,
                    CSPIntExpr,
                    CSPBoolExpr,
                >>::Output,
            >
            where
                Y: IntoValue,
                (X, <Y as IntoValue>::Output): PropagateBinary<CSPIntExpr, CSPIntExpr, CSPBoolExpr>,
            {
                Value((self.0, rhs.into_value().0).generate(|x, y| x.$func_name(y)))
            }
        }
    };
}

comparator!(eq);
comparator!(ne);
comparator!(ge);
comparator!(gt);
comparator!(le);
comparator!(lt);

impl<T> Value<T>
where
    T: Clone + IntoIterator,
    <T as IntoIterator>::Item: Convertible<CSPBoolExpr>,
{
    pub fn count_true(&self) -> Value<Array0DImpl<CSPIntExpr>> {
        let terms = self
            .0
            .clone()
            .into_iter()
            .map(|e| {
                (
                    Box::new(e.convert().ite(CSPIntExpr::Const(1), CSPIntExpr::Const(0))),
                    1,
                )
            })
            .collect::<Vec<_>>();
        Value(Array0DImpl {
            data: CSPIntExpr::Linear(terms),
        })
    }

    pub fn all(&self) -> Value<Array0DImpl<CSPBoolExpr>> {
        let terms = self
            .0
            .clone()
            .into_iter()
            .map(|e| Box::new(e.convert()))
            .collect::<Vec<_>>();
        Value(Array0DImpl {
            data: CSPBoolExpr::And(terms),
        })
    }

    pub fn any(&self) -> Value<Array0DImpl<CSPBoolExpr>> {
        let terms = self
            .0
            .clone()
            .into_iter()
            .map(|e| Box::new(e.convert()))
            .collect::<Vec<_>>();
        Value(Array0DImpl {
            data: CSPBoolExpr::Or(terms),
        })
    }
}

impl<T> Value<T>
where
    T: Clone + IntoIterator,
    <T as IntoIterator>::Item: Convertible<CSPIntExpr>,
{
    pub fn sum(&self) -> Value<Array0DImpl<CSPIntExpr>> {
        let terms = self
            .0
            .clone()
            .into_iter()
            .map(|e| (Box::new(e.convert()), 1))
            .collect::<Vec<_>>();
        Value(Array0DImpl {
            data: CSPIntExpr::Linear(terms),
        })
    }
}

impl<T: Clone> Value<Array1DImpl<T>> {
    pub fn at(&self, idx: usize) -> Value<Array0DImpl<T>> {
        Value(Array0DImpl {
            data: self.0.data[idx].clone(),
        })
    }
}

fn resolve_range<T: RangeBounds<usize>>(len: usize, range: &T) -> (usize, usize) {
    let start = match range.start_bound() {
        Bound::Unbounded => 0,
        Bound::Included(&x) => x,
        Bound::Excluded(&x) => x + 1,
    };
    let end = match range.end_bound() {
        Bound::Unbounded => len,
        Bound::Included(&x) => x + 1,
        Bound::Excluded(&x) => x,
    };
    if start >= end {
        (0, 0)
    } else {
        (start, end)
    }
}

impl<T: Clone> Value<Array2DImpl<T>> {
    pub fn at(&self, idx: (usize, usize)) -> Value<Array0DImpl<T>> {
        let (y, x) = idx;
        let (h, w) = self.0.shape;
        assert!(y < h && x < w);
        Value(Array0DImpl {
            data: self.0.data[y * w + x].clone(),
        })
    }

    pub fn select<I, X>(&self, idx: I) -> Value<Array1DImpl<T>>
    where
        I: IntoIterator<Item = X>,
        X: std::borrow::Borrow<(usize, usize)>,
    {
        let mut data = vec![];
        let (h, w) = self.0.shape;
        for p in idx {
            let &(y, x) = p.borrow();
            assert!(y < h && x < w);
            data.push(self.0.data[y * w + x].clone());
        }
        Value(Array1DImpl { data })
    }

    pub fn slice_fixed_y<X: RangeBounds<usize>>(&self, idx: (usize, X)) -> Value<Array1DImpl<T>> {
        let (y, xs) = idx;
        let (_, w) = self.0.shape;
        let (x_start, x_end) = resolve_range(w, &xs);

        let items = (x_start..x_end).map(|x| self.at((y, x)).0.data).collect();
        Value(Array1DImpl { data: items })
    }

    pub fn slice_fixed_x<Y: RangeBounds<usize>>(&self, idx: (Y, usize)) -> Value<Array1DImpl<T>> {
        let (ys, x) = idx;
        let (h, _) = self.0.shape;
        let (y_start, y_end) = resolve_range(h, &ys);

        let items = (y_start..y_end).map(|y| self.at((y, x)).0.data).collect();
        Value(Array1DImpl { data: items })
    }

    pub fn slice<Y: RangeBounds<usize>, X: RangeBounds<usize>>(
        &self,
        idx: (Y, X),
    ) -> Value<Array2DImpl<T>> {
        let (ys, xs) = idx;
        let (h, w) = self.0.shape;
        let (y_start, y_end) = resolve_range(h, &ys); // [y_start, y_end)
        let (x_start, x_end) = resolve_range(w, &xs); // [x_start, x_end)

        let slice_shape = (y_end - y_start, x_end - x_start);
        let mut items = vec![];
        for y in y_start..y_end {
            for x in x_start..x_end {
                items.push(self.0.data[y * w + x].clone());
            }
        }
        Value(Array2DImpl {
            shape: slice_shape,
            data: items,
        })
    }
}

pub type BoolVar = Value<Array0DImpl<CSPBoolVar>>;
pub type BoolVarArray1D = Value<Array1DImpl<CSPBoolVar>>;
pub type BoolVarArray2D = Value<Array2DImpl<CSPBoolVar>>;
pub type BoolExpr = Value<Array0DImpl<CSPBoolExpr>>;
pub type IntVar = Value<Array0DImpl<CSPIntVar>>;
pub type IntVarArray1D = Value<Array1DImpl<CSPIntVar>>;
pub type IntVarArray2D = Value<Array2DImpl<CSPIntVar>>;

pub struct Solver {
    solver: IntegratedSolver,
    answer_key_bool: Vec<CSPBoolVar>,
    answer_key_int: Vec<CSPIntVar>,
}

impl Solver {
    pub fn new() -> Solver {
        Solver {
            solver: IntegratedSolver::new(),
            answer_key_bool: vec![],
            answer_key_int: vec![],
        }
    }

    pub fn bool_var(&mut self) -> BoolVar {
        Value(Array0DImpl {
            data: self.solver.new_bool_var(),
        })
    }

    pub fn bool_var_1d(&mut self, len: usize) -> BoolVarArray1D {
        Value(Array1DImpl {
            data: (0..len).map(|_| self.solver.new_bool_var()).collect(),
        })
    }

    pub fn bool_var_2d(&mut self, shape: (usize, usize)) -> BoolVarArray2D {
        let (h, w) = shape;
        Value(Array2DImpl {
            shape,
            data: (0..(h * w)).map(|_| self.solver.new_bool_var()).collect(),
        })
    }

    pub fn int_var(&mut self, low: i32, high: i32) -> IntVar {
        Value(Array0DImpl {
            data: self.solver.new_int_var(Domain::range(low, high)),
        })
    }

    pub fn int_var_1d(&mut self, len: usize, low: i32, high: i32) -> IntVarArray1D {
        Value(Array1DImpl {
            data: (0..len)
                .map(|_| self.solver.new_int_var(Domain::range(low, high)))
                .collect(),
        })
    }

    pub fn int_var_2d(&mut self, shape: (usize, usize), low: i32, high: i32) -> IntVarArray2D {
        let (h, w) = shape;
        Value(Array2DImpl {
            shape,
            data: (0..(h * w))
                .map(|_| self.solver.new_int_var(Domain::range(low, high)))
                .collect(),
        })
    }

    pub fn add_expr<T>(&mut self, exprs: T)
    where
        T: IntoIterator,
        <T as IntoIterator>::Item: Convertible<CSPBoolExpr>,
    {
        exprs
            .into_iter()
            .for_each(|e| self.solver.add_expr(e.convert()));
    }

    pub fn add_answer_key_bool<T>(&mut self, keys: T)
    where
        T: IntoIterator,
        <T as IntoIterator>::Item: Convertible<CSPBoolVar>,
    {
        self.answer_key_bool
            .extend(keys.into_iter().map(|x| x.convert()))
    }

    pub fn add_answer_key_int<T>(&mut self, keys: T)
    where
        T: IntoIterator,
        <T as IntoIterator>::Item: Convertible<CSPIntVar>,
    {
        self.answer_key_int
            .extend(keys.into_iter().map(|x| x.convert()))
    }

    pub fn solve<'a>(&'a mut self) -> Option<Model<'a>> {
        self.solver.solve().map(|model| Model { model })
    }

    pub fn irrefutable_facts(self) -> Option<IrrefutableFacts> {
        self.solver
            .decide_irrefutable_facts(&self.answer_key_bool, &self.answer_key_int)
            .map(|assignment| IrrefutableFacts { assignment })
    }
}

pub trait MapForArray<A, B> {
    type Output;

    fn map<F>(&self, func: F) -> Self::Output
    where
        F: Fn(&A) -> B;
}

impl<A, B> MapForArray<A, B> for Array0DImpl<A> {
    type Output = B;

    fn map<F>(&self, func: F) -> B
    where
        F: Fn(&A) -> B,
    {
        func(&self.data)
    }
}

impl<A, B> MapForArray<A, B> for Array1DImpl<A> {
    type Output = Vec<B>;

    fn map<F>(&self, func: F) -> Vec<B>
    where
        F: Fn(&A) -> B,
    {
        self.data.iter().map(func).collect()
    }
}

impl<A, B> MapForArray<A, B> for Array2DImpl<A> {
    type Output = Vec<Vec<B>>;

    fn map<F>(&self, func: F) -> Vec<Vec<B>>
    where
        F: Fn(&A) -> B,
    {
        let func = &func;
        let (h, w) = self.shape;
        (0..h)
            .into_iter()
            .map(|i| self.data[(i * w)..((i + 1) * w)].iter().map(func).collect())
            .collect()
    }
}

pub trait FromModel {
    type Output;

    fn from_model(&self, model: &Model) -> Self::Output;
}

impl FromModel for Value<Array0DImpl<CSPBoolVar>> {
    type Output = <Array0DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, bool>>::Output;

    fn from_model(&self, model: &Model) -> Self::Output {
        <Array0DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, bool>>::map(&self.0, |v| {
            model.model.get_bool(*v)
        })
    }
}

impl FromModel for Value<Array1DImpl<CSPBoolVar>> {
    type Output = <Array1DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, bool>>::Output;

    fn from_model(&self, model: &Model) -> Self::Output {
        <Array1DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, bool>>::map(&self.0, |v| {
            model.model.get_bool(*v)
        })
    }
}

impl FromModel for Value<Array2DImpl<CSPBoolVar>> {
    type Output = <Array2DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, bool>>::Output;

    fn from_model(&self, model: &Model) -> Self::Output {
        <Array2DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, bool>>::map(&self.0, |v| {
            model.model.get_bool(*v)
        })
    }
}

impl FromModel for Value<Array0DImpl<CSPIntVar>> {
    type Output = <Array0DImpl<CSPIntVar> as MapForArray<CSPIntVar, i32>>::Output;

    fn from_model(&self, model: &Model) -> Self::Output {
        <Array0DImpl<CSPIntVar> as MapForArray<CSPIntVar, i32>>::map(&self.0, |v| {
            model.model.get_int(*v)
        })
    }
}

impl FromModel for Value<Array1DImpl<CSPIntVar>> {
    type Output = <Array1DImpl<CSPIntVar> as MapForArray<CSPIntVar, i32>>::Output;

    fn from_model(&self, model: &Model) -> Self::Output {
        <Array1DImpl<CSPIntVar> as MapForArray<CSPIntVar, i32>>::map(&self.0, |v| {
            model.model.get_int(*v)
        })
    }
}

impl FromModel for Value<Array2DImpl<CSPIntVar>> {
    type Output = <Array2DImpl<CSPIntVar> as MapForArray<CSPIntVar, i32>>::Output;

    fn from_model(&self, model: &Model) -> Self::Output {
        <Array2DImpl<CSPIntVar> as MapForArray<CSPIntVar, i32>>::map(&self.0, |v| {
            model.model.get_int(*v)
        })
    }
}

pub struct Model<'a> {
    model: IntegratedModel<'a>,
}

impl<'a> Model<'a> {
    pub fn get<T>(&self, var: &T) -> <T as FromModel>::Output
    where
        T: FromModel,
    {
        var.from_model(self)
    }
}

pub trait FromIrrefutableFacts {
    type Output;

    fn from_irrefutable_facts(&self, irrefutable_facts: &IrrefutableFacts) -> Self::Output;
}

impl FromIrrefutableFacts for Value<Array0DImpl<CSPBoolVar>> {
    type Output = <Array0DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, Option<bool>>>::Output;

    fn from_irrefutable_facts(&self, irrefutable_facts: &IrrefutableFacts) -> Self::Output {
        <Array0DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, Option<bool>>>::map(&self.0, |v| {
            irrefutable_facts.assignment.get_bool(*v)
        })
    }
}

impl FromIrrefutableFacts for Value<Array1DImpl<CSPBoolVar>> {
    type Output = <Array1DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, Option<bool>>>::Output;

    fn from_irrefutable_facts(&self, irrefutable_facts: &IrrefutableFacts) -> Self::Output {
        <Array1DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, Option<bool>>>::map(&self.0, |v| {
            irrefutable_facts.assignment.get_bool(*v)
        })
    }
}

impl FromIrrefutableFacts for Value<Array2DImpl<CSPBoolVar>> {
    type Output = <Array2DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, Option<bool>>>::Output;

    fn from_irrefutable_facts(&self, irrefutable_facts: &IrrefutableFacts) -> Self::Output {
        <Array2DImpl<CSPBoolVar> as MapForArray<CSPBoolVar, Option<bool>>>::map(&self.0, |v| {
            irrefutable_facts.assignment.get_bool(*v)
        })
    }
}

impl FromIrrefutableFacts for Value<Array0DImpl<CSPIntVar>> {
    type Output = <Array0DImpl<CSPIntVar> as MapForArray<CSPIntVar, Option<i32>>>::Output;

    fn from_irrefutable_facts(&self, irrefutable_facts: &IrrefutableFacts) -> Self::Output {
        <Array0DImpl<CSPIntVar> as MapForArray<CSPIntVar, Option<i32>>>::map(&self.0, |v| {
            irrefutable_facts.assignment.get_int(*v)
        })
    }
}

impl FromIrrefutableFacts for Value<Array1DImpl<CSPIntVar>> {
    type Output = <Array1DImpl<CSPIntVar> as MapForArray<CSPIntVar, Option<i32>>>::Output;

    fn from_irrefutable_facts(&self, irrefutable_facts: &IrrefutableFacts) -> Self::Output {
        <Array1DImpl<CSPIntVar> as MapForArray<CSPIntVar, Option<i32>>>::map(&self.0, |v| {
            irrefutable_facts.assignment.get_int(*v)
        })
    }
}

impl FromIrrefutableFacts for Value<Array2DImpl<CSPIntVar>> {
    type Output = <Array2DImpl<CSPIntVar> as MapForArray<CSPIntVar, Option<i32>>>::Output;

    fn from_irrefutable_facts(&self, irrefutable_facts: &IrrefutableFacts) -> Self::Output {
        <Array2DImpl<CSPIntVar> as MapForArray<CSPIntVar, Option<i32>>>::map(&self.0, |v| {
            irrefutable_facts.assignment.get_int(*v)
        })
    }
}

pub struct IrrefutableFacts {
    assignment: Assignment,
}

impl IrrefutableFacts {
    pub fn get<T>(&self, var: &T) -> <T as FromIrrefutableFacts>::Output
    where
        T: FromIrrefutableFacts,
    {
        var.from_irrefutable_facts(self)
    }
}
