use std::cmp::{Ordering, PartialEq, PartialOrd};
use std::collections::{btree_map, BTreeMap};
use std::ops::{Add, AddAssign, BitAnd, BitOr, Mul, MulAssign, Neg, Sub, SubAssign};

/// Comparison operators.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Lt,
    Ge,
    Gt,
}

impl CmpOp {
    pub fn compare<T: Ord>(self, lhs: T, rhs: T) -> bool {
        let ord = lhs.cmp(&rhs);
        match self {
            CmpOp::Eq => ord == Ordering::Equal,
            CmpOp::Ne => ord != Ordering::Equal,
            CmpOp::Le => ord != Ordering::Greater,
            CmpOp::Lt => ord == Ordering::Less,
            CmpOp::Ge => ord != Ordering::Less,
            CmpOp::Gt => ord == Ordering::Greater,
        }
    }

    pub fn negate(&self) -> CmpOp {
        match *self {
            CmpOp::Eq => CmpOp::Ne,
            CmpOp::Ne => CmpOp::Eq,
            CmpOp::Le => CmpOp::Gt,
            CmpOp::Lt => CmpOp::Ge,
            CmpOp::Ge => CmpOp::Lt,
            CmpOp::Gt => CmpOp::Le,
        }
    }
}

impl std::fmt::Display for CmpOp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CmpOp::Eq => "==",
                CmpOp::Ne => "!=",
                CmpOp::Le => "<=",
                CmpOp::Lt => "<",
                CmpOp::Ge => ">=",
                CmpOp::Gt => ">",
            }
        )
    }
}

/// Integer type for internal use.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct CheckedInt(i32);

impl CheckedInt {
    pub fn new(value: i32) -> CheckedInt {
        CheckedInt(value)
    }

    pub fn min_value() -> CheckedInt {
        CheckedInt(i32::min_value())
    }

    pub fn max_value() -> CheckedInt {
        CheckedInt(i32::max_value())
    }

    pub fn get(self) -> i32 {
        self.0
    }

    pub fn div_floor(self, rhs: CheckedInt) -> CheckedInt {
        CheckedInt(self.0.checked_div_euclid(rhs.0).unwrap())
    }

    pub fn div_ceil(self, rhs: CheckedInt) -> CheckedInt {
        CheckedInt(
            self.0
                .checked_add(rhs.0 - 1)
                .unwrap()
                .checked_div_euclid(rhs.0)
                .unwrap(),
        )
    }

    pub fn abs(self) -> CheckedInt {
        CheckedInt(self.0.checked_abs().unwrap())
    }
}

impl Add for CheckedInt {
    type Output = CheckedInt;

    fn add(self, rhs: Self) -> Self::Output {
        CheckedInt(self.0.checked_add(rhs.0).unwrap())
    }
}

impl AddAssign for CheckedInt {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.checked_add(rhs.0).unwrap();
    }
}

impl Sub for CheckedInt {
    type Output = CheckedInt;

    fn sub(self, rhs: Self) -> Self::Output {
        CheckedInt(self.0.checked_sub(rhs.0).unwrap())
    }
}

impl SubAssign for CheckedInt {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.checked_sub(rhs.0).unwrap();
    }
}

impl Mul for CheckedInt {
    type Output = CheckedInt;

    fn mul(self, rhs: Self) -> Self::Output {
        CheckedInt(self.0.checked_mul(rhs.0).unwrap())
    }
}

impl MulAssign for CheckedInt {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0.checked_mul(rhs.0).unwrap();
    }
}

impl Neg for CheckedInt {
    type Output = CheckedInt;

    fn neg(self) -> Self::Output {
        CheckedInt(self.0.checked_neg().unwrap())
    }
}

impl PartialEq<i32> for CheckedInt {
    fn eq(&self, other: &i32) -> bool {
        self.0 == *other
    }
}

impl PartialEq<CheckedInt> for i32 {
    fn eq(&self, other: &CheckedInt) -> bool {
        *self == other.0
    }
}

impl PartialOrd<i32> for CheckedInt {
    fn partial_cmp(&self, other: &i32) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(other))
    }
}

impl PartialOrd<CheckedInt> for i32 {
    fn partial_cmp(&self, other: &CheckedInt) -> Option<std::cmp::Ordering> {
        Some(self.cmp(&other.0))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub low: CheckedInt,
    pub high: CheckedInt,
}

impl Range {
    pub fn new(low: CheckedInt, high: CheckedInt) -> Range {
        Range { low, high }
    }

    pub fn empty() -> Range {
        Range {
            low: CheckedInt::max_value(),
            high: CheckedInt::min_value(),
        }
    }

    pub fn any() -> Range {
        Range {
            low: CheckedInt::min_value(),
            high: CheckedInt::max_value(),
        }
    }

    pub fn at_least(c: CheckedInt) -> Range {
        Range {
            low: c,
            high: CheckedInt::max_value(),
        }
    }

    pub fn at_most(c: CheckedInt) -> Range {
        Range {
            low: CheckedInt::min_value(),
            high: c,
        }
    }

    pub fn constant(c: CheckedInt) -> Range {
        Range { low: c, high: c }
    }

    pub fn is_empty(&self) -> bool {
        self.low > self.high
    }
}

impl Add<Range> for Range {
    type Output = Range;

    fn add(self, rhs: Range) -> Self::Output {
        if self.is_empty() || rhs.is_empty() {
            Range::empty()
        } else {
            Range::new(self.low + rhs.low, self.high + rhs.high)
        }
    }
}

impl Mul<CheckedInt> for Range {
    type Output = Range;

    fn mul(self, rhs: CheckedInt) -> Self::Output {
        if self.is_empty() {
            Range::empty()
        } else if rhs >= 0 {
            Range::new(self.low * rhs, self.high * rhs)
        } else {
            Range::new(self.high * rhs, self.low * rhs)
        }
    }
}

impl BitAnd<Range> for Range {
    type Output = Range;

    fn bitand(self, rhs: Range) -> Self::Output {
        Range::new(self.low.max(rhs.low), self.high.min(rhs.high))
    }
}

impl BitOr<Range> for Range {
    type Output = Range;

    fn bitor(self, rhs: Range) -> Self::Output {
        Range::new(self.low.min(rhs.low), self.high.max(rhs.high))
    }
}

#[derive(Clone, Debug)]
pub struct LinearSum<T: Ord> {
    pub(super) term: BTreeMap<T, CheckedInt>,
    pub(super) constant: CheckedInt,
}

impl<T: Ord> LinearSum<T> {
    pub fn new() -> LinearSum<T> {
        LinearSum {
            term: BTreeMap::new(),
            constant: CheckedInt::new(0),
        }
    }

    pub(super) fn constant(v: CheckedInt) -> LinearSum<T> {
        LinearSum {
            term: BTreeMap::new(),
            constant: v,
        }
    }

    pub fn singleton(var: T) -> LinearSum<T> {
        let mut ret = LinearSum::new();
        ret.add_coef(var, CheckedInt::new(1));
        ret
    }

    pub fn is_constant(&self) -> bool {
        self.term.is_empty()
    }

    pub fn len(&self) -> usize {
        self.term.len()
    }

    pub(super) fn add_constant(&mut self, v: CheckedInt) {
        self.constant += v;
    }

    pub(super) fn add_coef(&mut self, var: T, coef: CheckedInt) {
        if coef == 0 {
            return;
        }
        let new_coef = match self.term.get(&var) {
            Some(&e) => e + coef,
            _ => coef,
        };
        if new_coef == 0 {
            self.term.remove(&var);
        } else {
            self.term.insert(var, new_coef);
        }
    }

    pub(super) fn iter(&self) -> btree_map::Iter<T, CheckedInt> {
        self.term.iter()
    }
}

impl<T: Ord> AddAssign<LinearSum<T>> for LinearSum<T> {
    fn add_assign(&mut self, rhs: LinearSum<T>) {
        for (key, value) in rhs.term.into_iter() {
            self.add_coef(key, value);
        }
        self.add_constant(rhs.constant);
    }
}

impl<T: Ord> Add<LinearSum<T>> for LinearSum<T> {
    type Output = LinearSum<T>;

    fn add(self, rhs: LinearSum<T>) -> LinearSum<T> {
        let mut ret = self;
        ret += rhs;
        ret
    }
}

impl<T: Ord> Add<i32> for LinearSum<T> {
    type Output = LinearSum<T>;

    fn add(self, rhs: i32) -> LinearSum<T> {
        let mut ret = self;
        ret += rhs;
        ret
    }
}

impl<T: Ord> AddAssign<i32> for LinearSum<T> {
    fn add_assign(&mut self, rhs: i32) {
        self.add_constant(CheckedInt::new(rhs));
    }
}

impl<T: Ord> SubAssign<LinearSum<T>> for LinearSum<T> {
    fn sub_assign(&mut self, rhs: LinearSum<T>) {
        for (key, value) in rhs.term.into_iter() {
            self.add_coef(key, -value);
        }
        self.add_constant(-rhs.constant);
    }
}

impl<T: Ord> Sub<LinearSum<T>> for LinearSum<T> {
    type Output = LinearSum<T>;

    fn sub(self, rhs: LinearSum<T>) -> LinearSum<T> {
        let mut ret = self;
        ret -= rhs;
        ret
    }
}

impl<T: Ord> MulAssign<CheckedInt> for LinearSum<T> {
    fn mul_assign(&mut self, rhs: CheckedInt) {
        if rhs == 0 {
            *self = LinearSum::new();
        }
        self.constant *= rhs;
        for (_, value) in self.term.iter_mut() {
            *value *= rhs;
        }
    }
}

impl<T: Ord> Mul<CheckedInt> for LinearSum<T> {
    type Output = LinearSum<T>;

    fn mul(self, rhs: CheckedInt) -> LinearSum<T> {
        let mut ret = self;
        ret *= rhs;
        ret
    }
}

impl<T: Ord> MulAssign<i32> for LinearSum<T> {
    fn mul_assign(&mut self, rhs: i32) {
        *self *= CheckedInt::new(rhs);
    }
}

impl<T: Ord> Mul<i32> for LinearSum<T> {
    type Output = LinearSum<T>;

    fn mul(self, rhs: i32) -> LinearSum<T> {
        self * CheckedInt::new(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_floor() {
        assert_eq!(CheckedInt::new(12).div_floor(CheckedInt::new(4)), 3);
        assert_eq!(CheckedInt::new(10).div_floor(CheckedInt::new(3)), 3);
        assert_eq!(CheckedInt::new(0).div_floor(CheckedInt::new(7)), 0);
        assert_eq!(CheckedInt::new(-42).div_floor(CheckedInt::new(4)), -11);
        assert_eq!(CheckedInt::new(-42).div_floor(CheckedInt::new(3)), -14);
    }

    #[test]
    fn test_div_ceil() {
        assert_eq!(CheckedInt::new(12).div_ceil(CheckedInt::new(4)), 3);
        assert_eq!(CheckedInt::new(10).div_ceil(CheckedInt::new(3)), 4);
        assert_eq!(CheckedInt::new(0).div_ceil(CheckedInt::new(7)), 0);
        assert_eq!(CheckedInt::new(-42).div_ceil(CheckedInt::new(4)), -10);
        assert_eq!(CheckedInt::new(-42).div_ceil(CheckedInt::new(3)), -14);
    }
}
