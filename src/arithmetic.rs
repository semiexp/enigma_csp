use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_floor() {
        assert_eq!(
            CheckedInt::new(12).div_floor(CheckedInt::new(4)),
            CheckedInt::new(3)
        );
        assert_eq!(
            CheckedInt::new(10).div_floor(CheckedInt::new(3)),
            CheckedInt::new(3)
        );
        assert_eq!(
            CheckedInt::new(0).div_floor(CheckedInt::new(7)),
            CheckedInt::new(0)
        );
        assert_eq!(
            CheckedInt::new(-42).div_floor(CheckedInt::new(4)),
            CheckedInt::new(-11)
        );
        assert_eq!(
            CheckedInt::new(-42).div_floor(CheckedInt::new(3)),
            CheckedInt::new(-14)
        );
    }

    #[test]
    fn test_div_ceil() {
        assert_eq!(
            CheckedInt::new(12).div_ceil(CheckedInt::new(4)),
            CheckedInt::new(3)
        );
        assert_eq!(
            CheckedInt::new(10).div_ceil(CheckedInt::new(3)),
            CheckedInt::new(4)
        );
        assert_eq!(
            CheckedInt::new(0).div_ceil(CheckedInt::new(7)),
            CheckedInt::new(0)
        );
        assert_eq!(
            CheckedInt::new(-42).div_ceil(CheckedInt::new(4)),
            CheckedInt::new(-10)
        );
        assert_eq!(
            CheckedInt::new(-42).div_ceil(CheckedInt::new(3)),
            CheckedInt::new(-14)
        );
    }
}
