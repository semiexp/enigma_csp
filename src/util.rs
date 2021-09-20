use std::marker::PhantomData;
use std::ops::{BitOr, BitOrAssign, Index, IndexMut};

/// Returns `floor(a / b)` assuming b > 0.
pub fn div_floor(a: i32, b: i32) -> i32 {
    a.checked_div_euclid(b).unwrap()
}

/// Returns `ceil(a / b)` assuming b > 0.
pub fn div_ceil(a: i32, b: i32) -> i32 {
    a.checked_add(b - 1).unwrap().checked_div_euclid(b).unwrap()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateStatus {
    NotUpdated,
    Updated,
    Unsatisfiable,
}

impl BitOr<UpdateStatus> for UpdateStatus {
    type Output = UpdateStatus;

    fn bitor(self, rhs: UpdateStatus) -> Self::Output {
        match (self, rhs) {
            (UpdateStatus::Unsatisfiable, _) | (_, UpdateStatus::Unsatisfiable) => {
                UpdateStatus::Unsatisfiable
            }
            (UpdateStatus::Updated, _) | (_, UpdateStatus::Updated) => UpdateStatus::Updated,
            _ => UpdateStatus::NotUpdated,
        }
    }
}

impl BitOrAssign<UpdateStatus> for UpdateStatus {
    fn bitor_assign(&mut self, rhs: UpdateStatus) {
        *self = *self | rhs;
    }
}

pub trait ConvertMapIndex {
    fn to_index(&self) -> usize;
}

pub struct ConvertMap<K: ConvertMapIndex, V> {
    data: Vec<Option<V>>,
    key_type: PhantomData<K>,
}

impl<K: ConvertMapIndex, V> ConvertMap<K, V> {
    pub fn new() -> ConvertMap<K, V> {
        ConvertMap {
            data: vec![],
            key_type: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<K: ConvertMapIndex, V> Index<K> for ConvertMap<K, V> {
    type Output = Option<V>;

    fn index(&self, index: K) -> &Self::Output {
        let index = index.to_index();
        if index < self.len() {
            &self.data[index]
        } else {
            &None
        }
    }
}

impl<K: ConvertMapIndex, V> IndexMut<K> for ConvertMap<K, V> {
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        let index = index.to_index();
        while self.len() <= index {
            self.data.push(None);
        }
        &mut self.data[index]
    }
}

/// Given v1 and v2, return v1 * v2 as Vec<(A, B)>, where * represents the Cartesian product.
#[allow(dead_code)]
pub fn product_binary<A: Clone, B: Clone>(a: &[A], b: &[B]) -> Vec<(A, B)> {
    let mut ret = vec![];

    for x in a {
        for y in b {
            ret.push((x.clone(), y.clone()));
        }
    }

    ret
}

/// Given [v1, v2, ..., vn], return v1 * v2 * ... * vn, where * represents the Cartesian product.
#[allow(dead_code)]
pub fn product_multi<T: Clone>(inputs: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut ret = vec![];

    fn visit<T: Clone>(inputs: &Vec<Vec<T>>, i: usize, buf: &mut Vec<T>, ret: &mut Vec<Vec<T>>) {
        if i == inputs.len() {
            ret.push(buf.clone());
            return;
        }
        for x in &inputs[i] {
            buf.push(x.clone());
            visit(inputs, i + 1, buf, ret);
            buf.pop();
        }
    }

    visit(inputs, 0, &mut vec![], &mut ret);
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_floor() {
        assert_eq!(div_floor(12, 4), 3);
        assert_eq!(div_floor(10, 3), 3);
        assert_eq!(div_floor(0, 7), 0);
        assert_eq!(div_floor(-42, 4), -11);
        assert_eq!(div_floor(-42, 3), -14);
    }

    #[test]
    fn test_div_ceil() {
        assert_eq!(div_ceil(12, 4), 3);
        assert_eq!(div_ceil(10, 3), 4);
        assert_eq!(div_ceil(0, 7), 0);
        assert_eq!(div_ceil(-42, 4), -10);
        assert_eq!(div_ceil(-42, 3), -14);
    }

    #[test]
    fn test_product_binary1() {
        assert_eq!(
            product_binary(&[1, 2, 3], &[true, false]),
            vec![
                (1, true),
                (1, false),
                (2, true),
                (2, false),
                (3, true),
                (3, false)
            ]
        );
    }

    #[test]
    fn test_product_binary2() {
        assert_eq!(product_binary::<_, bool>(&[1, 2, 3], &[]), vec![]);
    }

    #[test]
    fn test_product_multi1() {
        assert_eq!(product_multi(&vec![vec![1, 2]]), vec![vec![1], vec![2]]);
    }

    #[test]
    fn test_product_multi2() {
        assert_eq!(
            product_multi(&vec![vec![1, 2], vec![3, 4, 5], vec![6]]),
            vec![
                vec![1, 3, 6],
                vec![1, 4, 6],
                vec![1, 5, 6],
                vec![2, 3, 6],
                vec![2, 4, 6],
                vec![2, 5, 6],
            ]
        );
    }

    #[test]
    fn test_product_multi3() {
        assert_eq!(product_multi::<i32>(&vec![]), vec![vec![]]);
    }

    #[test]
    fn test_product_multi4() {
        assert_eq!(
            product_multi(&vec![vec![1, 2], vec![], vec![3, 4, 5]]),
            Vec::<Vec::<i32>>::new()
        );
    }
}
