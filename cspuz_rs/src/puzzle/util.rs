use std::ops::{Index, IndexMut};

pub fn infer_shape<T>(array: &[Vec<T>]) -> (usize, usize) {
    let height = array.len();
    assert!(height > 0);
    let width = array[0].len();
    (height, width)
}

#[derive(Clone)]
pub struct Grid<T: Clone> {
    data: Vec<T>,
    height: usize,
    width: usize,
}

impl<T: Clone> Grid<T> {
    pub fn new(height: usize, width: usize, default: T) -> Grid<T> {
        Grid {
            data: vec![default; height * width],
            height,
            width,
        }
    }

    pub fn from_vecs(vecs: &[Vec<T>]) -> Grid<T> {
        let height = vecs.len();
        assert!(height > 0);
        let width = vecs[0].len();
        let mut buf = vec![];
        for i in 0..height {
            assert_eq!(vecs[i].len(), width);
            buf.extend_from_slice(&vecs[i]);
        }

        Grid {
            data: buf,
            height,
            width,
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }
}

impl<T: Clone> Index<(usize, usize)> for Grid<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (y, x) = index;
        assert!(y < self.height && x < self.width);
        unsafe { self.data.get_unchecked(y * self.width + x) }
    }
}

impl<T: Clone> IndexMut<(usize, usize)> for Grid<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (y, x) = index;
        assert!(y < self.height && x < self.width);
        unsafe { self.data.get_unchecked_mut(y * self.width + x) }
    }
}

#[cfg(test)]
pub mod tests {
    use enigma_csp::custom_constraints::SimpleCustomConstraint;

    pub fn to_option_2d<X, Y, T>(array: X) -> Vec<Vec<Option<T>>>
    where
        X: IntoIterator<Item = Y>,
        Y: IntoIterator<Item = T>,
    {
        array
            .into_iter()
            .map(|row| row.into_iter().map(Some).collect())
            .collect()
    }

    pub fn to_bool_2d<X, Y>(array: X) -> Vec<Vec<bool>>
    where
        X: IntoIterator<Item = Y>,
        Y: IntoIterator<Item = i32>,
    {
        array
            .into_iter()
            .map(|row| row.into_iter().map(|x| x != 0).collect())
            .collect()
    }

    pub fn to_option_bool_2d<X, Y>(array: X) -> Vec<Vec<Option<bool>>>
    where
        X: IntoIterator<Item = Y>,
        Y: IntoIterator<Item = i32>,
    {
        to_option_2d(to_bool_2d(array))
    }

    pub fn check_all_some<T>(input: &[Vec<Option<T>>]) {
        for row in input {
            for x in row {
                assert!(x.is_some());
            }
        }
    }

    pub fn serializer_test<T, F, G>(problem: T, url: &str, serializer: F, deserializer: G)
    where
        T: PartialEq + std::fmt::Debug,
        F: Fn(&T) -> Option<String>,
        G: Fn(&str) -> Option<T>,
    {
        let deserialized = deserializer(url);
        assert!(deserialized.is_some());
        let deserialized = deserialized.unwrap();
        assert_eq!(problem, deserialized);
        let reserialized = serializer(&deserialized);
        assert!(reserialized.is_some());
        let reserialized = reserialized.unwrap();
        assert_eq!(reserialized, url);
    }

    pub struct ReasonVerifier<T: SimpleCustomConstraint> {
        constraint: T,
        cloned_constraint: T,
    }

    impl<T: SimpleCustomConstraint> ReasonVerifier<T> {
        pub fn new(constraint: T, cloned_constraint: T) -> ReasonVerifier<T> {
            ReasonVerifier {
                constraint,
                cloned_constraint,
            }
        }
    }

    impl<T: SimpleCustomConstraint> SimpleCustomConstraint for ReasonVerifier<T> {
        fn initialize_sat(&mut self, num_inputs: usize) {
            self.constraint.initialize_sat(num_inputs);
            self.cloned_constraint.initialize_sat(num_inputs);
        }

        fn notify(&mut self, index: usize, value: bool) {
            self.constraint.notify(index, value);
        }

        fn undo(&mut self) {
            self.constraint.undo();
        }

        fn find_inconsistency(&mut self) -> Option<Vec<(usize, bool)>> {
            let reason = self.constraint.find_inconsistency();

            if let Some(reason) = &reason {
                let mut reason = reason.clone();
                reason.sort();
                reason.dedup();
                for &(idx, value) in &reason {
                    self.cloned_constraint.notify(idx, value);
                }

                let cloned_reason = self.cloned_constraint.find_inconsistency();
                assert!(cloned_reason.is_some());

                // TODO: check if `cloned_reason` is a subset of `reason`
                for item in &cloned_reason.unwrap() {
                    assert!(reason.binary_search(item).is_ok());
                }

                for _ in 0..reason.len() {
                    self.cloned_constraint.undo();
                }
            }

            reason
        }
    }
}
