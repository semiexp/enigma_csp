pub fn infer_shape<T>(array: &[Vec<T>]) -> (usize, usize) {
    let height = array.len();
    assert!(height > 0);
    let width = array[0].len();
    (height, width)
}

#[cfg(test)]
pub mod tests {
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
}
