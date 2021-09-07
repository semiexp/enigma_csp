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
