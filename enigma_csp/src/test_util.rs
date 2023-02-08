/// Given v1 and v2, return v1 * v2 as Vec<(A, B)>, where * represents the Cartesian product.
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

pub fn check_graph_division(
    sizes: &[Option<i32>],
    edges: &[(usize, usize)],
    edge_disconnected: &[bool],
) -> bool {
    let mut adj = vec![vec![]; sizes.len()];
    for i in 0..edges.len() {
        if !edge_disconnected[i] {
            let (u, v) = edges[i];
            adj[u].push(v);
            adj[v].push(u);
        }
    }

    let mut grp_id = vec![!0; sizes.len()];
    fn dfs(grp_id: &mut [usize], adj: &[Vec<usize>], p: usize, id: usize) -> i32 {
        if grp_id[p] != !0 {
            return 0;
        }
        grp_id[p] = id;
        let mut ret = 1;
        for &q in &adj[p] {
            ret += dfs(grp_id, adj, q, id);
        }
        ret
    }

    let mut grp_size = vec![];
    for i in 0..sizes.len() {
        if grp_id[i] != !0 {
            continue;
        }
        let size = dfs(&mut grp_id, &adj, i, grp_size.len());
        grp_size.push(size);
    }

    for i in 0..sizes.len() {
        if let Some(s) = sizes[i] {
            if s != grp_size[grp_id[i]] {
                return false;
            }
        }
    }
    for i in 0..edges.len() {
        if edge_disconnected[i] {
            let (u, v) = edges[i];
            if grp_id[u] == grp_id[v] {
                return false;
            }
        }
    }

    true
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
        assert_eq!(product_multi::<i32>(&vec![]), vec![Vec::<i32>::new()]);
    }

    #[test]
    fn test_product_multi4() {
        assert_eq!(
            product_multi(&vec![vec![1, 2], vec![], vec![3, 4, 5]]),
            Vec::<Vec::<i32>>::new()
        );
    }
}
