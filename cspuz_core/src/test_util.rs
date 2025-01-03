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

pub fn check_graph_active_vertices_connected(is_active: &[bool], edges: &[(usize, usize)]) -> bool {
    let n = is_active.len();
    let mut graph = vec![vec![]; n];
    for &(u, v) in edges {
        graph[u].push(v);
        graph[v].push(u);
    }

    let mut visited = vec![false; n];
    fn visit(graph: &[Vec<usize>], is_active: &[bool], visited: &mut [bool], p: usize) {
        if visited[p] || !is_active[p] {
            return;
        }
        visited[p] = true;
        for &q in &graph[p] {
            visit(graph, is_active, visited, q);
        }
    }

    let mut n_connected_components = 0;
    for u in 0..n {
        if is_active[u] && !visited[u] {
            n_connected_components += 1;
            visit(&graph, &is_active, &mut visited, u);
        }
    }

    n_connected_components <= 1
}

pub fn check_circuit(values: &[i32]) -> bool {
    let n = values.len();
    if values.iter().any(|&x| x < 0 || x >= n as i32) {
        return false;
    }
    let values = values.iter().map(|&x| x as usize).collect::<Vec<_>>();

    let mut cyc_size = 0;
    for i in 0..n {
        if values[i] != i {
            cyc_size += 1;
        }
    }

    let mut visited = vec![false; n];
    for i in 0..n {
        if values[i] != i {
            let mut size = 0;
            let mut p = i;
            while !visited[p] {
                if values[p] == p {
                    return false;
                }
                size += 1;
                visited[p] = true;
                p = values[p];
            }
            if p != i {
                return false;
            }
            if size != cyc_size {
                return false;
            }
            break;
        }
    }
    true
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

    #[test]
    fn test_check_active_vertices_connected() {
        let edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)];

        assert_eq!(
            check_graph_active_vertices_connected(&[false, false, false, false, false], &edges),
            true
        );
        assert_eq!(
            check_graph_active_vertices_connected(&[true, false, false, false, false], &edges),
            true
        );
        assert_eq!(
            check_graph_active_vertices_connected(&[true, false, false, true, false], &edges),
            false
        );
        assert_eq!(
            check_graph_active_vertices_connected(&[true, false, true, true, false], &edges),
            true
        );
        assert_eq!(
            check_graph_active_vertices_connected(&[true, true, true, true, true], &edges),
            true
        );
    }
}
