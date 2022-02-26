use std::ops::Index;

use super::solver::{
    Array0DImpl, Array2DImpl, CSPBoolExpr, Operand, Solver,
};

pub struct Graph {
    n_vertices: usize,
    edges: Vec<(usize, usize)>,
}

impl Graph {
    pub fn new(n_vertices: usize) -> Graph {
        Graph {
            n_vertices,
            edges: vec![],
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        assert!(u < self.n_vertices);
        assert!(v < self.n_vertices);
        self.edges.push((u, v));
    }

    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

impl Index<usize> for Graph {
    type Output = (usize, usize);

    fn index(&self, index: usize) -> &Self::Output {
        &self.edges[index]
    }
}

fn infer_graph_from_2d_array(shape: (usize, usize)) -> Graph {
    let (h, w) = shape;
    let mut graph = Graph::new(h * w);
    for y in 0..h {
        for x in 0..w {
            if x < w - 1 {
                graph.add_edge(y * w + x, y * w + (x + 1));
            }
            if y < h - 1 {
                graph.add_edge(y * w + x, (y + 1) * w + x);
            }
        }
    }
    graph
}

pub fn active_vertices_connected<T>(solver: &mut Solver, is_active: T, graph: &Graph)
where
    T: IntoIterator,
    <T as IntoIterator>::Item: Operand<Output = Array0DImpl<CSPBoolExpr>>,
{
    solver.add_active_vertices_connected(is_active, &graph.edges);
}

pub fn active_vertices_connected_2d<T>(solver: &mut Solver, is_active: T)
where
    T: Operand<Output = Array2DImpl<CSPBoolExpr>>,
{
    let is_active = is_active.as_expr_array_value();
    let graph = infer_graph_from_2d_array(is_active.shape());
    active_vertices_connected(solver, is_active, &graph)
}
