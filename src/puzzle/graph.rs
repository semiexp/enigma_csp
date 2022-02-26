use std::ops::Index;

use super::solver::{
    count_true, Array0DImpl, Array2DImpl, BoolVar, BoolVarArray1D, BoolVarArray2D, CSPBoolExpr,
    FromIrrefutableFacts, FromModel, IrrefutableFacts, Model, Operand, Solver, Value,
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

#[derive(PartialEq, Eq, Debug)]
pub struct GridFrame<T> {
    pub horizontal: T,
    pub vertical: T,
}

pub type BoolGridFrame = GridFrame<BoolVarArray2D>;
pub type BoolGridFrameModel = GridFrame<Vec<Vec<bool>>>;
pub type BoolGridFrameIrrefutableFacts = GridFrame<Vec<Vec<Option<bool>>>>;

impl BoolGridFrame {
    pub fn new(solver: &mut Solver, shape: (usize, usize)) -> BoolGridFrame {
        let (height, width) = shape;
        BoolGridFrame {
            horizontal: solver.bool_var_2d((height + 1, width)),
            vertical: solver.bool_var_2d((height, width + 1)),
        }
    }

    pub fn base_shape(&self) -> (usize, usize) {
        let horizontal_shape = self.horizontal.shape();
        (horizontal_shape.0 - 1, horizontal_shape.1)
    }

    pub fn representation(&self) -> (Vec<BoolVar>, Graph) {
        let (height, width) = self.base_shape();

        let mut edges = vec![];
        let mut graph = Graph::new((height + 1) * (width + 1));

        for y in 0..=height {
            for x in 0..=width {
                if y < height {
                    edges.push(self.vertical.at((y, x)));
                    graph.add_edge(y * (width + 1) + x, (y + 1) * (width + 1) + x);
                }
                if x < width {
                    edges.push(self.horizontal.at((y, x)));
                    graph.add_edge(y * (width + 1) + x, y * (width + 1) + (x + 1));
                }
            }
        }

        (edges, graph)
    }

    pub fn cell_neighbors(&self, cell: (usize, usize)) -> BoolVarArray1D {
        let (y, x) = cell;
        BoolVarArray1D::new([
            self.horizontal.at((y, x)),
            self.horizontal.at((y + 1, x)),
            self.vertical.at((y, x)),
            self.vertical.at((y, x + 1)),
        ])
    }
}

impl FromModel for BoolGridFrame {
    type Output = GridFrame<Vec<Vec<bool>>>;

    fn from_model(&self, model: &Model) -> Self::Output {
        GridFrame {
            horizontal: model.get(&self.horizontal),
            vertical: model.get(&self.vertical),
        }
    }
}

impl FromIrrefutableFacts for BoolGridFrame {
    type Output = GridFrame<Vec<Vec<Option<bool>>>>;

    fn from_irrefutable_facts(&self, irrefutable_facts: &IrrefutableFacts) -> Self::Output {
        GridFrame {
            horizontal: irrefutable_facts.get(&self.horizontal),
            vertical: irrefutable_facts.get(&self.vertical),
        }
    }
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

pub fn active_edges_single_cycle<T>(
    solver: &mut Solver,
    is_active_edge: T,
    graph: &Graph,
) -> BoolVarArray1D
where
    T: IntoIterator,
    <T as IntoIterator>::Item: Operand<Output = Array0DImpl<CSPBoolExpr>>,
{
    let is_active_edge: Vec<Value<Array0DImpl<CSPBoolExpr>>> = is_active_edge
        .into_iter()
        .map(|x| x.as_expr_array_value())
        .collect::<Vec<_>>();
    assert_eq!(is_active_edge.len(), graph.n_edges());

    let mut adj: Vec<Vec<(usize, usize)>> = vec![]; // (edge id, adjacent vertex)
    for _ in 0..graph.n_vertices() {
        adj.push(vec![]);
    }
    for (i, &(u, v)) in graph.edges.iter().enumerate() {
        adj[u].push((i, v));
        adj[v].push((i, u));
    }

    // degree constraints
    let is_passed = solver.bool_var_1d(graph.n_vertices());
    for u in 0..graph.n_vertices() {
        let adj_edges = adj[u].iter().map(|&(i, _)| is_active_edge[i].clone());
        solver.add_expr(count_true(adj_edges).eq(is_passed.at(u).ite(2, 0)));
    }

    let mut line_graph = Graph::new(graph.n_edges());
    for a in &adj {
        for i in 0..a.len() {
            for j in (i + 1)..a.len() {
                let e = a[i].0;
                let f = a[j].0;
                line_graph.add_edge(e, f);
            }
        }
    }
    active_vertices_connected(solver, &is_active_edge, &line_graph);

    is_passed
}

pub fn single_cycle_grid_frame(solver: &mut Solver, grid_frame: &BoolGridFrame) {
    let (edges, graph) = grid_frame.representation();
    active_edges_single_cycle(solver, edges, &graph);
}
