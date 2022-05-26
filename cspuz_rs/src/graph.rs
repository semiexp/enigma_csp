use std::ops::Index;

use super::solver::{
    count_true, Array0DImpl, Array2DImpl, BoolVar, BoolVarArray1D, BoolVarArray2D, CSPBoolExpr,
    FromModel, FromOwnedPartialModel, Model, Operand, OwnedPartialModel, Solver, Value,
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

/// A struct for maintaining "edges" of a grid, including those on the outer border.
/// Suppose we have a H * W grid. Then, each cell is surrounded by 2 horizontal edges and 2 vertical edges.
/// Thus we have (H + 1) * W horizontal edges and H * (W + 1) vertical edges in total.
///
/// As the method `dual` suggests, `GridEdges` of a H * W grid is in a sense "dual" of `InnerGridEdges` of a (H + 1) * (W + 1) grid.
/// Therefore, we can interchangeably use `GridEdges` and `InnerGridEdges`.
/// Even though, we basically use the one which correctly reflects the "orientation" of edges.
/// For example, answers of H * W Slitherlink problems can be naturally seen as `GridEdges` instances of H * W grids.
/// On the other hand, although answers of H * W Masyu problems can be represented by `InnerGridEdges` instances of H * W grids,
/// they ignore the orientation of edges: the horizontal line connecting cells (y, x) = (0, 0) and (0, 1) is represented by `vertical[0][0]`.
/// Thus Masyu answers are better represented by `GridEdges` instances of (H - 1) * (W - 1) grids.
/// Similarly, edges dividing a grid are better represented by `InnerGridEdges`.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct GridEdges<T> {
    pub horizontal: T,
    pub vertical: T,
}

/// A struct for maintaining "inner edges" of a grid. They do not include edges on the outer border.
/// `InnerGridEdges` for a H * W grid have (H - 1) * W horizontal edges and H * (W - 1) vertical edges.
/// See the description of `GridEdges` for distinction between `GridEdges` and `InnerGridEdges`.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct InnerGridEdges<T> {
    pub horizontal: T,
    pub vertical: T,
}

impl<T> InnerGridEdges<Vec<Vec<T>>> {
    pub fn base_shape(&self) -> (usize, usize) {
        let height = self.vertical.len();
        assert!(height > 0);
        let width = self.vertical[0].len() + 1;
        (height, width)
    }
}

pub fn borders_to_rooms(borders: &InnerGridEdges<Vec<Vec<bool>>>) -> Vec<Vec<(usize, usize)>> {
    fn visit(
        y: usize,
        x: usize,
        borders: &InnerGridEdges<Vec<Vec<bool>>>,
        visited: &mut Vec<Vec<bool>>,
        room: &mut Vec<(usize, usize)>,
    ) {
        if visited[y][x] {
            return;
        }
        visited[y][x] = true;
        room.push((y, x));
        if y > 0 && !borders.horizontal[y - 1][x] {
            visit(y - 1, x, borders, visited, room);
        }
        if y < borders.horizontal.len() && !borders.horizontal[y][x] {
            visit(y + 1, x, borders, visited, room);
        }
        if x > 0 && !borders.vertical[y][x - 1] {
            visit(y, x - 1, borders, visited, room);
        }
        if x < borders.vertical[0].len() && !borders.vertical[y][x] {
            visit(y, x + 1, borders, visited, room);
        }
    }

    let height = borders.vertical.len();
    let width = borders.vertical[0].len() + 1;

    let mut visited = vec![vec![false; width]; height];
    let mut ret = vec![];
    for y in 0..height {
        for x in 0..width {
            if visited[y][x] {
                continue;
            }
            let mut room = vec![];
            visit(y, x, borders, &mut visited, &mut room);
            ret.push(room);
        }
    }

    ret
}

pub type BoolGridEdges = GridEdges<BoolVarArray2D>;
pub type BoolGridEdgesModel = GridEdges<Vec<Vec<bool>>>;
pub type BoolGridEdgesIrrefutableFacts = GridEdges<Vec<Vec<Option<bool>>>>;
pub type BoolInnerGridEdges = InnerGridEdges<BoolVarArray2D>;
pub type BoolInnerGridEdgesModel = InnerGridEdges<Vec<Vec<bool>>>;
pub type BoolInnerGridEdgesIrrefutableFacts = InnerGridEdges<Vec<Vec<Option<bool>>>>;

impl<T> GridEdges<T> {
    pub fn dual(self) -> InnerGridEdges<T> {
        InnerGridEdges {
            horizontal: self.vertical,
            vertical: self.horizontal,
        }
    }
}

impl<T> InnerGridEdges<T> {
    pub fn dual(self) -> GridEdges<T> {
        GridEdges {
            horizontal: self.vertical,
            vertical: self.horizontal,
        }
    }
}

impl BoolGridEdges {
    pub fn new(solver: &mut Solver, shape: (usize, usize)) -> BoolGridEdges {
        let (height, width) = shape;
        BoolGridEdges {
            horizontal: solver.bool_var_2d((height + 1, width)),
            vertical: solver.bool_var_2d((height, width + 1)),
        }
    }

    pub fn base_shape(&self) -> (usize, usize) {
        let horizontal_shape = self.horizontal.shape();
        (horizontal_shape.0 - 1, horizontal_shape.1)
    }

    pub fn at(&self, pos: (usize, usize)) -> BoolVar {
        let (y, x) = pos;
        match (y % 2, x % 2) {
            (0, 1) => self.horizontal.at((y / 2, x / 2)),
            (1, 0) => self.vertical.at((y / 2, x / 2)),
            _ => panic!(),
        }
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

impl FromModel for BoolGridEdges {
    type Output = GridEdges<Vec<Vec<bool>>>;

    fn from_model(&self, model: &Model) -> Self::Output {
        GridEdges {
            horizontal: model.get(&self.horizontal),
            vertical: model.get(&self.vertical),
        }
    }
}

impl FromOwnedPartialModel for BoolGridEdges {
    type Output = GridEdges<Vec<Vec<Option<bool>>>>;
    type OutputUnwrap = GridEdges<Vec<Vec<bool>>>;

    fn from_irrefutable_facts(&self, irrefutable_facts: &OwnedPartialModel) -> Self::Output {
        GridEdges {
            horizontal: irrefutable_facts.get(&self.horizontal),
            vertical: irrefutable_facts.get(&self.vertical),
        }
    }

    fn from_irrefutable_facts_unwrap(
        &self,
        irrefutable_facts: &OwnedPartialModel,
    ) -> Self::OutputUnwrap {
        GridEdges {
            horizontal: irrefutable_facts.get_unwrap(&self.horizontal),
            vertical: irrefutable_facts.get_unwrap(&self.vertical),
        }
    }
}

impl BoolInnerGridEdges {
    pub fn new(solver: &mut Solver, shape: (usize, usize)) -> BoolInnerGridEdges {
        let (height, width) = shape;
        BoolInnerGridEdges {
            horizontal: solver.bool_var_2d((height - 1, width)),
            vertical: solver.bool_var_2d((height, width - 1)),
        }
    }
}

impl FromModel for BoolInnerGridEdges {
    type Output = InnerGridEdges<Vec<Vec<bool>>>;

    fn from_model(&self, model: &Model) -> Self::Output {
        InnerGridEdges {
            horizontal: model.get(&self.horizontal),
            vertical: model.get(&self.vertical),
        }
    }
}

impl FromOwnedPartialModel for BoolInnerGridEdges {
    type Output = InnerGridEdges<Vec<Vec<Option<bool>>>>;
    type OutputUnwrap = InnerGridEdges<Vec<Vec<bool>>>;

    fn from_irrefutable_facts(&self, irrefutable_facts: &OwnedPartialModel) -> Self::Output {
        InnerGridEdges {
            horizontal: irrefutable_facts.get(&self.horizontal),
            vertical: irrefutable_facts.get(&self.vertical),
        }
    }

    fn from_irrefutable_facts_unwrap(
        &self,
        irrefutable_facts: &OwnedPartialModel,
    ) -> Self::OutputUnwrap {
        InnerGridEdges {
            horizontal: irrefutable_facts.get_unwrap(&self.horizontal),
            vertical: irrefutable_facts.get_unwrap(&self.vertical),
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

pub fn single_cycle_grid_edges(solver: &mut Solver, grid_frame: &BoolGridEdges) -> BoolVarArray2D {
    let (edges, graph) = grid_frame.representation();
    let is_passed_flat = active_edges_single_cycle(solver, edges, &graph);
    let (height, width) = grid_frame.base_shape();
    is_passed_flat.reshape_as_2d((height + 1, width + 1))
}
