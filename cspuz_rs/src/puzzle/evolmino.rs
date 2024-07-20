use std::ops::Index;

use super::util;
use crate::graph;
use crate::serializer::{
    from_base36, problem_to_url_with_context, to_base36, url_to_problem, Combinator, Context,
    ContextBasedGrid, Map, MultiDigit, Size, Tuple3,
};
use crate::solver::{count_true, Solver};

use enigma_csp::custom_constraints::SimpleCustomConstraint;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProblemCell {
    Empty,
    Black,
    Square,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Problem {
    pub cells: Vec<Vec<ProblemCell>>,
    pub arrows: Vec<Vec<(usize, usize)>>,
}

pub fn solve_evolmino(problem: &Problem) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(&problem.cells);
    let mut solver = Solver::new();
    let is_square = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_square);

    for y in 0..h {
        for x in 0..w {
            match problem.cells[y][x] {
                ProblemCell::Empty => {}
                ProblemCell::Black => {
                    solver.add_expr(!is_square.at((y, x)));
                }
                ProblemCell::Square => {
                    solver.add_expr(is_square.at((y, x)));
                }
            }
        }
    }
    for arrow in &problem.arrows {
        let mut cells = vec![];
        for &p in arrow {
            cells.push(is_square.at(p));
        }
        solver.add_expr(count_true(cells).ge(2));
    }

    let problem = ProblemWithArrowId::new(problem)?;

    #[cfg(not(test))]
    {
        let constraint = EvolminoConstraint {
            board: BoardManager::new(problem.clone()),
            problem,
        };
        solver.add_custom_constraint(Box::new(constraint), is_square);
    }

    #[cfg(test)]
    {
        let constraint = EvolminoConstraint {
            board: BoardManager::new(problem.clone()),
            problem: problem.clone(),
        };
        let cloned_constraint = EvolminoConstraint {
            board: BoardManager::new(problem.clone()),
            problem,
        };

        solver.add_custom_constraint(
            Box::new(util::tests::ReasonVerifier::new(
                constraint,
                cloned_constraint,
            )),
            is_square,
        );
    }

    solver.irrefutable_facts().map(|f| f.get(is_square))
}

type ProblemProxy = (
    Vec<Vec<ProblemCell>>,
    graph::InnerGridEdges<Vec<Vec<bool>>>,
    graph::InnerGridEdges<Vec<Vec<bool>>>,
);

struct ArrowCombinator;

impl Combinator<graph::InnerGridEdges<Vec<Vec<bool>>>> for ArrowCombinator {
    fn serialize(
        &self,
        ctx: &Context,
        input: &[graph::InnerGridEdges<Vec<Vec<bool>>>],
    ) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let input = &input[0];
        let (in_height, in_width) = input.base_shape();

        let height = ctx.height.unwrap();
        assert_eq!(in_height, height);
        let width = ctx.width.unwrap();
        assert_eq!(in_width, width);

        let mut seq = vec![];
        let mut num_false = 0;
        for y in 0..height {
            for x in 0..(width - 1) {
                if input.vertical[y][x] {
                    seq.push(to_base36(num_false));
                    num_false = 0;
                } else {
                    num_false += 1;
                    if num_false == 35 {
                        seq.push(to_base36(35));
                        num_false = 0;
                    }
                }
            }
        }
        for y in 0..(height - 1) {
            for x in 0..width {
                if input.horizontal[y][x] {
                    seq.push(to_base36(num_false));
                    num_false = 0;
                } else {
                    num_false += 1;
                    if num_false == 35 {
                        seq.push(to_base36(35));
                        num_false = 0;
                    }
                }
            }
        }
        if num_false > 0 {
            seq.push(to_base36(num_false));
        }

        Some((1, seq))
    }

    fn deserialize(
        &self,
        ctx: &Context,
        input: &[u8],
    ) -> Option<(usize, Vec<graph::InnerGridEdges<Vec<Vec<bool>>>>)> {
        let height = ctx.height.unwrap();
        let width = ctx.width.unwrap();

        let mut ret = graph::InnerGridEdges {
            vertical: vec![vec![false; width - 1]; height],
            horizontal: vec![vec![false; width]; height - 1],
        };

        let mut num_read = 0;
        let mut pos = 0;
        let lim = height * (width - 1) + (height - 1) * width;
        while pos < lim {
            if num_read >= input.len() {
                return None;
            }
            let n = from_base36(input[num_read])?;
            num_read += 1;

            pos += n as usize;
            if n == 35 {
                continue;
            }
            if pos >= lim {
                break;
            }
            if pos >= height * (width - 1) {
                let p = pos - height * (width - 1);
                ret.horizontal[p / width][p % width] = true;
            } else {
                ret.vertical[pos / (width - 1)][pos % (width - 1)] = true;
            }
            pos += 1;
        }

        Some((num_read, vec![ret]))
    }
}

fn combinator() -> impl Combinator<ProblemProxy> {
    Size::new(Tuple3::new(
        ContextBasedGrid::new(Map::new(
            MultiDigit::new(3, 3),
            |x: ProblemCell| {
                Some(match x {
                    ProblemCell::Empty => 0,
                    ProblemCell::Black => 1,
                    ProblemCell::Square => 2,
                })
            },
            |n: i32| match n {
                0 => Some(ProblemCell::Empty),
                1 => Some(ProblemCell::Black),
                2 => Some(ProblemCell::Square),
                _ => None,
            },
        )),
        ArrowCombinator,
        ArrowCombinator,
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let (h, w) = util::infer_shape(&problem.cells);
    let mut edges_ul = graph::InnerGridEdges {
        vertical: vec![vec![false; w - 1]; h],
        horizontal: vec![vec![false; w]; h - 1],
    };
    let mut edges_dr = graph::InnerGridEdges {
        vertical: vec![vec![false; w - 1]; h],
        horizontal: vec![vec![false; w]; h - 1],
    };

    for arrow in &problem.arrows {
        for i in 1..arrow.len() {
            let (y1, x1) = arrow[i - 1];
            let (y2, x2) = arrow[i];

            if y1 == y2 {
                if x2 + 1 == x1 {
                    edges_ul.vertical[y1][x1 - 1] = true;
                } else if x2 == x1 + 1 {
                    edges_dr.vertical[y1][x1] = true;
                } else {
                    return None;
                }
            } else if x1 == x2 {
                if y2 + 1 == y1 {
                    edges_ul.horizontal[y1 - 1][x1] = true;
                } else if y2 == y1 + 1 {
                    edges_dr.horizontal[y1][x1] = true;
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
    }

    let proxy = (problem.cells.clone(), edges_ul, edges_dr);
    problem_to_url_with_context(combinator(), "evolmino", proxy, &Context::sized(h, w))
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let (cells, edges_ul, edges_dr) = url_to_problem(combinator(), &["evolmino"], url)?;

    let (h, w) = util::infer_shape(&cells);

    let mut arrows = vec![];
    let mut visited = vec![vec![false; w]; h];

    for y in 0..h {
        for x in 0..w {
            if visited[y][x] {
                continue;
            }

            let mut has_in_edge = false;
            if y > 0 {
                has_in_edge |= edges_dr.horizontal[y - 1][x];
            }
            if x > 0 {
                has_in_edge |= edges_dr.vertical[y][x - 1];
            }
            if y < h - 1 {
                has_in_edge |= edges_ul.horizontal[y][x];
            }
            if x < w - 1 {
                has_in_edge |= edges_ul.vertical[y][x];
            }

            if has_in_edge {
                continue;
            }

            let mut arrow = vec![];
            let mut yp = y;
            let mut xp = x;

            loop {
                if visited[yp][xp] {
                    return None;
                }
                visited[yp][xp] = true;
                arrow.push((yp, xp));

                let mut next_cand: Option<(usize, usize)> = None;
                let mut maybe_update = |y2, x2| {
                    if next_cand.is_none() {
                        next_cand = Some((y2, x2));
                        true
                    } else {
                        false
                    }
                };

                if yp > 0 && edges_ul.horizontal[yp - 1][xp] && !maybe_update(yp - 1, xp) {
                    return None;
                }
                if xp > 0 && edges_ul.vertical[yp][xp - 1] && !maybe_update(yp, xp - 1) {
                    return None;
                }
                if yp < h - 1 && edges_dr.horizontal[yp][xp] && !maybe_update(yp + 1, xp) {
                    return None;
                }
                if xp < w - 1 && edges_dr.vertical[yp][xp] && !maybe_update(yp, xp + 1) {
                    return None;
                }

                if let Some((y2, x2)) = next_cand {
                    yp = y2;
                    xp = x2;
                } else {
                    break;
                }
            }

            if arrow.len() >= 2 {
                arrows.push(arrow);
            }
        }
    }

    Some(Problem { cells, arrows })
}

const NO_GROUP: usize = !0;
const INVALID_GROUP: usize = !1;

#[derive(Clone)]
struct ProblemWithArrowId {
    cells: Vec<Vec<ProblemCell>>,
    arrow_id: Vec<Vec<usize>>,
    arrows: Vec<Vec<(usize, usize)>>,
}

impl ProblemWithArrowId {
    fn new(problem: &Problem) -> Option<ProblemWithArrowId> {
        let height = problem.cells.len();
        let width = problem.cells[0].len();
        let mut arrow_id = vec![vec![NO_GROUP; width]; height];

        for (i, arrow) in problem.arrows.iter().enumerate() {
            for &(y, x) in arrow {
                if arrow_id[y][x] != NO_GROUP {
                    return None;
                }
                arrow_id[y][x] = i;
            }
        }

        Some(ProblemWithArrowId {
            cells: problem.cells.clone(),
            arrow_id,
            arrows: problem.arrows.clone(),
        })
    }
}

struct GroupInfo {
    groups_flat: Vec<(usize, usize)>,
    groups_offset: Vec<usize>,
}

impl GroupInfo {
    fn new(group_id: Vec<Vec<usize>>) -> GroupInfo {
        let height = group_id.len();
        let width = group_id[0].len();

        let mut group_size = vec![];
        for y in 0..height {
            for x in 0..width {
                if group_id[y][x] == NO_GROUP {
                    continue;
                }

                while group_size.len() <= group_id[y][x] {
                    group_size.push(0);
                }

                group_size[group_id[y][x]] += 1;
            }
        }

        let mut groups_offset = vec![0];
        for i in 0..group_size.len() {
            groups_offset.push(groups_offset[i] + group_size[i]);
        }

        let mut groups_flat = vec![(0, 0); groups_offset[group_size.len()]];
        let mut cur_pos = groups_offset.clone();

        for y in 0..height {
            for x in 0..width {
                let id = group_id[y][x];
                if id == NO_GROUP {
                    continue;
                }

                groups_flat[cur_pos[id]] = (y, x);
                cur_pos[id] += 1;
            }
        }

        GroupInfo {
            groups_flat,
            groups_offset,
        }
    }

    fn num_groups(&self) -> usize {
        self.groups_offset.len() - 1
    }
}

impl Index<usize> for GroupInfo {
    type Output = [(usize, usize)];

    fn index(&self, index: usize) -> &Self::Output {
        let start = self.groups_offset[index];
        let end = self.groups_offset[index + 1];
        &self.groups_flat[start..end]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoardCell {
    Undecided,
    Square,
    Empty,
}

struct BoardInfoSimple {
    blocks: GroupInfo,
    potential_blocks: GroupInfo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DetailedCellKind {
    // A cell which is decided to be `BoardCell::Empty`. Undecided cells adjacent to different `ArrowBlock`s are also included.
    Empty,

    // A square cell which is orthogonally connected to an arrow cell.
    ArrowBlock,

    // An undecided cell directly adjacent to a `ArrowBlock` cell.
    ArrowBlockNeighbor,

    // Other square / undecided cell.
    Floating,
}

#[derive(Debug)]
struct BoardInfoDetailed {
    cell_info: Vec<Vec<(DetailedCellKind, usize)>>,
    arrow_blocks: Vec<Vec<(usize, usize)>>,
    arrow_block_neighbors: Vec<Vec<(usize, usize)>>,
    floatings: Vec<Vec<(usize, usize)>>,
}

struct BoardManager {
    height: usize,
    width: usize,
    problem: ProblemWithArrowId,
    cells: Vec<Vec<BoardCell>>,
    decision_stack: Vec<(usize, usize)>,
}

impl BoardManager {
    pub fn new(problem: ProblemWithArrowId) -> BoardManager {
        let height = problem.cells.len();
        let width = problem.cells[0].len();
        BoardManager {
            height,
            width,
            problem,
            cells: vec![vec![BoardCell::Undecided; width]; height],
            decision_stack: vec![],
        }
    }

    pub fn decide(&mut self, y: usize, x: usize, cell: BoardCell) {
        assert!(self.cells[y][x] == BoardCell::Undecided);
        self.cells[y][x] = cell;
        self.decision_stack.push((y, x));
    }

    pub fn undo(&mut self) {
        assert!(!self.decision_stack.is_empty());
        let (y, x) = self.decision_stack.pop().unwrap();
        assert!(self.cells[y][x] != BoardCell::Undecided);
        self.cells[y][x] = BoardCell::Undecided;
    }

    #[allow(unused)]
    pub fn dump(&self) {
        for y in 0..self.height {
            for x in 0..self.width {
                eprint!(
                    "{} ",
                    match self.cells[y][x] {
                        BoardCell::Undecided => '.',
                        BoardCell::Square => '#',
                        BoardCell::Empty => '_',
                    }
                );
            }
            eprintln!();
        }
        eprintln!();
    }

    pub fn compute_board_info_simple(&self) -> BoardInfoSimple {
        BoardInfoSimple {
            blocks: self.compute_connected_components(false),
            potential_blocks: self.compute_connected_components(true),
        }
    }

    fn compute_connected_components(&self, is_potential: bool) -> GroupInfo {
        let mut group_id = vec![vec![NO_GROUP; self.width]; self.height];
        let mut id_last = 0;

        let mut stack = vec![];

        let is_check_cell = |y: usize, x: usize| {
            !(self.cells[y][x] == BoardCell::Empty
                || (!is_potential && self.cells[y][x] == BoardCell::Undecided))
        };

        for y in 0..self.height {
            for x in 0..self.width {
                if group_id[y][x] != NO_GROUP {
                    continue;
                }

                if !is_check_cell(y, x) {
                    continue;
                }

                assert!(stack.is_empty());
                stack.push((y, x));

                while let Some((y, x)) = stack.pop() {
                    group_id[y][x] = id_last;

                    foreach_neighbor(y, x, self.height, self.width, |y2, x2| {
                        if group_id[y2][x2] == NO_GROUP && is_check_cell(y2, x2) {
                            stack.push((y2, x2));
                        }
                    });
                }

                id_last += 1;
            }
        }

        GroupInfo::new(group_id)
    }

    pub fn compute_board_info_detailed(&self, info: &BoardInfoSimple) -> BoardInfoDetailed {
        let mut cell_info =
            vec![vec![(DetailedCellKind::Empty, INVALID_GROUP); self.width]; self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                if self.cells[y][x] == BoardCell::Empty {
                    cell_info[y][x] = (DetailedCellKind::Empty, NO_GROUP);
                }
            }
        }

        let mut arrow_blocks = vec![];
        for i in 0..info.blocks.num_groups() {
            let mut has_arrow = false;
            for &(y, x) in &info.blocks[i] {
                if self.problem.arrow_id[y][x] != NO_GROUP {
                    assert!(!has_arrow);
                    has_arrow = true;
                }
            }
            if has_arrow {
                let mut group = vec![];
                for &(y, x) in &info.blocks[i] {
                    cell_info[y][x] = (DetailedCellKind::ArrowBlock, arrow_blocks.len());
                    group.push((y, x));
                }
                arrow_blocks.push(group);
            }
        }

        let mut arrow_block_neighbors = vec![vec![]; arrow_blocks.len()];
        for y in 0..self.height {
            for x in 0..self.width {
                if self.cells[y][x] != BoardCell::Undecided {
                    continue;
                }

                let mut neighbor_block_id = NO_GROUP;
                foreach_neighbor(y, x, self.height, self.width, |ny, nx| {
                    let info = cell_info[ny][nx];
                    if info.0 == DetailedCellKind::ArrowBlock {
                        if neighbor_block_id == NO_GROUP {
                            neighbor_block_id = info.1;
                        } else if neighbor_block_id != info.1 {
                            neighbor_block_id = INVALID_GROUP;
                        }
                    }
                });

                if neighbor_block_id != NO_GROUP && neighbor_block_id != INVALID_GROUP {
                    cell_info[y][x] = (DetailedCellKind::ArrowBlockNeighbor, neighbor_block_id);
                    arrow_block_neighbors[neighbor_block_id].push((y, x));
                } else if neighbor_block_id == INVALID_GROUP {
                    cell_info[y][x] = (DetailedCellKind::Empty, NO_GROUP);
                }
            }
        }

        let mut visited = vec![false; self.height * self.width];
        let mut floatings = vec![];
        for y in 0..self.height {
            for x in 0..self.width {
                if cell_info[y][x].1 != INVALID_GROUP {
                    continue;
                }

                let mut group = vec![];
                traverse_grid(
                    y,
                    x,
                    self.height,
                    self.width,
                    &mut visited,
                    &mut group,
                    |y, x| cell_info[y][x].1 == INVALID_GROUP,
                );

                for &(ny, nx) in &group {
                    cell_info[ny][nx] = (DetailedCellKind::Floating, floatings.len());
                }
                floatings.push(group);
            }
        }

        BoardInfoDetailed {
            cell_info,
            arrow_blocks,
            arrow_block_neighbors,
            floatings,
        }
    }

    fn cell_id(&self, y: usize, x: usize) -> usize {
        y * self.width + x
    }

    fn reason_for_path(&self, ya: usize, xa: usize, yb: usize, xb: usize) -> Vec<(usize, bool)> {
        assert_eq!(self.cells[ya][xa], BoardCell::Square);
        assert_eq!(self.cells[yb][xb], BoardCell::Square);

        let mut bfs: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; self.width]; self.height];
        let mut qu = std::collections::VecDeque::<(usize, usize)>::new();
        bfs[ya][xa] = Some((!0, !0));
        qu.push_back((ya, xa));

        while let Some((y, x)) = qu.pop_front() {
            if y == yb && x == xb {
                break;
            }

            foreach_neighbor(y, x, self.height, self.width, |y2, x2| {
                if self.cells[y2][x2] != BoardCell::Square || bfs[y2][x2].is_some() {
                    return;
                }
                bfs[y2][x2] = Some((y, x));
                qu.push_back((y2, x2));
            });
        }

        assert!(bfs[yb][xb].is_some());
        let mut ret = vec![];

        let mut y = yb;
        let mut x = xb;
        loop {
            ret.push((self.cell_id(y, x), true));

            if y == ya && x == xa {
                break;
            }
            (y, x) = bfs[y][x].unwrap();
        }

        ret
    }

    fn reason_for_potential_unit_boundary(
        &self,
        info: &BoardInfoSimple,
        group_id: usize,
    ) -> Vec<(usize, bool)> {
        let mut ret = vec![];
        for &(y, x) in &info.potential_blocks[group_id] {
            foreach_neighbor(y, x, self.height, self.width, |y2, x2| {
                if self.cells[y2][x2] == BoardCell::Empty {
                    ret.push((self.cell_id(y2, x2), false));
                }
            });
        }
        ret.sort();
        ret.dedup();
        ret
    }

    fn reason_for_arrow_block(
        &self,
        info: &BoardInfoDetailed,
        block_id: usize,
    ) -> Vec<(usize, bool)> {
        let mut ret = vec![];
        for &(y, x) in &info.arrow_blocks[block_id] {
            ret.push((self.cell_id(y, x), true));
        }
        ret
    }

    fn reason_for_adjacent_floating_boundary(
        &self,
        info: &BoardInfoDetailed,
        block_id: usize,
    ) -> Vec<(usize, bool)> {
        let height = self.height;
        let width = self.width;

        let mut ret = vec![];

        let mut disturbing_blocks = vec![];
        for &(y, x) in &info.arrow_blocks[block_id] {
            foreach_neighbor(y, x, height, width, |y2, x2| {
                if self.cells[y2][x2] == BoardCell::Empty {
                    ret.push((self.cell_id(y2, x2), false));
                } else {
                    foreach_neighbor(y2, x2, height, width, |y3, x3| {
                        let i = info.cell_info[y3][x3];
                        if i.0 == DetailedCellKind::ArrowBlock && i.1 != block_id {
                            disturbing_blocks.push(i.1);
                        }
                    })
                }
            });
        }

        for &(y, x) in &info.arrow_block_neighbors[block_id] {
            foreach_neighbor(y, x, height, width, |y2, x2| {
                if self.cells[y2][x2] == BoardCell::Empty {
                    ret.push((self.cell_id(y2, x2), false));
                } else if self.cells[y2][x2] == BoardCell::Undecided {
                    foreach_neighbor(y2, x2, height, width, |y3, x3| {
                        let i = info.cell_info[y3][x3];
                        if i.0 == DetailedCellKind::ArrowBlock && i.1 != block_id {
                            disturbing_blocks.push(i.1);
                        }
                    })
                }
            })
        }

        let mut adjacent_floatings = vec![];
        for &(y, x) in &info.arrow_block_neighbors[block_id] {
            foreach_neighbor(y, x, height, width, |y2, x2| {
                let i = info.cell_info[y2][x2];
                if i.0 == DetailedCellKind::Floating {
                    adjacent_floatings.push(i.1);
                } else if i.0 == DetailedCellKind::ArrowBlockNeighbor && i.1 != block_id {
                    disturbing_blocks.push(i.1);
                }
            })
        }

        adjacent_floatings.sort();
        adjacent_floatings.dedup();

        for f in adjacent_floatings {
            for &(y, x) in &info.floatings[f] {
                foreach_neighbor(y, x, height, width, |y2, x2| {
                    if self.cells[y2][x2] == BoardCell::Empty {
                        ret.push((self.cell_id(y2, x2), false));
                        return;
                    }

                    let i = info.cell_info[y2][x2];
                    if i.0 == DetailedCellKind::ArrowBlockNeighbor && i.1 != block_id {
                        disturbing_blocks.push(i.1);
                    }

                    foreach_neighbor(y2, x2, height, width, |y3, x3| {
                        let i = info.cell_info[y3][x3];
                        if i.0 == DetailedCellKind::ArrowBlock && i.1 != block_id {
                            disturbing_blocks.push(i.1);
                        }
                    })
                });
            }
        }

        disturbing_blocks.sort();
        disturbing_blocks.dedup();

        for d in disturbing_blocks {
            for &(y, x) in &info.arrow_blocks[d] {
                ret.push((self.cell_id(y, x), true));
            }
        }

        ret.sort();
        ret.dedup();
        ret
    }
}

struct EvolminoConstraint {
    board: BoardManager,
    problem: ProblemWithArrowId,
}

impl SimpleCustomConstraint for EvolminoConstraint {
    fn initialize_sat(&mut self, num_inputs: usize) {
        assert_eq!(num_inputs, self.board.height * self.board.width);
    }

    fn notify(&mut self, index: usize, value: bool) {
        let y = index / self.board.width;
        let x = index % self.board.width;
        self.board.decide(
            y,
            x,
            if value {
                BoardCell::Square
            } else {
                BoardCell::Empty
            },
        );
    }

    fn undo(&mut self) {
        self.board.undo();
    }

    fn find_inconsistency(&mut self) -> Option<Vec<(usize, bool)>> {
        let height = self.board.height;
        let width = self.board.width;

        let board_info_simple = self.board.compute_board_info_simple();

        // each block is reachable to an arrow cell
        for i in 0..board_info_simple.potential_blocks.num_groups() {
            let mut square_cell = None;
            let mut has_arrow = false;

            for &(y, x) in &board_info_simple.potential_blocks[i] {
                if self.problem.arrow_id[y][x] != NO_GROUP {
                    has_arrow = true;
                }
                if self.board.cells[y][x] == BoardCell::Square {
                    square_cell = Some((y, x));
                }
            }

            if square_cell.is_some() && !has_arrow {
                let (sy, sx) = square_cell.unwrap();
                let mut ret = self
                    .board
                    .reason_for_potential_unit_boundary(&board_info_simple, i);
                ret.push((self.board.cell_id(sy, sx), true));
                return Some(ret);
            }
        }

        // each block does not contain more than one arrow cell
        for i in 0..board_info_simple.blocks.num_groups() {
            let mut arrow_cell = None;
            for &(y, x) in &board_info_simple.blocks[i] {
                if self.problem.arrow_id[y][x] != NO_GROUP {
                    if let Some((ay, ax)) = arrow_cell {
                        assert_ne!((y, x), (ay, ax));
                        return Some(self.board.reason_for_path(y, x, ay, ax));
                    } else {
                        arrow_cell = Some((y, x));
                    }
                }
            }
        }

        // If two adjacent blocks X, Y appears in this order along an arrow, Y must be an "extension" of X, that is,
        // Y can be obtained by adding at least 1 square cells to X (without flipping and rotation).
        // Note that we can add arbitrarily as many square cells.
        let board_info_detail = self.board.compute_board_info_detailed(&board_info_simple);
        for i in 0..self.problem.arrows.len() {
            let arrow = &self.problem.arrows[i];
            let mut last_block_id: Option<usize> = None;

            for j in 0..arrow.len() {
                let (y, x) = arrow[j];
                if self.board.cells[y][x] != BoardCell::Square {
                    continue;
                }

                assert_eq!(
                    board_info_detail.cell_info[y][x].0,
                    DetailedCellKind::ArrowBlock
                );
                let block_id = board_info_detail.cell_info[y][x].1;

                let mut allowed_floatings = vec![false; board_info_detail.floatings.len()];
                for &(y, x) in &board_info_detail.arrow_block_neighbors[block_id] {
                    foreach_neighbor(y, x, height, width, |y2, x2| {
                        let f = board_info_detail.cell_info[y2][x2];
                        if f.0 == DetailedCellKind::Floating {
                            allowed_floatings[f.1] = true;
                        }
                    });
                }

                if let Some(last_block_id) = last_block_id {
                    let mut isok = false;
                    let last_block = &board_info_detail.arrow_blocks[last_block_id];
                    assert!(last_block.len() > 0);

                    for y in 0..height {
                        for x in 0..width {
                            let mut flg = true;
                            let dy = y as i32 - last_block[0].0 as i32;
                            let dx = x as i32 - last_block[0].1 as i32;

                            for v in 0..last_block.len() {
                                let y2 = last_block[v].0 as i32 + dy;
                                let x2 = last_block[v].1 as i32 + dx;
                                if !(0 <= y2 && y2 < height as i32 && 0 <= x2 && x2 < width as i32)
                                {
                                    flg = false;
                                    break;
                                }

                                let y2 = y2 as usize;
                                let x2 = x2 as usize;
                                let d = board_info_detail.cell_info[y2][x2];
                                if !((d.0 == DetailedCellKind::Floating && allowed_floatings[d.1])
                                    || (d.0 != DetailedCellKind::Floating && d.1 == block_id))
                                {
                                    flg = false;
                                    break;
                                }
                            }

                            if flg {
                                isok = true;
                                break;
                            }
                        }
                        if isok {
                            break;
                        }
                    }

                    if !isok {
                        let mut ret = self
                            .board
                            .reason_for_arrow_block(&board_info_detail, last_block_id);
                        ret.push((self.board.cell_id(y, x), true));
                        ret.extend(
                            self.board.reason_for_adjacent_floating_boundary(
                                &board_info_detail,
                                block_id,
                            ),
                        );
                        return Some(ret);
                    }
                }

                last_block_id = Some(block_id);
            }
        }

        let mut potential_block_size = vec![];
        for i in 0..board_info_detail.arrow_blocks.len() {
            let mut neighbor_floatings = vec![];
            for &(y, x) in &board_info_detail.arrow_block_neighbors[i] {
                foreach_neighbor(y, x, height, width, |y2, x2| {
                    let f = board_info_detail.cell_info[y2][x2];
                    if f.0 == DetailedCellKind::Floating {
                        neighbor_floatings.push(f.1);
                    }
                });
            }
            neighbor_floatings.sort();
            neighbor_floatings.dedup();

            let mut ub = board_info_detail.arrow_blocks[i].len()
                + board_info_detail.arrow_block_neighbors[i].len();
            for f in neighbor_floatings {
                ub += board_info_detail.floatings[f].len();
            }
            potential_block_size.push(ub);
        }

        for i in 0..self.problem.arrows.len() {
            let arrow: &Vec<(usize, usize)> = &self.problem.arrows[i];
            let mut last_block_idx: Option<usize> = None;

            for j in 0..arrow.len() {
                let (y, x) = arrow[j];
                if self.board.cells[y][x] != BoardCell::Square {
                    continue;
                }
                assert_eq!(
                    board_info_detail.cell_info[y][x].0,
                    DetailedCellKind::ArrowBlock
                );

                if let Some(last_block_idx) = last_block_idx {
                    let last_block_id = board_info_detail.cell_info[arrow[last_block_idx].0]
                        [arrow[last_block_idx].1]
                        .1;
                    let cur_block_id = board_info_detail.cell_info[y][x].1;
                    assert_ne!(last_block_id, cur_block_id);

                    let mut gap_ub = 1;
                    {
                        let mut k = last_block_idx + 2;
                        while k < j - 1 {
                            let c = self.board.cells[arrow[k].0][arrow[k].1];
                            assert_ne!(c, BoardCell::Square);
                            if c == BoardCell::Undecided {
                                gap_ub += 1;
                                k += 2;
                            } else {
                                k += 1;
                            }
                        }
                    }

                    let last_lb = board_info_detail.arrow_blocks[last_block_id].len();
                    let last_ub = potential_block_size[last_block_id];
                    let cur_lb = board_info_detail.arrow_blocks[cur_block_id].len();
                    let cur_ub = potential_block_size[cur_block_id];

                    if cur_ub < last_lb + 1 {
                        let mut ret = self
                            .board
                            .reason_for_arrow_block(&board_info_detail, last_block_id);
                        ret.push((self.board.cell_id(y, x), true));
                        ret.extend(self.board.reason_for_adjacent_floating_boundary(
                            &board_info_detail,
                            cur_block_id,
                        ));
                        return Some(ret);
                    }
                    if last_ub + gap_ub < cur_lb {
                        let mut ret = self
                            .board
                            .reason_for_arrow_block(&board_info_detail, cur_block_id);
                        ret.push((
                            arrow[last_block_idx].0 * width + arrow[last_block_idx].1,
                            true,
                        ));
                        ret.extend(self.board.reason_for_adjacent_floating_boundary(
                            &board_info_detail,
                            last_block_id,
                        ));
                        for k in last_block_idx + 2..j - 1 {
                            let (y, x) = arrow[k];
                            if self.board.cells[y][x] == BoardCell::Empty {
                                ret.push((self.board.cell_id(y, x), false));
                            }
                        }
                        return Some(ret);
                    }
                }
                last_block_idx = Some(j);
            }
        }

        None
    }
}

fn foreach_neighbor<F>(y: usize, x: usize, height: usize, width: usize, mut op: F)
where
    F: FnMut(usize, usize) -> (),
{
    if y > 0 {
        op(y - 1, x);
    }
    if y + 1 < height {
        op(y + 1, x);
    }
    if x > 0 {
        op(y, x - 1);
    }
    if x + 1 < width {
        op(y, x + 1);
    }
}

fn traverse_grid<F>(
    y: usize,
    x: usize,
    height: usize,
    width: usize,
    visited: &mut [bool],
    group: &mut Vec<(usize, usize)>,
    enterable: F,
) where
    F: Fn(usize, usize) -> bool + Copy,
{
    if y >= height || x >= width || visited[y * width + x] || !enterable(y, x) {
        return;
    }

    visited[y * width + x] = true;
    group.push((y, x));
    if y > 0 {
        traverse_grid(y - 1, x, height, width, visited, group, enterable);
    }
    if y + 1 < height {
        traverse_grid(y + 1, x, height, width, visited, group, enterable);
    }
    if x > 0 {
        traverse_grid(y, x - 1, height, width, visited, group, enterable);
    }
    if x + 1 < width {
        traverse_grid(y, x + 1, height, width, visited, group, enterable);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn problem_for_tests() -> Problem {
        let mut cells = vec![vec![ProblemCell::Empty; 6]; 7];
        cells[0][0] = ProblemCell::Square;
        cells[0][4] = ProblemCell::Square;
        cells[4][0] = ProblemCell::Square;
        cells[4][2] = ProblemCell::Square;
        cells[6][5] = ProblemCell::Square;
        cells[1][0] = ProblemCell::Black;
        cells[2][3] = ProblemCell::Black;
        cells[3][2] = ProblemCell::Black;
        cells[6][4] = ProblemCell::Black;

        let arrows = vec![
            vec![(0, 2), (1, 2), (2, 2)],
            vec![(0, 5), (1, 5), (2, 5)],
            vec![(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)],
        ];

        Problem { cells, arrows }
    }

    #[test]
    fn test_evolmino_problem() {
        let problem = problem_for_tests();

        let ans = solve_evolmino(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [1, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1],
            [0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 0, 1],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_evolmino_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?evolmino/6/7/i6900910k00005zz1p0008222o";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }

    #[test]
    #[ignore]
    fn test_evolmino_problems() {
        let urls = [
            "https://puzz.link/p?evolmino/8/8/0000io6000800022ii0060zzdsozzh",
            "https://puzz.link/p?evolmino/8/8/00600090ii60000900200iza0m77q10zz07u",
            "https://puzz.link/p?evolmino/9/9/i00000000q008i02o00000000000t0h0000uxp82000010mu968688k",
            "https://puzz.link/p?evolmino/10/10/o2o82o22q0020000000000000000000000wzzn999nzj09000z6999br",
        ];

        for url in urls {
            let problem = deserialize_problem(url).unwrap();
            let ans = solve_evolmino(&problem);
            assert!(ans.is_some());
        }
    }
}
