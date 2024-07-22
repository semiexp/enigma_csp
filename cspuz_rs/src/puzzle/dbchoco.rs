use std::ops::Index;

use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, ContextBasedGrid,
    HexInt, MultiDigit, Optionalize, Size, Spaces, Tuple2,
};
use crate::solver::Solver;

use enigma_csp::custom_constraints::SimpleCustomConstraint;

pub fn solve_doublechoco(
    color: &[Vec<i32>],
    num: &[Vec<Option<i32>>],
) -> Option<graph::BoolInnerGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(color);
    assert_eq!(util::infer_shape(num), (h, w));

    let mut solver = Solver::new();
    let is_border = graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&is_border.horizontal);
    solver.add_answer_key_bool(&is_border.vertical);

    color.to_vec();

    let edges_flat = is_border
        .vertical
        .clone()
        .into_iter()
        .chain(is_border.horizontal.clone().into_iter())
        .collect::<Vec<_>>();

    #[cfg(not(test))]
    {
        let constraint = DoublechocoConstraint {
            board: BoardManager::new(color),
            cell_color: color.to_vec(),
            cell_num: num
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&n| n.map(|x| x as usize))
                        .collect::<Vec<_>>()
                })
                .collect(),
        };

        solver.add_custom_constraint(Box::new(constraint), edges_flat);
    }

    #[cfg(test)]
    {
        let constraint = DoublechocoConstraint {
            board: BoardManager::new(color),
            cell_color: color.to_vec(),
            cell_num: num
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&n| n.map(|x| x as usize))
                        .collect::<Vec<_>>()
                })
                .collect(),
        };
        let cloned_constraint = DoublechocoConstraint {
            board: BoardManager::new(color),
            cell_color: color.to_vec(),
            cell_num: num
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&n| n.map(|x| x as usize))
                        .collect::<Vec<_>>()
                })
                .collect(),
        };

        solver.add_custom_constraint(
            Box::new(util::tests::ReasonVerifier::new(
                constraint,
                cloned_constraint,
            )),
            edges_flat,
        );
    }

    solver.irrefutable_facts().map(|f| f.get(&is_border))
}

type Problem = (Vec<Vec<i32>>, Vec<Vec<Option<i32>>>);

fn combinator() -> impl Combinator<Problem> {
    Size::new(Tuple2::new(
        ContextBasedGrid::new(MultiDigit::new(2, 5)),
        ContextBasedGrid::new(Choice::new(vec![
            Box::new(Optionalize::new(HexInt)),
            Box::new(Spaces::new(None, 'g')),
        ])),
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let (h, w) = util::infer_shape(&problem.0);
    problem_to_url_with_context(
        combinator(),
        "dbchoco",
        problem.clone(),
        &Context::sized(h, w),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["dbchoco"], url)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Border {
    Undecided,
    Wall,
    Connected,
}

// TODO: remove duplicated codes (`evolmino.rs` has the same code)

const NO_GROUP: usize = !0;

struct GroupInfo {
    group_id: Vec<Vec<usize>>,
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
            group_id,
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

#[derive(Clone, Debug)]
struct Shape {
    // invariant:
    // - `cells` is sorted
    // - `cells[0]` is (0, 0)
    cells: Vec<(i32, i32)>,

    // If two adjacent cells (y, x) and (y', x') are in `cells`, (y + y', x + x') must be in `connections`.
    // `connections` need not be necessarily sorted
    connections: Vec<(i32, i32)>,
}

impl std::cmp::PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        // we only compare `cells` because `connections` is derived from `cells`
        self.cells == other.cells
    }
}

impl std::cmp::Eq for Shape {}

impl std::cmp::PartialOrd for Shape {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cells.cmp(&other.cells))
    }
}

impl std::cmp::Ord for Shape {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cells.cmp(&other.cells)
    }
}

impl Shape {
    fn is_invariant_met(&self) -> bool {
        if self.cells[0] != (0, 0) {
            return false;
        }
        for i in 1..self.cells.len() {
            if !(self.cells[i - 1] < self.cells[i]) {
                return false;
            }
        }

        true
    }

    // Normalize the shape so that the top-left cell is (0, 0) assuming that `cells` is sorted.
    fn normalize(&mut self) {
        let (min_y, min_x) = self.cells[0];
        for (y, x) in &mut self.cells {
            *y -= min_y;
            *x -= min_x;
        }
        for (y, x) in &mut self.connections {
            *y -= min_y * 2;
            *x -= min_x * 2;
        }
    }

    fn rotate90(&self) -> Shape {
        let mut new_cells = self.cells.iter().map(|&(y, x)| (x, -y)).collect::<Vec<_>>();
        let new_connections = self
            .connections
            .iter()
            .map(|&(y, x)| (x, -y))
            .collect::<Vec<_>>();
        new_cells.sort();

        let mut ret = Shape {
            cells: new_cells,
            connections: new_connections,
        };
        ret.normalize();
        ret
    }

    fn flip_y(&self) -> Shape {
        let mut new_cells = self.cells.iter().map(|&(y, x)| (-y, x)).collect::<Vec<_>>();
        let new_connections = self
            .connections
            .iter()
            .map(|&(y, x)| (-y, x))
            .collect::<Vec<_>>();
        new_cells.sort();

        let mut ret = Shape {
            cells: new_cells,
            connections: new_connections,
        };
        ret.normalize();
        ret
    }
}

fn enumerate_transforms(shape: &Shape) -> Vec<Shape> {
    let mut ret = vec![];
    ret.push(shape.clone());
    for i in 0..3 {
        ret.push(ret[i].rotate90());
    }
    for i in 0..4 {
        ret.push(ret[i].flip_y());
    }

    ret.sort();
    ret.dedup();

    ret
}

// Connectivity information of a puzzle board.
// - unit: connected components consists of cells of one color (connections between differently colored cells are ignored)
// - block: connected components consists of cells of both colors. "potential" means the connected components are computed
//   assuming that all undecided borders are connecting adjacent cells. In other words, any block cannot be extended
//   beyond the potential block boundaries.
struct BoardInfo {
    units: GroupInfo,
    blocks: GroupInfo,
    potential_units: GroupInfo,
}

struct BoardManager {
    height: usize,
    width: usize,
    cell_color: Vec<Vec<i32>>,
    decision_stack: Vec<usize>,

    // borders between horizontally adjacent cells
    horizontal_borders: Vec<Vec<Border>>,

    // borders between vertically adjacent cells
    vertical_borders: Vec<Vec<Border>>,
}

impl BoardManager {
    pub fn new(cell_color: &[Vec<i32>]) -> BoardManager {
        let height = cell_color.len();
        let width = cell_color[0].len();

        BoardManager {
            height,
            width,
            cell_color: cell_color.to_vec(),
            decision_stack: vec![],
            horizontal_borders: vec![vec![Border::Undecided; width - 1]; height],
            vertical_borders: vec![vec![Border::Undecided; width]; height - 1],
        }
    }

    fn horizontal_idx(&self, y: usize, x: usize) -> usize {
        y * (self.width - 1) + x
    }

    fn vertical_idx(&self, y: usize, x: usize) -> usize {
        self.height * (self.width - 1) + y * self.width + x
    }

    fn idx_to_border(&self, idx: usize) -> (bool, usize, usize) {
        if idx >= self.height * (self.width - 1) {
            let idx = idx - self.height * (self.width - 1);
            let y = idx / self.width;
            let x = idx % self.width;
            (true, y, x)
        } else {
            let y = idx / (self.width - 1);
            let x = idx % (self.width - 1);
            (false, y, x)
        }
    }

    pub fn decide(&mut self, idx: usize, value: bool) {
        let (is_vertical, y, x) = self.idx_to_border(idx);

        if is_vertical {
            assert_eq!(self.vertical_borders[y][x], Border::Undecided);
            self.vertical_borders[y][x] = if value {
                Border::Wall
            } else {
                Border::Connected
            };
        } else {
            assert_eq!(self.horizontal_borders[y][x], Border::Undecided);
            self.horizontal_borders[y][x] = if value {
                Border::Wall
            } else {
                Border::Connected
            };
        }

        self.decision_stack.push(idx);
    }

    pub fn undo(&mut self) {
        assert!(!self.decision_stack.is_empty());
        let top = self.decision_stack.pop().unwrap();
        let (is_vertical, y, x) = self.idx_to_border(top);

        if is_vertical {
            assert_ne!(self.vertical_borders[y][x], Border::Undecided);
            self.vertical_borders[y][x] = Border::Undecided;
        } else {
            assert_ne!(self.horizontal_borders[y][x], Border::Undecided);
            self.horizontal_borders[y][x] = Border::Undecided;
        }
    }

    pub fn reason_for_unit(&self, info: &BoardInfo, unit_id: usize) -> Vec<(usize, bool)> {
        let mut ret = vec![];

        for &(y, x) in &info.units[unit_id] {
            if y < self.height - 1
                && info.units.group_id[y + 1][x] == unit_id
                && self.vertical_borders[y][x] == Border::Connected
            {
                ret.push((self.vertical_idx(y, x), false));
            }
            if x < self.width - 1
                && info.units.group_id[y][x + 1] == unit_id
                && self.horizontal_borders[y][x] == Border::Connected
            {
                ret.push((self.horizontal_idx(y, x), false));
            }
        }

        ret
    }

    pub fn reason_for_block(&self, info: &BoardInfo, block_id: usize) -> Vec<(usize, bool)> {
        let mut ret = vec![];

        for &(y, x) in &info.blocks[block_id] {
            if y < self.height - 1
                && info.blocks.group_id[y + 1][x] == block_id
                && self.vertical_borders[y][x] == Border::Connected
            {
                ret.push((self.vertical_idx(y, x), false));
            }
            if x < self.width - 1
                && info.blocks.group_id[y][x + 1] == block_id
                && self.horizontal_borders[y][x] == Border::Connected
            {
                ret.push((self.horizontal_idx(y, x), false));
            }
        }

        ret
    }

    pub fn reason_for_potential_unit_boundary(
        &self,
        info: &BoardInfo,
        unit_id: usize,
    ) -> Vec<(usize, bool)> {
        let mut ret = vec![];

        for &(y, x) in &info.potential_units[unit_id] {
            if y > 0
                && info.potential_units.group_id[y - 1][x] != unit_id
                && self.cell_color[y][x] == self.cell_color[y - 1][x]
                && self.vertical_borders[y - 1][x] == Border::Wall
            {
                ret.push((self.vertical_idx(y - 1, x), true));
            }
            if y < self.height - 1
                && info.potential_units.group_id[y + 1][x] != unit_id
                && self.cell_color[y][x] == self.cell_color[y + 1][x]
                && self.vertical_borders[y][x] == Border::Wall
            {
                ret.push((self.vertical_idx(y, x), true));
            }
            if x > 0
                && info.potential_units.group_id[y][x - 1] != unit_id
                && self.cell_color[y][x] == self.cell_color[y][x - 1]
                && self.horizontal_borders[y][x - 1] == Border::Wall
            {
                ret.push((self.horizontal_idx(y, x - 1), true));
            }
            if x < self.width - 1
                && info.potential_units.group_id[y][x + 1] != unit_id
                && self.cell_color[y][x] == self.cell_color[y][x + 1]
                && self.horizontal_borders[y][x] == Border::Wall
            {
                ret.push((self.horizontal_idx(y, x), true));
            }
        }

        ret
    }

    pub fn reason_for_path(
        &self,
        y1: usize,
        x1: usize,
        y2: usize,
        x2: usize,
    ) -> Vec<(usize, bool)> {
        let mut bfs: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; self.width]; self.height];
        bfs[y1][x1] = Some((y1, x1));

        let mut qu = std::collections::VecDeque::<(usize, usize)>::new();
        qu.push_back((y1, x1));
        while let Some((y, x)) = qu.pop_front() {
            if y == y2 && x == x2 {
                break;
            }

            if y > 0
                && self.vertical_borders[y - 1][x] == Border::Connected
                && bfs[y - 1][x].is_none()
            {
                bfs[y - 1][x] = Some((y, x));
                qu.push_back((y - 1, x));
            }
            if y < self.height - 1
                && self.vertical_borders[y][x] == Border::Connected
                && bfs[y + 1][x].is_none()
            {
                bfs[y + 1][x] = Some((y, x));
                qu.push_back((y + 1, x));
            }
            if x > 0
                && self.horizontal_borders[y][x - 1] == Border::Connected
                && bfs[y][x - 1].is_none()
            {
                bfs[y][x - 1] = Some((y, x));
                qu.push_back((y, x - 1));
            }
            if x < self.width - 1
                && self.horizontal_borders[y][x] == Border::Connected
                && bfs[y][x + 1].is_none()
            {
                bfs[y][x + 1] = Some((y, x));
                qu.push_back((y, x + 1));
            }
        }

        assert!(bfs[y2][x2].is_some());

        let mut ret = vec![];
        let mut y = y2;
        let mut x = x2;

        while !(y == y1 && x == x1) {
            let (yf, xf) = bfs[y][x].unwrap();

            if y == yf {
                ret.push((self.horizontal_idx(y, x.min(xf)), false));
            } else {
                ret.push((self.vertical_idx(y.min(yf), x), false));
            }

            y = yf;
            x = xf;
        }

        ret
    }

    pub fn compute_board_info(&self) -> BoardInfo {
        BoardInfo {
            units: self.compute_connected_components(false, false),
            blocks: self.compute_connected_components(true, false),
            potential_units: self.compute_connected_components(false, true),
        }
    }

    pub fn compute_connected_components(
        &self,
        ignore_color: bool,
        is_potential: bool,
    ) -> GroupInfo {
        let mut group_id = vec![vec![NO_GROUP; self.width]; self.height];
        let mut stack = vec![];
        let mut last_id = 0;

        for y in 0..self.height {
            for x in 0..self.width {
                if group_id[y][x] != NO_GROUP {
                    continue;
                }

                assert!(stack.is_empty());

                group_id[y][x] = last_id;
                stack.push((y, x));

                while let Some((y, x)) = stack.pop() {
                    let mut traverse = |y2: usize, x2: usize, border: Border| {
                        if !ignore_color && self.cell_color[y][x] != self.cell_color[y2][x2] {
                            return;
                        }

                        if border == Border::Connected
                            || (is_potential && border == Border::Undecided)
                        {
                            if group_id[y2][x2] == NO_GROUP {
                                group_id[y2][x2] = last_id;
                                stack.push((y2, x2));
                            }
                        }
                    };

                    if y > 0 {
                        traverse(y - 1, x, self.vertical_borders[y - 1][x]);
                    }
                    if y < self.height - 1 {
                        traverse(y + 1, x, self.vertical_borders[y][x]);
                    }
                    if x > 0 {
                        traverse(y, x - 1, self.horizontal_borders[y][x - 1]);
                    }
                    if x < self.width - 1 {
                        traverse(y, x + 1, self.horizontal_borders[y][x]);
                    }
                }

                last_id += 1;
            }
        }

        GroupInfo::new(group_id)
    }
}

struct DoublechocoConstraint {
    board: BoardManager,
    cell_color: Vec<Vec<i32>>,
    cell_num: Vec<Vec<Option<usize>>>,
}

impl SimpleCustomConstraint for DoublechocoConstraint {
    fn initialize_sat(&mut self, num_inputs: usize) {
        assert_eq!(
            num_inputs,
            self.board.height * (self.board.width - 1) + (self.board.height - 1) * self.board.width
        );
    }

    fn notify(&mut self, index: usize, value: bool) {
        self.board.decide(index, value);
    }

    fn undo(&mut self) {
        self.board.undo();
    }

    fn find_inconsistency(&mut self) -> Option<Vec<(usize, bool)>> {
        let height = self.board.height;
        let width = self.board.width;
        let info = self.board.compute_board_info();

        // connecter & size checker
        for i in 0..info.blocks.num_groups() {
            let mut num = None;
            let mut has_num = [false, false];
            let mut size_by_color = [0, 0];
            let mut potential_unit_id = [NO_GROUP, NO_GROUP];

            for &(y, x) in &info.blocks[i] {
                let c = self.cell_color[y][x] as usize;
                let pb_id = info.potential_units.group_id[y][x];

                if potential_unit_id[c] == NO_GROUP {
                    potential_unit_id[c] = pb_id;
                } else if potential_unit_id[c] != pb_id {
                    // A block contains several potential units of the same color
                    let mut ret = self.board.reason_for_block(&info, i);
                    ret.extend(self.board.reason_for_potential_unit_boundary(&info, pb_id));
                    return Some(ret);
                }

                size_by_color[c] += 1;
                let n = self.cell_num[y][x];
                if let Some(n) = n {
                    has_num[c] = true;
                    if let Some(num) = num {
                        // A block contains cells with different numbers
                        if num != n {
                            return Some(self.board.reason_for_block(&info, i));
                        }
                    } else {
                        num = Some(n);
                    }
                }
            }

            // A unit is unconditionally larger than that of the another color
            if potential_unit_id[0] != NO_GROUP
                && info.potential_units[potential_unit_id[0]].len() < size_by_color[1]
            {
                let mut ret = self.board.reason_for_block(&info, i);
                ret.extend(
                    self.board
                        .reason_for_potential_unit_boundary(&info, potential_unit_id[0]),
                );
                return Some(ret);
            }
            if potential_unit_id[1] != NO_GROUP
                && info.potential_units[potential_unit_id[1]].len() < size_by_color[0]
            {
                let mut ret = self.board.reason_for_block(&info, i);
                ret.extend(
                    self.board
                        .reason_for_potential_unit_boundary(&info, potential_unit_id[1]),
                );
                return Some(ret);
            }

            if let Some(num) = num {
                // Connected component larger than the number
                if num < size_by_color[0] || num < size_by_color[1] {
                    return Some(self.board.reason_for_block(&info, i));
                }

                // Possible connected component size smaller than the clue number
                if potential_unit_id[0] != NO_GROUP
                    && num > info.potential_units[potential_unit_id[0]].len()
                {
                    let mut ret = self
                        .board
                        .reason_for_potential_unit_boundary(&info, potential_unit_id[0]);
                    if !has_num[0] {
                        // If the unit of color 0 has no clue, inconsistency occurs only if the block is connected to the unit of color 1
                        ret.extend(self.board.reason_for_block(&info, i));
                    }
                    return Some(ret);
                }
                if potential_unit_id[1] != NO_GROUP
                    && num > info.potential_units[potential_unit_id[1]].len()
                {
                    let mut ret = self
                        .board
                        .reason_for_potential_unit_boundary(&info, potential_unit_id[1]);
                    if !has_num[1] {
                        ret.extend(self.board.reason_for_block(&info, i));
                    }
                    return Some(ret);
                }
            }
        }

        for y in 0..height {
            for x in 0..width {
                // Extra walls between cells in the same block
                if y < height - 1
                    && info.blocks.group_id[y][x] == info.blocks.group_id[y + 1][x]
                    && self.board.vertical_borders[y][x] == Border::Wall
                {
                    let mut ret = self.board.reason_for_path(y, x, y + 1, x);
                    ret.push((self.board.vertical_idx(y, x), true));
                    return Some(ret);
                }
                if x < width - 1
                    && info.blocks.group_id[y][x] == info.blocks.group_id[y][x + 1]
                    && self.board.horizontal_borders[y][x] == Border::Wall
                {
                    let mut ret = self.board.reason_for_path(y, x, y, x + 1);
                    ret.push((self.board.horizontal_idx(y, x), true));
                    return Some(ret);
                }
            }
        }

        let mut adjacent_potential_units_flat = vec![];
        for y in 0..height {
            for x in 0..width {
                if y < height - 1
                    && self.cell_color[y][x] != self.cell_color[y + 1][x]
                    && self.board.vertical_borders[y][x] != Border::Wall
                {
                    let i = info.potential_units.group_id[y][x];
                    let j = info.potential_units.group_id[y + 1][x];
                    adjacent_potential_units_flat.push((i, j));
                    adjacent_potential_units_flat.push((j, i));
                }

                if x < width - 1
                    && self.cell_color[y][x] != self.cell_color[y][x + 1]
                    && self.board.horizontal_borders[y][x] != Border::Wall
                {
                    let i = info.potential_units.group_id[y][x];
                    let j = info.potential_units.group_id[y][x + 1];
                    adjacent_potential_units_flat.push((i, j));
                    adjacent_potential_units_flat.push((j, i));
                }
            }
        }

        adjacent_potential_units_flat.sort();
        adjacent_potential_units_flat.dedup();

        let mut adjacent_potential_units = vec![vec![]; info.potential_units.num_groups()];
        for &(i, j) in &adjacent_potential_units_flat {
            adjacent_potential_units[i].push(j);
        }

        for i in 0..info.units.num_groups() {
            let mut cells = vec![];
            let mut connections = vec![];

            // Note: by construction, info.units[i] is sorted

            for &(y, x) in &info.units[i] {
                cells.push((y as i32, x as i32));

                if y < height - 1 && info.units.group_id[y + 1][x] == i {
                    connections.push((y as i32 * 2 + 1, x as i32 * 2));
                }
                if x < width - 1 && info.units.group_id[y][x + 1] == i {
                    connections.push((y as i32 * 2, x as i32 * 2 + 1));
                }
            }

            let mut shape = Shape { cells, connections };
            shape.normalize();
            assert!(shape.is_invariant_met());

            let one_cell = info.units[i][0];
            let potential_unit_id = info.potential_units.group_id[one_cell.0][one_cell.1];
            let mut origins = vec![];
            for &g in &adjacent_potential_units[potential_unit_id] {
                for &p in &info.potential_units[g] {
                    origins.push(p);
                }
            }

            let transforms = enumerate_transforms(&shape);
            let mut found = false;
            let mut blockers = vec![];

            'outer: for tr in &transforms {
                for &(oy, ox) in &origins {
                    let mut is_invalid = false;
                    let mut blocker_cand = None;

                    for &(dy, dx) in &tr.connections {
                        let py = oy as i32 * 2 + dy;
                        let px = ox as i32 * 2 + dx;

                        if !(0 <= py
                            && py <= (height - 1) as i32 * 2
                            && 0 <= px
                            && px <= (width - 1) as i32 * 2)
                        {
                            is_invalid = true;
                            blocker_cand = None;
                            break;
                        }
                        let py = py as usize;
                        let px = px as usize;
                        if self.cell_color[py >> 1][px >> 1]
                            != self.cell_color[(py + 1) >> 1][(px + 1) >> 1]
                        {
                            is_invalid = true;
                            blocker_cand = None;
                            break;
                        }
                        if (py & 1) == 1 {
                            if self.board.vertical_borders[py >> 1][px >> 1] == Border::Wall {
                                is_invalid = true;
                                blocker_cand =
                                    Some((self.board.vertical_idx(py >> 1, px >> 1), true));
                            }
                        } else {
                            if self.board.horizontal_borders[py >> 1][px >> 1] == Border::Wall {
                                is_invalid = true;
                                blocker_cand =
                                    Some((self.board.horizontal_idx(py >> 1, px >> 1), true));
                            }
                        }
                    }

                    if !is_invalid {
                        found = true;
                        break 'outer;
                    }
                    blockers.extend(blocker_cand);
                }
            }

            if !found {
                let mut reason = blockers;
                reason.extend(self.board.reason_for_unit(&info, i));
                reason.extend(
                    self.board
                        .reason_for_potential_unit_boundary(&info, potential_unit_id),
                );
                for &g in &adjacent_potential_units[potential_unit_id] {
                    reason.extend(self.board.reason_for_potential_unit_boundary(&info, g));
                }

                for &(y, x) in &info.potential_units[potential_unit_id] {
                    if y > 0
                        && self.board.vertical_borders[y - 1][x] == Border::Wall
                        && self.cell_color[y][x] != self.cell_color[y - 1][x]
                        && adjacent_potential_units_flat
                            .binary_search(&(
                                potential_unit_id,
                                info.potential_units.group_id[y - 1][x],
                            ))
                            .is_err()
                    {
                        reason.push((self.board.vertical_idx(y - 1, x), true));
                    }
                    if y < height - 1
                        && self.board.vertical_borders[y][x] == Border::Wall
                        && self.cell_color[y][x] != self.cell_color[y + 1][x]
                        && adjacent_potential_units_flat
                            .binary_search(&(
                                potential_unit_id,
                                info.potential_units.group_id[y + 1][x],
                            ))
                            .is_err()
                    {
                        reason.push((self.board.vertical_idx(y, x), true));
                    }
                    if x > 0
                        && self.board.horizontal_borders[y][x - 1] == Border::Wall
                        && self.cell_color[y][x] != self.cell_color[y][x - 1]
                        && adjacent_potential_units_flat
                            .binary_search(&(
                                potential_unit_id,
                                info.potential_units.group_id[y][x - 1],
                            ))
                            .is_err()
                    {
                        reason.push((self.board.horizontal_idx(y, x - 1), true));
                    }
                    if x < width - 1
                        && self.board.horizontal_borders[y][x] == Border::Wall
                        && self.cell_color[y][x] != self.cell_color[y][x + 1]
                        && adjacent_potential_units_flat
                            .binary_search(&(
                                potential_unit_id,
                                info.potential_units.group_id[y][x + 1],
                            ))
                            .is_err()
                    {
                        reason.push((self.board.horizontal_idx(y, x), true));
                    }
                }
                return Some(reason);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_transforms() {
        // TODO: add more shapes for testing
        {
            let shape = Shape {
                cells: vec![(0, 0)],
                connections: vec![],
            };

            assert_eq!(enumerate_transforms(&shape), vec![shape]);
        }

        {
            let shape = Shape {
                cells: vec![(0, 0), (0, 1)],
                connections: vec![(0, 1)],
            };

            assert_eq!(
                enumerate_transforms(&shape),
                vec![
                    shape,
                    Shape {
                        cells: vec![(0, 0), (1, 0)],
                        connections: vec![(1, 0)],
                    },
                ]
            );
        }

        {
            let shape = Shape {
                cells: vec![(0, 0), (0, 1), (1, 0)],
                connections: vec![(0, 1), (1, 0)],
            };

            assert_eq!(enumerate_transforms(&shape).len(), 4);
        }

        {
            let shape = Shape {
                cells: vec![(0, 0), (0, 1), (0, 2), (1, 0)],
                connections: vec![(0, 1), (0, 3), (1, 0)],
            };

            assert_eq!(enumerate_transforms(&shape).len(), 8);
        }
    }

    fn problem_for_tests() -> Problem {
        (
            vec![
                vec![1, 1, 0, 0, 1, 1],
                vec![1, 1, 1, 0, 0, 1],
                vec![0, 0, 1, 1, 0, 0],
                vec![0, 1, 1, 0, 0, 0],
                vec![0, 1, 1, 0, 0, 1],
                vec![0, 1, 1, 1, 0, 0],
            ],
            vec![
                vec![Some(5), None, None, None, None, None],
                vec![None, None, None, None, None, None],
                vec![None, None, None, None, None, None],
                vec![None, None, None, None, None, None],
                vec![None, None, None, None, None, None],
                vec![None, None, None, None, None, None],
            ],
        )
    }

    #[test]
    fn test_doublechoco_problem() {
        let (color, num) = problem_for_tests();
        let ans = solve_doublechoco(&color, &num);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::InnerGridEdges {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 0, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 1, 1],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_doublechoco_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?dbchoco/6/6/pu9hgpe05zu";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }

    #[test]
    #[ignore]
    fn test_doublechoco_stress_test() {
        let urls = [
            // https://puzsq.logicpuzzle.app/puzzle/22054
            "https://puzz.link/p?dbchoco/8/8/01v4sjjufpv00p76g9zg4zn",
            // https://puzsq.logicpuzzle.app/puzzle/22059
            "https://puzz.link/p?dbchoco/10/10/v0v30r0rororooo0rs3szx8g7k6zx9t",
        ];

        for url in urls {
            let (color, num) = deserialize_problem(url).unwrap();
            let ans = solve_doublechoco(&color, &num);
            assert!(ans.is_some());
        }
    }
}
