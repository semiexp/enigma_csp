use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::{any, Solver};
use std::collections::VecDeque;

use enigma_csp::custom_constraints::SimpleCustomConstraint;

pub fn solve_chainedb(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(clues);
    if h < 2 || w < 2 {
        // a block cannot touch another block
        return None;
    }

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    let mut clue_pos = vec![];
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                clue_pos.push((y, x, n));
            }
        }
    }

    let group_id = solver.int_var_2d((h, w), -1, clue_pos.len() as i32 - 1);
    solver.add_expr(is_black ^ group_id.eq(-1));

    solver.add_expr(
        is_black.conv2d_and((2, 1)).imp(
            group_id
                .slice((..(h - 1), ..))
                .eq(group_id.slice((1.., ..))),
        ),
    );
    solver.add_expr(
        is_black.conv2d_and((1, 2)).imp(
            group_id
                .slice((.., ..(w - 1)))
                .eq(group_id.slice((.., 1..))),
        ),
    );

    let incident_to_another_block = solver.bool_var_2d((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut cond = vec![];

            if y > 0 && x > 0 {
                cond.push(
                    is_black.at((y, x))
                        & !is_black.at((y - 1, x))
                        & !is_black.at((y, x - 1))
                        & is_black.at((y - 1, x - 1))
                        & group_id.at((y, x)).ne(group_id.at((y - 1, x - 1))),
                );
            }
            if y > 0 && x + 1 < w {
                cond.push(
                    is_black.at((y, x))
                        & !is_black.at((y - 1, x))
                        & !is_black.at((y, x + 1))
                        & is_black.at((y - 1, x + 1))
                        & group_id.at((y, x)).ne(group_id.at((y - 1, x + 1))),
                );
            }
            if y + 1 < h && x > 0 {
                cond.push(
                    is_black.at((y, x))
                        & !is_black.at((y + 1, x))
                        & !is_black.at((y, x - 1))
                        & is_black.at((y + 1, x - 1))
                        & group_id.at((y, x)).ne(group_id.at((y + 1, x - 1))),
                );
            }
            if y + 1 < h && x + 1 < w {
                cond.push(
                    is_black.at((y, x))
                        & !is_black.at((y + 1, x))
                        & !is_black.at((y, x + 1))
                        & is_black.at((y + 1, x + 1))
                        & group_id.at((y, x)).ne(group_id.at((y + 1, x + 1))),
                );
            }

            solver.add_expr(incident_to_another_block.at((y, x)).iff(any(cond)));
        }
    }

    for i in 0..clue_pos.len() {
        graph::active_vertices_connected_2d(&mut solver, group_id.eq(i as i32));
        solver.add_expr((group_id.eq(i as i32) & &incident_to_another_block).any());

        let (y, x, n) = clue_pos[i];
        solver.add_expr(group_id.at((y, x)).eq(i as i32));
        if n > 0 {
            solver.add_expr(group_id.eq(i as i32).count_true().eq(n));
        }
    }

    #[cfg(not(test))]
    {
        solver.add_custom_constraint(Box::new(ChainedbConstraint::new(h, w)), is_black);
    }

    #[cfg(test)]
    {
        solver.add_custom_constraint(
            Box::new(util::tests::ReasonVerifier::new(
                ChainedbConstraint::new(h, w),
                ChainedbConstraint::new(h, w),
            )),
            is_black,
        );
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum ChainedbCell {
    White,
    Black,
    Undecided,
}

struct ChainedbConstraint {
    height: usize,
    width: usize,
    board: Vec<Vec<ChainedbCell>>,
    decision_stack: Vec<(usize, usize)>,
}

impl ChainedbConstraint {
    fn new(height: usize, width: usize) -> ChainedbConstraint {
        ChainedbConstraint {
            height,
            width,
            board: vec![vec![ChainedbCell::Undecided; width]; height],
            decision_stack: vec![],
        }
    }
}

fn adjust_bbox(block: &mut [(i32, i32)]) {
    let mut min_y = std::i32::MAX;
    let mut min_x = std::i32::MAX;
    for &(y, x) in block.iter() {
        min_y = min_y.min(y);
        min_x = min_x.min(x);
    }
    for p in block.iter_mut() {
        p.0 -= min_y;
        p.1 -= min_x;
    }
}

fn flip_block(block: &[(i32, i32)]) -> Vec<(i32, i32)> {
    let mut ymax = 0;
    for &(y, _) in block.iter() {
        ymax = ymax.max(y);
    }
    let mut ret = block
        .iter()
        .map(|&(y, x)| (ymax - y, x))
        .collect::<Vec<_>>();
    ret.sort();
    ret
}

fn rotate_block(block: &[(i32, i32)]) -> Vec<(i32, i32)> {
    let mut ymax = 0;
    for &(y, _) in block.iter() {
        ymax = ymax.max(y);
    }

    let mut ret = block
        .iter()
        .map(|&(y, x)| (x, ymax - y))
        .collect::<Vec<_>>();
    ret.sort();
    ret
}

fn normalize_block(mut block: Vec<(i32, i32)>) -> Vec<(i32, i32)> {
    adjust_bbox(&mut block);
    block.sort();

    let mut ret = block.clone();
    for i in 0..4 {
        ret = ret.min(block.clone());
        ret = ret.min(flip_block(&block));
        if i < 3 {
            block = rotate_block(&block);
        }
    }

    ret
}

fn offset(
    (y, x): (usize, usize),
    (dy, dx): (i32, i32),
    height: usize,
    width: usize,
) -> Option<(usize, usize)> {
    let ny = y as i32 + dy;
    let nx = x as i32 + dx;
    if ny < 0 || ny >= height as i32 || nx < 0 || nx >= width as i32 {
        None
    } else {
        Some((ny as usize, nx as usize))
    }
}

const FOUR_DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const EIGHT_DIRECTIONS: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

impl SimpleCustomConstraint for ChainedbConstraint {
    fn initialize_sat(&mut self, num_inputs: usize) {
        assert_eq!(num_inputs, self.height * self.width);
    }

    fn notify(&mut self, index: usize, value: bool) {
        let y = index / self.width;
        let x = index % self.width;
        self.board[y][x] = if value {
            ChainedbCell::Black
        } else {
            ChainedbCell::White
        };
        self.decision_stack.push((y, x));
    }

    fn find_inconsistency(&mut self) -> Option<Vec<(usize, bool)>> {
        let height = self.height;
        let width = self.width;

        let mut chain_id = vec![vec![!0; self.width]; self.height];
        let mut last_id = 0;

        let mut queue = VecDeque::new();
        for y in 0..height {
            for x in 0..width {
                if chain_id[y][x] != !0 || self.board[y][x] != ChainedbCell::Black {
                    continue;
                }

                assert!(queue.is_empty());
                queue.push_back((y, x));
                chain_id[y][x] = last_id as i32;
                while !queue.is_empty() {
                    let (y, x) = queue.pop_front().unwrap();
                    for &(dy, dx) in &EIGHT_DIRECTIONS {
                        if let Some((ny, nx)) = offset((y, x), (dy, dx), height, width) {
                            if self.board[ny][nx] == ChainedbCell::Black && chain_id[ny][nx] == !0 {
                                chain_id[ny][nx] = last_id as i32;
                                queue.push_back((ny, nx));
                            }
                        }
                    }
                }
                last_id += 1;
            }
        }

        let mut block_id = vec![vec![-1; self.width]; self.height];
        let mut blocks_by_chain = vec![vec![]; last_id];
        let mut last_block_id = 0i32;
        for y in 0..height {
            for x in 0..width {
                if block_id[y][x] != -1 || self.board[y][x] != ChainedbCell::Black {
                    continue;
                }

                let mut block = vec![];
                let mut is_closed = true;

                assert!(queue.is_empty());
                queue.push_back((y, x));
                block_id[y][x] = last_block_id;
                while !queue.is_empty() {
                    let (y, x) = queue.pop_front().unwrap();
                    block.push((y as i32, x as i32));
                    for &(dy, dx) in &FOUR_DIRECTIONS {
                        if let Some((ny, nx)) = offset((y, x), (dy, dx), height, width) {
                            if self.board[ny][nx] == ChainedbCell::Black {
                                if block_id[ny][nx] == -1 {
                                    block_id[ny][nx] = last_block_id;
                                    queue.push_back((ny, nx));
                                }
                            } else if self.board[ny][nx] == ChainedbCell::Undecided {
                                is_closed = false;
                            }
                        }
                    }
                }
                if is_closed {
                    blocks_by_chain[chain_id[y][x] as usize]
                        .push((normalize_block(block), last_block_id));
                }
                last_block_id += 1;
            }
        }

        for i in 0..last_id {
            blocks_by_chain[i].sort();
            let mut duplicate_block_id = None;
            for j in 1..blocks_by_chain[i].len() {
                if blocks_by_chain[i][j].0 == blocks_by_chain[i][j - 1].0 {
                    duplicate_block_id =
                        Some((blocks_by_chain[i][j].1, blocks_by_chain[i][j - 1].1));
                    break;
                }
            }

            if let Some((u, v)) = duplicate_block_id {
                let mut ret = vec![];

                let mut origin = vec![vec![None; width]; height];
                assert!(queue.is_empty());

                for y in 0..height {
                    for x in 0..width {
                        if self.board[y][x] == ChainedbCell::Black
                            && (block_id[y][x] == u || block_id[y][x] == v)
                        {
                            if block_id[y][x] == u {
                                queue.push_back((y, x));
                                origin[y][x] = Some((!0, !0));
                            }
                            ret.push((y * width + x, true));
                        } else if self.board[y][x] == ChainedbCell::White {
                            let mut is_neighbor = false;
                            for &(dy, dx) in &FOUR_DIRECTIONS {
                                if let Some((ny, nx)) = offset((y, x), (dy, dx), height, width) {
                                    if block_id[ny][nx] == u || block_id[ny][nx] == v {
                                        is_neighbor = true;
                                    }
                                }
                            }
                            if is_neighbor {
                                ret.push((y * width + x, false));
                            }
                        }
                    }
                }

                let mut reached = false;
                while !queue.is_empty() {
                    let (y, x) = queue.pop_front().unwrap();
                    if block_id[y][x] == v {
                        let (mut y, mut x) = origin[y][x].unwrap();
                        loop {
                            let (ny, nx) = origin[y][x].unwrap();
                            if ny == !0 {
                                break;
                            }
                            ret.push((y * width + x, true));
                            y = ny;
                            x = nx;
                        }
                        reached = true;
                        break;
                    }

                    for &(dy, dx) in &EIGHT_DIRECTIONS {
                        if let Some((ny, nx)) = offset((y, x), (dy, dx), height, width) {
                            if self.board[ny][nx] == ChainedbCell::Black && origin[ny][nx].is_none()
                            {
                                queue.push_back((ny, nx));
                                origin[ny][nx] = Some((y, x));
                            }
                        }
                    }
                }
                assert!(reached);

                return Some(ret);
            }
        }

        None
    }

    fn undo(&mut self) {
        let (y, x) = self.decision_stack.pop().unwrap();
        self.board[y][x] = ChainedbCell::Undecided;
    }
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Optionalize::new(HexInt)),
        Box::new(Spaces::new(None, 'g')),
        Box::new(Dict::new(Some(-1), ".")),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "chainedb", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["chainedb"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chainedb_normalize_block() {
        let block = vec![(1, 2), (2, 2), (2, 1), (2, 0)];

        let normalized = normalize_block(block.clone());
        assert_eq!(normalized, vec![(0, 0), (0, 1), (0, 2), (1, 0),]);
    }

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        vec![
            vec![Some(3), None, Some(3), None, None, None],
            vec![None, None, None, None, None, None],
            vec![None, None, None, None, None, Some(3)],
            vec![None, None, None, None, None, None],
            vec![None, Some(1), None, None, Some(-1), None],
        ]
    }

    #[test]
    fn test_chainedb_problem() {
        let problem = problem_for_tests();

        let ans = solve_chainedb(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [1, 0, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_chainedb_problem2() {
        let problem = deserialize_problem("https://puzz.link/p?chainedb/4/3/g7p").unwrap();
        let ans = solve_chainedb(&problem);
        assert!(ans.is_none());
    }

    #[test]
    fn test_chainedb_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?chainedb/6/5/3g3t3m1h.g";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
