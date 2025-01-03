use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url_with_context_and_site, url_to_problem, Choice, Combinator, Context, Dict, Grid,
    HexInt, Optionalize, Spaces,
};
use crate::solver::{Solver, TRUE};
use std::collections::VecDeque;

use cspuz_core::custom_constraints::SimpleCustomConstraint;

pub fn solve_archipelago(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    let mut maxn = 0;
    for i in 1.. {
        let s = i * (i + 1) / 2;
        if s <= h * w {
            maxn = i as i32;
        } else {
            break;
        }
    }

    let nums = &solver.int_var_2d((h, w), 1, maxn);
    solver.add_expr((!is_black).imp(nums.eq(1)));

    let is_border = graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_expr(is_black.conv2d_and((1, 2)) ^ &is_border.vertical);
    solver.add_expr(is_black.conv2d_and((2, 1)) ^ &is_border.horizontal);
    graph::graph_division_2d(&mut solver, nums, &is_border);

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(is_black.at((y, x)));

                if n > 0 {
                    solver.add_expr(nums.at((y, x)).eq(n));
                }
            }
        }
    }

    let mut aux_graph = graph::Graph::new(h * w * 2 + 1);
    for y in 0..h {
        for x in 0..w {
            let p = y * w + x;
            aux_graph.add_edge(p, p + h * w);
            aux_graph.add_edge(p + h * w, h * w * 2);

            if y < h - 1 {
                aux_graph.add_edge(p, p + w);
            }
            if x < w - 1 {
                aux_graph.add_edge(p, p + 1);
            }
            if y < h - 1 && x < w - 1 {
                aux_graph.add_edge(p, p + w + 1);
            }
            if y < h - 1 && x > 0 {
                aux_graph.add_edge(p, p + w - 1);
            }
        }
    }

    for n in 1..=maxn {
        let reachable = &solver.bool_var_2d((h, w));
        solver.add_expr(reachable.imp(is_black));

        if n == 2 {
            solver.add_expr(is_black.imp(reachable));
        } else {
            solver.add_expr(nums.gt(n).imp(reachable));
        }

        let vertices = reachable
            .expr()
            .into_iter()
            .chain((is_black & nums.eq(n)).into_iter())
            .chain([TRUE].into_iter())
            .collect::<Vec<_>>();
        graph::active_vertices_connected(&mut solver, &vertices, &mut aux_graph);
    }

    #[cfg(not(test))]
    {
        solver.add_custom_constraint(Box::new(ArchipelagoConstraint::new(h, w)), is_black);
    }

    #[cfg(test)]
    {
        solver.add_custom_constraint(
            Box::new(util::tests::ReasonVerifier::new(
                ArchipelagoConstraint::new(h, w),
                ArchipelagoConstraint::new(h, w),
            )),
            is_black,
        );
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum ArchipelagoCell {
    White,
    Black,
    Undecided,
}

struct ArchipelagoConstraint {
    height: usize,
    width: usize,
    board: Vec<Vec<ArchipelagoCell>>,
    decision_stack: Vec<(usize, usize)>,
}

impl ArchipelagoConstraint {
    fn new(height: usize, width: usize) -> ArchipelagoConstraint {
        ArchipelagoConstraint {
            height,
            width,
            board: vec![vec![ArchipelagoCell::Undecided; width]; height],
            decision_stack: vec![],
        }
    }
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

impl SimpleCustomConstraint for ArchipelagoConstraint {
    fn initialize_sat(&mut self, num_inputs: usize) {
        assert_eq!(num_inputs, self.height * self.width);
    }

    fn notify(&mut self, index: usize, value: bool) {
        let y = index / self.width;
        let x = index % self.width;
        self.board[y][x] = if value {
            ArchipelagoCell::Black
        } else {
            ArchipelagoCell::White
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
                if chain_id[y][x] != !0 || self.board[y][x] != ArchipelagoCell::Black {
                    continue;
                }

                assert!(queue.is_empty());
                queue.push_back((y, x));
                chain_id[y][x] = last_id as i32;
                while !queue.is_empty() {
                    let (y, x) = queue.pop_front().unwrap();
                    for &(dy, dx) in &EIGHT_DIRECTIONS {
                        if let Some((ny, nx)) = offset((y, x), (dy, dx), height, width) {
                            if self.board[ny][nx] == ArchipelagoCell::Black
                                && chain_id[ny][nx] == !0
                            {
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
                if block_id[y][x] != -1 || self.board[y][x] != ArchipelagoCell::Black {
                    continue;
                }

                let mut block_size = 0;
                let mut is_closed = true;

                assert!(queue.is_empty());
                queue.push_back((y, x));
                block_id[y][x] = last_block_id;
                while !queue.is_empty() {
                    let (y, x) = queue.pop_front().unwrap();
                    block_size += 1;
                    for &(dy, dx) in &FOUR_DIRECTIONS {
                        if let Some((ny, nx)) = offset((y, x), (dy, dx), height, width) {
                            if self.board[ny][nx] == ArchipelagoCell::Black {
                                if block_id[ny][nx] == -1 {
                                    block_id[ny][nx] = last_block_id;
                                    queue.push_back((ny, nx));
                                }
                            } else if self.board[ny][nx] == ArchipelagoCell::Undecided {
                                is_closed = false;
                            }
                        }
                    }
                }
                if is_closed {
                    blocks_by_chain[chain_id[y][x] as usize].push((block_size, last_block_id));
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
                        if self.board[y][x] == ArchipelagoCell::Black
                            && (block_id[y][x] == u || block_id[y][x] == v)
                        {
                            if block_id[y][x] == u {
                                queue.push_back((y, x));
                                origin[y][x] = Some((!0, !0));
                            }
                            ret.push((y * width + x, true));
                        } else if self.board[y][x] == ArchipelagoCell::White {
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
                            if self.board[ny][nx] == ArchipelagoCell::Black
                                && origin[ny][nx].is_none()
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
        self.board[y][x] = ArchipelagoCell::Undecided;
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
    let height = problem.len();
    let width = problem[0].len();
    problem_to_url_with_context_and_site(
        combinator(),
        "archipelago",
        "https://pzprxs.vercel.app/p?",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["archipelago"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        vec![
            vec![None, Some(3), None, None, None, None],
            vec![Some(2), None, None, None, None, None],
            vec![None, None, None, None, None, None],
            vec![None, None, Some(2), None, Some(3), None],
            vec![None, Some(-1), None, None, None, None],
        ]
    }

    #[test]
    fn test_archipelago_problem() {
        let problem = problem_for_tests();

        let ans = solve_archipelago(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_archipelago_serializer() {
        let problem = problem_for_tests();
        let url = "https://pzprxs.vercel.app/p?archipelago/6/5/g3j2s2g3h.j";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
