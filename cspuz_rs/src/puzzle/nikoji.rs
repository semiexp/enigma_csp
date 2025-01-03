use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::Solver;

use cspuz_core::custom_constraints::SimpleCustomConstraint;

pub fn solve_nikoji(
    clues: &[Vec<Option<i32>>],
) -> Option<graph::BoolInnerGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_border = &graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&is_border.horizontal);
    solver.add_answer_key_bool(&is_border.vertical);

    let mut symbols = vec![];
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                symbols.push((n, y, x));
            }
        }
    }
    symbols.sort();

    let cell_group = &solver.int_var_2d((h, w), 0, symbols.len() as i32 - 1);
    solver.add_expr(
        cell_group
            .slice((.., ..(w - 1)))
            .ne(cell_group.slice((.., 1..)))
            .iff(&is_border.vertical),
    );
    solver.add_expr(
        cell_group
            .slice((..(h - 1), ..))
            .ne(cell_group.slice((1.., ..)))
            .iff(&is_border.horizontal),
    );

    for i in 0..symbols.len() {
        graph::active_vertices_connected_2d(&mut solver, cell_group.eq(i as i32));

        let (_, y, x) = symbols[i];
        solver.add_expr(cell_group.at((y as usize, x as usize)).eq(i as i32));
    }

    let mut leader_ids = vec![];
    let mut p = 0;
    while p < symbols.len() {
        let mut end = p + 1;
        while end < symbols.len() && symbols[p].0 == symbols[end].0 {
            end += 1;
        }
        leader_ids.push(p);

        for q in (p + 1)..end {
            let (_, py, px) = symbols[p];
            let (_, qy, qx) = symbols[q];
            let py = py as i32;
            let px = px as i32;
            let qy = qy as i32;
            let qx = qx as i32;

            let dy = qy - py;
            let dx = qx - px;

            for y1 in 0..(h as i32) {
                for x1 in 0..(w as i32) {
                    let y2 = y1 + dy;
                    let x2 = x1 + dx;

                    if 0 <= y2 && y2 < h as i32 && 0 <= x2 && x2 < w as i32 {
                        solver.add_expr(
                            cell_group
                                .at((y1 as usize, x1 as usize))
                                .eq(p as i32)
                                .iff(cell_group.at((y2 as usize, x2 as usize)).eq(q as i32)),
                        );
                    } else {
                        solver.add_expr(cell_group.at((y1 as usize, x1 as usize)).ne(p as i32));
                    }

                    let y0 = y1 - dy;
                    let x0 = x1 - dx;
                    if !(0 <= y0 && y0 < h as i32 && 0 <= x0 && x0 < w as i32) {
                        solver.add_expr(cell_group.at((y1 as usize, x1 as usize)).ne(q as i32));
                    }
                }
            }
        }

        p = end;
    }

    let mut leader_indicators = vec![];
    for i in 0..leader_ids.len() {
        leader_indicators.push(
            cell_group
                .eq(leader_ids[i] as i32)
                .flatten()
                .into_iter()
                .collect::<Vec<_>>(),
        );
    }

    for i in 0..leader_ids.len() {
        for j in 0..i {
            let cells = leader_indicators[i]
                .iter()
                .chain(leader_indicators[j].iter())
                .cloned()
                .collect::<Vec<_>>();

            #[cfg(not(test))]
            {
                solver.add_custom_constraint(Box::new(DifferentShape::new(h, w)), cells);
            }

            #[cfg(test)]
            {
                solver.add_custom_constraint(
                    Box::new(util::tests::ReasonVerifier::new(
                        DifferentShape::new(h, w),
                        DifferentShape::new(h, w),
                    )),
                    cells,
                );
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_border))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CellState {
    White,
    Black,
    Undecided,
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

struct DifferentShape {
    height: usize,
    width: usize,
    boards: [Vec<Vec<CellState>>; 2],
    decision_stack: Vec<(usize, usize, usize)>,
}

impl DifferentShape {
    fn new(height: usize, width: usize) -> DifferentShape {
        DifferentShape {
            height,
            width,
            boards: [
                vec![vec![CellState::Undecided; width]; height],
                vec![vec![CellState::Undecided; width]; height],
            ],
            decision_stack: vec![],
        }
    }
}

impl SimpleCustomConstraint for DifferentShape {
    fn lazy_propagation(&self) -> bool {
        true
    }

    fn initialize_sat(&mut self, num_inputs: usize) {
        assert_eq!(num_inputs, self.height * self.width * 2);
    }

    fn notify(&mut self, index: usize, value: bool) {
        let (y, x, board) = (
            (index / self.width) % self.height,
            index % self.width,
            index / (self.height * self.width),
        );
        self.boards[board][y][x] = if value {
            CellState::Black
        } else {
            CellState::White
        };
        self.decision_stack.push((y, x, board));
    }

    fn find_inconsistency(&mut self) -> Option<Vec<(usize, bool)>> {
        let height = self.height;
        let width = self.width;

        for b in 0..2 {
            let mut has_black = false;

            for y in 0..height {
                for x in 0..width {
                    if self.boards[b][y][x] == CellState::Black {
                        has_black = true;

                        if y > 0 && self.boards[b][y - 1][x] == CellState::Undecided {
                            return None;
                        }
                        if y < height - 1 && self.boards[b][y + 1][x] == CellState::Undecided {
                            return None;
                        }
                        if x > 0 && self.boards[b][y][x - 1] == CellState::Undecided {
                            return None;
                        }
                        if x < width - 1 && self.boards[b][y][x + 1] == CellState::Undecided {
                            return None;
                        }
                    }
                }
            }

            if !has_black {
                return None;
            }
        }

        let mut shapes = [vec![], vec![]];
        for b in 0..2 {
            let mut ymin = height;
            let mut ymax = 0;
            let mut xmin = width;
            let mut xmax = 0;

            for y in 0..height {
                for x in 0..width {
                    if self.boards[b][y][x] == CellState::Black {
                        ymin = ymin.min(y);
                        ymax = ymax.max(y);
                        xmin = xmin.min(x);
                        xmax = xmax.max(x);
                    }
                }
            }

            assert!(ymin <= ymax);
            assert!(xmin <= xmax);

            let mut shape = vec![];
            for y in ymin..=ymax {
                for x in xmin..=xmax {
                    if self.boards[b][y][x] == CellState::Black {
                        shape.push(((y - ymin) as i32, (x - xmin) as i32));
                    }
                }
            }

            shapes[b] = normalize_block(shape);
        }

        if shapes[0] != shapes[1] {
            return None;
        }

        // inconsistent (same shape)
        let mut reason = vec![];
        for b in 0..2 {
            for y in 0..height {
                for x in 0..width {
                    match self.boards[b][y][x] {
                        CellState::Black => {
                            reason.push((y * width + x + b * height * width, true));
                        }
                        CellState::White => {
                            // add to the reason if it is adjacent to a black cell
                            let mut is_adjacent = false;

                            if y > 0 && self.boards[b][y - 1][x] == CellState::Black {
                                is_adjacent = true;
                            }
                            if y < height - 1 && self.boards[b][y + 1][x] == CellState::Black {
                                is_adjacent = true;
                            }
                            if x > 0 && self.boards[b][y][x - 1] == CellState::Black {
                                is_adjacent = true;
                            }
                            if x < width - 1 && self.boards[b][y][x + 1] == CellState::Black {
                                is_adjacent = true;
                            }

                            if is_adjacent {
                                reason.push((y * width + x + b * height * width, false));
                            }
                        }
                        CellState::Undecided => {}
                    }
                }
            }
        }

        Some(reason)
    }

    fn undo(&mut self) {
        let (y, x, board) = self.decision_stack.pop().unwrap();
        self.boards[board][y][x] = CellState::Undecided;
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
    problem_to_url(combinator(), "nikoji", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["nikoji"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![Some(1), None, None, None, Some(3), None],
            vec![None, None, None, None, None, None],
            vec![Some(2), Some(1), None, Some(5), None, None],
            vec![Some(3), None, None, None, Some(4), None],
            vec![None, Some(4), None, Some(2), Some(1), Some(1)],
        ]
    }

    #[test]
    fn test_nikoji_problem() {
        let problem = problem_for_tests();
        let ans = solve_nikoji(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolInnerGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 0, 0, 0, 1],
                [0, 1, 1, 0, 1, 0],
                [1, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 0, 1, 0],
                [1, 0, 0, 1, 1],
                [1, 1, 1, 0, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 1, 1, 1],
            ]),
        };

        assert_eq!(ans, expected);
    }

    #[test]
    fn test_nikoji_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?nikoji/6/5/1i3m21g5h3i4h4g211";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
