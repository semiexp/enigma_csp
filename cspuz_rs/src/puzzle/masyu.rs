use super::util;
use crate::graph;
use crate::serializer::{problem_to_url, url_to_problem, Combinator, Grid, Map, MultiDigit};
use crate::solver::Solver;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MasyuClue {
    None,
    White,
    Black,
}

pub fn solve_masyu(clues: &[Vec<MasyuClue>]) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    graph::single_cycle_grid_edges(&mut solver, &is_line);

    for y in 0..h {
        for x in 0..w {
            let p = (y, x);
            match clues[y][x] {
                MasyuClue::None => (),
                MasyuClue::White => {
                    solver.add_expr(
                        (is_line.vertical.at_offset(p, (-1, 0), false)
                            & is_line.vertical.at_offset(p, (0, 0), false)
                            & !(is_line.vertical.at_offset(p, (-2, 0), false)
                                & is_line.vertical.at_offset(p, (1, 0), false)))
                            | (is_line.horizontal.at_offset(p, (0, -1), false)
                                & is_line.horizontal.at_offset(p, (0, 0), false)
                                & !(is_line.horizontal.at_offset(p, (0, -2), false)
                                    & is_line.horizontal.at_offset(p, (0, 1), false))),
                    );
                }
                MasyuClue::Black => {
                    solver.add_expr(
                        (is_line.vertical.at_offset(p, (-2, 0), false)
                            & is_line.vertical.at_offset(p, (-1, 0), false))
                            | (is_line.vertical.at_offset(p, (0, 0), false)
                                & is_line.vertical.at_offset(p, (1, 0), false)),
                    );
                    solver.add_expr(
                        (is_line.horizontal.at_offset(p, (0, -2), false)
                            & is_line.horizontal.at_offset(p, (0, -1), false))
                            | (is_line.horizontal.at_offset(p, (0, 0), false)
                                & is_line.horizontal.at_offset(p, (0, 1), false)),
                    );
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<MasyuClue>>;

fn combinator() -> impl Combinator<Vec<Vec<MasyuClue>>> {
    Grid::new(Map::new(
        MultiDigit::new(3, 3),
        |x: MasyuClue| {
            Some(match x {
                MasyuClue::None => 0,
                MasyuClue::White => 1,
                MasyuClue::Black => 2,
            })
        },
        |n: i32| match n {
            0 => Some(MasyuClue::None),
            1 => Some(MasyuClue::White),
            2 => Some(MasyuClue::Black),
            _ => None,
        },
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "masyu", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["masyu", "mashu"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Vec<Vec<MasyuClue>> {
        // https://puzsq.jp/main/puzzle_play.php?pid=9833
        let mut ret = vec![vec![MasyuClue::None; 10]; 10];
        ret[0][4] = MasyuClue::Black;
        ret[1][9] = MasyuClue::White;
        ret[2][1] = MasyuClue::Black;
        ret[2][8] = MasyuClue::Black;
        ret[3][0] = MasyuClue::White;
        ret[3][2] = MasyuClue::Black;
        ret[3][5] = MasyuClue::White;
        ret[3][7] = MasyuClue::White;
        ret[4][6] = MasyuClue::Black;
        ret[6][3] = MasyuClue::White;
        ret[6][5] = MasyuClue::White;
        ret[6][7] = MasyuClue::White;
        ret[7][3] = MasyuClue::Black;
        ret[8][1] = MasyuClue::Black;
        ret[8][7] = MasyuClue::White;
        ret[9][4] = MasyuClue::White;
        ret[9][7] = MasyuClue::White;
        ret
    }

    #[test]
    fn test_masyu_problem() {
        let problem = problem_for_tests();
        let ans = solve_masyu(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        for y in 0..problem.len() {
            for x in 0..problem[0].len() {
                if y + 1 < problem.len() {
                    assert!(ans.vertical[y][x].is_some());
                }
                if x + 1 < problem[0].len() {
                    assert!(ans.horizontal[y][x].is_some());
                }
            }
        }
        assert_eq!(ans.horizontal[4][0], Some(true));
        assert_eq!(ans.horizontal[0][4], Some(false));
    }

    #[test]
    fn test_masyu_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?masyu/10/10/0600003i06b1300600000a30600i090330";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
