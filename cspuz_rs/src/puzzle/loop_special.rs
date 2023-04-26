use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Map, Spaces,
};
use crate::solver::Solver;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopSpecialClue {
    Num(i32),
    Empty,
    Cross,
    Vertical,
    Horizontal,
    UpRight,
    UpLeft,
    DownLeft,
    DownRight,
}

pub fn solve_loop_special(
    clues: &[Vec<LoopSpecialClue>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let mut max_num = 0;
    for y in 0..h {
        for x in 0..w {
            if let LoopSpecialClue::Num(n) = clues[y][x] {
                max_num = max_num.max(n);
            }
        }
    }

    let horizontal = &solver.int_var_2d((h, w - 1), 0, max_num);
    let vertical = &solver.int_var_2d((h - 1, w), 0, max_num);
    solver.add_expr(is_line.horizontal.iff(horizontal.ne(0)));
    solver.add_expr(is_line.vertical.iff(vertical.ne(0)));

    for y in 0..h {
        for x in 0..w {
            solver.add_expr(is_line.vertex_neighbors((y, x)).any());
        }
    }
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let is_cross = &solver.bool_var();
            solver.add_expr(is_cross.iff(is_line.vertex_neighbors((y, x)).count_true().eq(4)));
            solver.add_expr(is_cross.imp(horizontal.at((y, x - 1)).eq(horizontal.at((y, x)))));
            solver.add_expr(is_cross.imp(vertical.at((y - 1, x)).eq(vertical.at((y, x)))));
        }
    }

    for i in 1..=max_num {
        let loop_i = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
        solver.add_expr(loop_i.horizontal.iff(horizontal.eq(i)));
        solver.add_expr(loop_i.vertical.iff(vertical.eq(i)));
        graph::crossable_single_cycle_grid_edges(&mut solver, loop_i);

        for y in 0..h {
            for x in 0..w {
                if let LoopSpecialClue::Num(n) = clues[y][x] {
                    solver.add_expr(loop_i.vertex_neighbors((y, x)).count_true().eq(if n == i {
                        2
                    } else {
                        0
                    }));
                }
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            let (up, down, left, right) = match clues[y][x] {
                LoopSpecialClue::Num(_) => continue,
                LoopSpecialClue::Empty => continue,
                LoopSpecialClue::Cross => (true, true, true, true),
                LoopSpecialClue::Vertical => (true, true, false, false),
                LoopSpecialClue::Horizontal => (false, false, true, true),
                LoopSpecialClue::UpRight => (true, false, false, true),
                LoopSpecialClue::UpLeft => (true, false, true, false),
                LoopSpecialClue::DownLeft => (false, true, true, false),
                LoopSpecialClue::DownRight => (false, true, false, true),
            };
            if up {
                if y == 0 {
                    return None;
                }
                solver.add_expr(is_line.vertical.at((y - 1, x)));
            } else {
                if y > 0 {
                    solver.add_expr(!is_line.vertical.at((y - 1, x)));
                }
            }
            if down {
                if y == h - 1 {
                    return None;
                }
                solver.add_expr(is_line.vertical.at((y, x)));
            } else {
                if y < h - 1 {
                    solver.add_expr(!is_line.vertical.at((y, x)));
                }
            }
            if left {
                if x == 0 {
                    return None;
                }
                solver.add_expr(is_line.horizontal.at((y, x - 1)));
            } else {
                if x > 0 {
                    solver.add_expr(!is_line.horizontal.at((y, x - 1)));
                }
            }
            if right {
                if x == w - 1 {
                    return None;
                }
                solver.add_expr(is_line.horizontal.at((y, x)));
            } else {
                if x < w - 1 {
                    solver.add_expr(!is_line.horizontal.at((y, x)));
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<LoopSpecialClue>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Spaces::new(LoopSpecialClue::Empty, 'n')),
        Box::new(Map::new(
            HexInt,
            |x| match x {
                LoopSpecialClue::Num(n) => Some(n),
                _ => None,
            },
            |n: i32| {
                if n >= 1 {
                    Some(LoopSpecialClue::Num(n))
                } else {
                    None
                }
            },
        )),
        Box::new(Dict::new(LoopSpecialClue::Cross, "g")),
        Box::new(Dict::new(LoopSpecialClue::Vertical, "h")),
        Box::new(Dict::new(LoopSpecialClue::Horizontal, "i")),
        Box::new(Dict::new(LoopSpecialClue::UpRight, "j")),
        Box::new(Dict::new(LoopSpecialClue::UpLeft, "k")),
        Box::new(Dict::new(LoopSpecialClue::DownLeft, "l")),
        Box::new(Dict::new(LoopSpecialClue::DownRight, "m")),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "loopsp", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["loopsp"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        vec![
            vec![LoopSpecialClue::Num(1), LoopSpecialClue::Empty, LoopSpecialClue::Num(2), LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty],
            vec![LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::DownLeft, LoopSpecialClue::Empty],
            vec![LoopSpecialClue::Num(2), LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Vertical],
            vec![LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::DownRight, LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty],
            vec![LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty],
            vec![LoopSpecialClue::Num(1), LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::UpLeft, LoopSpecialClue::Empty, LoopSpecialClue::Empty],
            vec![LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty, LoopSpecialClue::Empty],
        ]
    }

    #[test]
    fn test_loop_special_problem() {
        let problem = problem_for_tests();
        let ans = solve_loop_special(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 1, 1, 0, 0, 1],
                [0, 1, 1, 0, 1, 1],
                [1, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 1],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_loop_special_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?loopsp/6/7/1n2tln2qhomv1oku";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
