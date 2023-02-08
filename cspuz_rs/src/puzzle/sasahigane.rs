use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Map, Optionalize,
    Spaces,
};
use crate::solver::Solver;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SashiganeClue {
    Up,
    Down,
    Left,
    Right,
    Corner(i32),
}

pub fn solve_sashigane(
    clues: &[Vec<Option<SashiganeClue>>],
) -> Option<graph::BoolInnerGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    if h <= 1 || w <= 1 {
        return None;
    }

    let mut solver = Solver::new();
    let cell_kind = solver.int_var_2d((h, w), 0, 4);
    let edges = &graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&edges.horizontal);
    solver.add_answer_key_bool(&edges.vertical);

    solver.add_expr(cell_kind.slice_fixed_y((0, ..)).ne(0));
    solver.add_expr(
        cell_kind
            .slice((1.., ..))
            .eq(0)
            .imp(cell_kind.slice((..(h - 1), ..)).eq(0) | cell_kind.slice((..(h - 1), ..)).eq(4)),
    );
    solver.add_expr(cell_kind.slice_fixed_y((h - 1, ..)).ne(1));
    solver.add_expr(
        cell_kind
            .slice((..(h - 1), ..))
            .eq(1)
            .imp(cell_kind.slice((1.., ..)).eq(1) | cell_kind.slice((1.., ..)).eq(4)),
    );
    solver.add_expr(cell_kind.slice_fixed_x((.., 0)).ne(2));
    solver.add_expr(
        cell_kind
            .slice((.., 1..))
            .eq(2)
            .imp(cell_kind.slice((.., ..(w - 1))).eq(2) | cell_kind.slice((.., ..(w - 1))).eq(4)),
    );
    solver.add_expr(cell_kind.slice_fixed_x((.., w - 1)).ne(3));
    solver.add_expr(
        cell_kind
            .slice((.., ..(w - 1)))
            .eq(3)
            .imp(cell_kind.slice((.., 1..)).eq(3) | cell_kind.slice((.., 1..)).eq(4)),
    );

    solver.add_expr(
        &edges.horizontal
            ^ (cell_kind.slice((1.., ..)).eq(0) | cell_kind.slice((..(h - 1), ..)).eq(1)),
    );
    solver.add_expr(
        &edges.vertical
            ^ (cell_kind.slice((.., 1..)).eq(2) | cell_kind.slice((.., ..(w - 1))).eq(3)),
    );

    for y in 0..h {
        for x in 0..w {
            let ud;
            if y == 0 {
                ud = cell_kind.at((y + 1, x)).eq(0);
            } else if y == h - 1 {
                ud = cell_kind.at((y - 1, x)).eq(1);
            } else {
                solver.add_expr(!(cell_kind.at((y + 1, x)).eq(0) & cell_kind.at((y - 1, x)).eq(1)));
                ud = cell_kind.at((y + 1, x)).eq(0) | cell_kind.at((y - 1, x)).eq(1);
            }
            let lr;
            if x == 0 {
                lr = cell_kind.at((y, x + 1)).eq(2);
            } else if x == w - 1 {
                lr = cell_kind.at((y, x - 1)).eq(3);
            } else {
                solver.add_expr(!(cell_kind.at((y, x + 1)).eq(2) & cell_kind.at((y, x - 1)).eq(3)));
                lr = cell_kind.at((y, x + 1)).eq(2) | cell_kind.at((y, x - 1)).eq(3);
            }

            solver.add_expr(cell_kind.at((y, x)).eq(4).imp(ud & lr));
        }
    }

    for y in 0..h {
        for x in 0..w {
            if let Some(c) = clues[y][x] {
                match c {
                    SashiganeClue::Up => {
                        solver.add_expr(cell_kind.at((y, x)).eq(0));
                        if y < h - 1 {
                            solver.add_expr(cell_kind.at((y + 1, x)).ne(0));
                        }
                    }
                    SashiganeClue::Down => {
                        solver.add_expr(cell_kind.at((y, x)).eq(1));
                        if y > 0 {
                            solver.add_expr(cell_kind.at((y - 1, x)).ne(1));
                        }
                    }
                    SashiganeClue::Left => {
                        solver.add_expr(cell_kind.at((y, x)).eq(2));
                        if x < w - 1 {
                            solver.add_expr(cell_kind.at((y, x + 1)).ne(2));
                        }
                    }
                    SashiganeClue::Right => {
                        solver.add_expr(cell_kind.at((y, x)).eq(3));
                        if x > 0 {
                            solver.add_expr(cell_kind.at((y, x - 1)).ne(3));
                        }
                    }
                    SashiganeClue::Corner(n) => {
                        solver.add_expr(cell_kind.at((y, x)).eq(4));
                        if n > 0 {
                            let up = cell_kind
                                .slice_fixed_x((..y, x))
                                .reverse()
                                .eq(1)
                                .consecutive_prefix_true();
                            let down = cell_kind
                                .slice_fixed_x(((y + 1).., x))
                                .eq(0)
                                .consecutive_prefix_true();
                            let left = cell_kind
                                .slice_fixed_y((y, ..x))
                                .reverse()
                                .eq(3)
                                .consecutive_prefix_true();
                            let right = cell_kind
                                .slice_fixed_y((y, (x + 1)..))
                                .eq(2)
                                .consecutive_prefix_true();
                            solver.add_expr((up + down + left + right + 1).eq(n));
                        }
                    }
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(edges))
}

type Problem = Vec<Vec<Option<SashiganeClue>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Optionalize::new(Choice::new(vec![
            Box::new(Dict::new(SashiganeClue::Up, "g")),
            Box::new(Dict::new(SashiganeClue::Down, "h")),
            Box::new(Dict::new(SashiganeClue::Left, "i")),
            Box::new(Dict::new(SashiganeClue::Right, "j")),
            Box::new(Dict::new(SashiganeClue::Corner(-1), ".")),
            Box::new(Map::new(
                HexInt,
                |x| match x {
                    SashiganeClue::Corner(x) => Some(x),
                    _ => None,
                },
                |x| Some(SashiganeClue::Corner(x)),
            )),
        ]))),
        Box::new(Spaces::new(None, 'k')),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "sashigane", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["sashigane"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        vec![
            vec![None, None, Some(SashiganeClue::Corner(4)), None, None, None],
            vec![Some(SashiganeClue::Right), None, None, None, Some(SashiganeClue::Corner(-1)), None],
            vec![None, None, None, Some(SashiganeClue::Up), None, None],
            vec![None, None, None, None, Some(SashiganeClue::Up), None],
            vec![Some(SashiganeClue::Right), None, None, None, Some(SashiganeClue::Left), None],
            vec![None, None, None, None, None, None],
        ]
    }

    #[test]
    fn test_sashigane_problem() {
        let problem = problem_for_tests();

        let ans = solve_sashigane(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        #[rustfmt::skip]
        let expected = graph::BoolInnerGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 0, 1],
                [0, 1, 0, 1, 0, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_sashigane_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?sashigane/6/6/l4mjm.ngpgkjmiq";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
