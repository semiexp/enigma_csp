use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Map, Spaces,
};
use crate::solver::{count_true, Solver};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PencilsClue {
    None,
    Num(i32),
    Up,
    Down,
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PencilsAnswer {
    Empty,
    Up,
    Down,
    Left,
    Right,
}

const ID_TO_ANSWER_COMPONENT: [PencilsAnswer; 5] = [
    PencilsAnswer::Empty,
    PencilsAnswer::Up,
    PencilsAnswer::Down,
    PencilsAnswer::Left,
    PencilsAnswer::Right,
];

pub fn solve_pencils(
    clues: &[Vec<PencilsClue>],
) -> Option<(
    Vec<Vec<Option<PencilsAnswer>>>,
    graph::BoolGridEdgesIrrefutableFacts,
    graph::BoolInnerGridEdgesIrrefutableFacts,
)> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let max_level = h.max(w) as i32;
    let cell_kind = &solver.int_var_2d((h, w), 0, 5);
    let cell_level = &solver.int_var_2d((h, w), 0, max_level);
    let pencil_size = &solver.int_var_2d((h, w), 0, max_level);
    let cell_answer = &solver.int_var_2d((h, w), 0, 4);
    let is_border = &graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_int(cell_answer);
    solver.add_answer_key_bool(&is_border.horizontal);
    solver.add_answer_key_bool(&is_border.vertical);
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    solver.add_expr(is_line.horizontal.imp(
        (cell_kind.slice((.., ..(w - 1))).eq(4) | cell_kind.slice((.., ..(w - 1))).eq(5))
            & (cell_kind.slice((.., 1..)).eq(4) | cell_kind.slice((.., 1..)).eq(5)),
    ));
    solver.add_expr(is_line.vertical.imp(
        (cell_kind.slice((..(h - 1), ..)).eq(4) | cell_kind.slice((..(h - 1), ..)).eq(5))
            & (cell_kind.slice((1.., ..)).eq(4) | cell_kind.slice((1.., ..)).eq(5)),
    ));
    solver.add_expr(pencil_size.gt(0).iff(cell_kind.ne(5)));
    solver.add_expr(cell_kind.eq(4).imp(pencil_size.eq(cell_level)));

    for y in 0..h {
        for x in 0..w {
            if y == 0 {
                solver.add_expr(cell_kind.at((y, x)).ne(0));
            } else {
                solver.add_expr(cell_kind.at((y, x)).eq(0).imp(
                    (cell_kind.at((y - 1, x)).eq(0) | cell_kind.at((y - 1, x)).eq(4))
                        & cell_level.at((y - 1, x)).eq(cell_level.at((y, x)) + 1)
                        & pencil_size.at((y - 1, x)).eq(pencil_size.at((y, x))),
                ));
                if y == h - 1 {
                    solver.add_expr(cell_kind.at((y, x)).eq(0).imp(cell_level.at((y, x)).eq(0)));
                } else {
                    solver.add_expr(cell_kind.at((y, x)).eq(0).imp(
                        cell_level.at((y, x)).eq(0)
                            | (cell_kind.at((y + 1, x)).eq(0)
                                & cell_level.at((y + 1, x)).eq(cell_level.at((y, x)) - 1)),
                    ));
                }
            }
            if y == h - 1 {
                solver.add_expr(cell_kind.at((y, x)).ne(1));
            } else {
                solver.add_expr(cell_kind.at((y, x)).eq(1).imp(
                    (cell_kind.at((y + 1, x)).eq(1) | cell_kind.at((y + 1, x)).eq(4))
                        & cell_level.at((y + 1, x)).eq(cell_level.at((y, x)) + 1)
                        & pencil_size.at((y + 1, x)).eq(pencil_size.at((y, x))),
                ));
                if y == 0 {
                    solver.add_expr(cell_kind.at((y, x)).eq(1).imp(cell_level.at((y, x)).eq(0)));
                } else {
                    solver.add_expr(cell_kind.at((y, x)).eq(1).imp(
                        cell_level.at((y, x)).eq(0)
                            | (cell_kind.at((y - 1, x)).eq(1)
                                & cell_level.at((y - 1, x)).eq(cell_level.at((y, x)) - 1)),
                    ));
                }
            }
            if x == 0 {
                solver.add_expr(cell_kind.at((y, x)).ne(2));
            } else {
                solver.add_expr(cell_kind.at((y, x)).eq(2).imp(
                    (cell_kind.at((y, x - 1)).eq(2) | cell_kind.at((y, x - 1)).eq(4))
                        & cell_level.at((y, x - 1)).eq(cell_level.at((y, x)) + 1)
                        & pencil_size.at((y, x - 1)).eq(pencil_size.at((y, x))),
                ));
                if x == w - 1 {
                    solver.add_expr(cell_kind.at((y, x)).eq(2).imp(cell_level.at((y, x)).eq(0)));
                } else {
                    solver.add_expr(cell_kind.at((y, x)).eq(2).imp(
                        cell_level.at((y, x)).eq(0)
                            | (cell_kind.at((y, x + 1)).eq(2)
                                & cell_level.at((y, x + 1)).eq(cell_level.at((y, x)) - 1)),
                    ));
                }
            }
            if x == w - 1 {
                solver.add_expr(cell_kind.at((y, x)).ne(3));
            } else {
                solver.add_expr(cell_kind.at((y, x)).eq(3).imp(
                    (cell_kind.at((y, x + 1)).eq(3) | cell_kind.at((y, x + 1)).eq(4))
                        & cell_level.at((y, x + 1)).eq(cell_level.at((y, x)) + 1)
                        & pencil_size.at((y, x + 1)).eq(pencil_size.at((y, x))),
                ));
                if x == 0 {
                    solver.add_expr(cell_kind.at((y, x)).eq(3).imp(cell_level.at((y, x)).eq(0)));
                } else {
                    solver.add_expr(cell_kind.at((y, x)).eq(3).imp(
                        cell_level.at((y, x)).eq(0)
                            | (cell_kind.at((y, x - 1)).eq(3)
                                & cell_level.at((y, x - 1)).eq(cell_level.at((y, x)) - 1)),
                    ));
                }
            }

            let deg = is_line.vertex_neighbors((y, x)).count_true();
            solver.add_expr(cell_kind.at((y, x)).le(3).imp(deg.eq(0)));
            solver.add_expr(cell_kind.at((y, x)).eq(4).imp(deg.eq(1)));
            solver.add_expr(cell_kind.at((y, x)).eq(5).imp(
                (cell_level.at((y, x)).eq(0).imp(deg.eq(1)))
                    & (cell_level.at((y, x)).gt(0).imp(deg.eq(2))),
            ));

            for d in [-1, 1] {
                let mut num = vec![];
                if y > 0 {
                    num.push(
                        is_line.vertical.at((y - 1, x))
                            & cell_level.at((y, x)).eq(cell_level.at((y - 1, x)) - d),
                    );
                }
                if y < h - 1 {
                    num.push(
                        is_line.vertical.at((y, x))
                            & cell_level.at((y, x)).eq(cell_level.at((y + 1, x)) - d),
                    );
                }
                if x > 0 {
                    num.push(
                        is_line.horizontal.at((y, x - 1))
                            & cell_level.at((y, x)).eq(cell_level.at((y, x - 1)) - d),
                    );
                }
                if x < w - 1 {
                    num.push(
                        is_line.horizontal.at((y, x))
                            & cell_level.at((y, x)).eq(cell_level.at((y, x + 1)) - d),
                    );
                }

                if d == -1 {
                    solver.add_expr(
                        (cell_kind.at((y, x)).eq(4)
                            | (cell_kind.at((y, x)).eq(5) & cell_level.at((y, x)).gt(0)))
                        .imp(count_true(num).eq(1)),
                    );
                } else {
                    solver.add_expr(cell_kind.at((y, x)).eq(5).imp(count_true(num).eq(1)));
                }
            }

            let mut in_axis = vec![];
            if y > 0 {
                in_axis.push(cell_kind.at((y - 1, x)).eq(1));
            }
            if y < h - 1 {
                in_axis.push(cell_kind.at((y + 1, x)).eq(0));
            }
            if x > 0 {
                in_axis.push(cell_kind.at((y, x - 1)).eq(3));
            }
            if x < w - 1 {
                in_axis.push(cell_kind.at((y, x + 1)).eq(2));
            }
            solver.add_expr(cell_kind.at((y, x)).eq(4).imp(count_true(in_axis).eq(1)));
        }
    }

    solver.add_expr(cell_answer.gt(0).iff(cell_kind.eq(4)));
    for y in 0..h {
        for x in 0..w {
            if y > 0 {
                solver.add_expr(
                    cell_answer
                        .at((y, x))
                        .eq(1)
                        .imp(cell_kind.at((y - 1, x)).eq(1)),
                );
            } else {
                solver.add_expr(cell_answer.at((y, x)).ne(1));
            }
            if y < h - 1 {
                solver.add_expr(
                    cell_answer
                        .at((y, x))
                        .eq(2)
                        .imp(cell_kind.at((y + 1, x)).eq(0)),
                );
            } else {
                solver.add_expr(cell_answer.at((y, x)).ne(2));
            }
            if x > 0 {
                solver.add_expr(
                    cell_answer
                        .at((y, x))
                        .eq(3)
                        .imp(cell_kind.at((y, x - 1)).eq(3)),
                );
            } else {
                solver.add_expr(cell_answer.at((y, x)).ne(3));
            }
            if x < w - 1 {
                solver.add_expr(
                    cell_answer
                        .at((y, x))
                        .eq(4)
                        .imp(cell_kind.at((y, x + 1)).eq(2)),
                );
            } else {
                solver.add_expr(cell_answer.at((y, x)).ne(4));
            }
        }
    }
    for y in 0..h {
        for x in 0..(w - 1) {
            solver.add_expr(is_border.vertical.at((y, x)).iff(
                cell_kind.at((y, x)).eq(0)
                    | cell_kind.at((y, x)).eq(1)
                    | cell_kind.at((y, x + 1)).eq(0)
                    | cell_kind.at((y, x + 1)).eq(1)
                    | (cell_kind.at((y, x)).eq(2) & cell_kind.at((y, x + 1)).ne(2))
                    | (cell_kind.at((y, x + 1)).eq(3) & cell_kind.at((y, x)).ne(3)),
            ));
        }
    }
    for y in 0..(h - 1) {
        for x in 0..w {
            solver.add_expr(is_border.horizontal.at((y, x)).iff(
                cell_kind.at((y, x)).eq(2)
                    | cell_kind.at((y, x)).eq(3)
                    | cell_kind.at((y + 1, x)).eq(2)
                    | cell_kind.at((y + 1, x)).eq(3)
                    | (cell_kind.at((y, x)).eq(0) & cell_kind.at((y + 1, x)).ne(0))
                    | (cell_kind.at((y + 1, x)).eq(1) & cell_kind.at((y, x)).ne(1)),
            ));
        }
    }

    for y in 0..h {
        for x in 0..w {
            match &clues[y][x] {
                &PencilsClue::None => (),
                &PencilsClue::Num(n) => {
                    solver.add_expr(cell_kind.at((y, x)).le(3));
                    solver.add_expr(pencil_size.at((y, x)).eq(n));
                }
                &PencilsClue::Up => solver.add_expr(cell_answer.at((y, x)).eq(1)),
                &PencilsClue::Down => solver.add_expr(cell_answer.at((y, x)).eq(2)),
                &PencilsClue::Left => solver.add_expr(cell_answer.at((y, x)).eq(3)),
                &PencilsClue::Right => solver.add_expr(cell_answer.at((y, x)).eq(4)),
            }
        }
    }

    solver.irrefutable_facts().map(|f| {
        (
            f.get(cell_answer)
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|x| x.map(|x| ID_TO_ANSWER_COMPONENT[x as usize]))
                        .collect()
                })
                .collect(),
            f.get(is_line),
            f.get(is_border),
        )
    })
}

type Problem = Vec<Vec<PencilsClue>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Map::new(
            HexInt,
            |n| match n {
                PencilsClue::Num(n) => Some(n),
                _ => None,
            },
            |n| Some(PencilsClue::Num(n)),
        )),
        Box::new(Spaces::new(PencilsClue::None, 'k')),
        Box::new(Dict::new(PencilsClue::Up, "h")),
        Box::new(Dict::new(PencilsClue::Down, "g")),
        Box::new(Dict::new(PencilsClue::Left, "j")),
        Box::new(Dict::new(PencilsClue::Right, "i")),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "pencils", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["pencils"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        vec![
            vec![PencilsClue::None, PencilsClue::Down, PencilsClue::Right, PencilsClue::None, PencilsClue::None, PencilsClue::None],
            vec![PencilsClue::None, PencilsClue::Num(3), PencilsClue::None, PencilsClue::None, PencilsClue::None, PencilsClue::None],
            vec![PencilsClue::None, PencilsClue::None, PencilsClue::Num(2), PencilsClue::None, PencilsClue::None, PencilsClue::None],
            vec![PencilsClue::None, PencilsClue::None, PencilsClue::Right, PencilsClue::None, PencilsClue::None, PencilsClue::Num(2)],
            vec![PencilsClue::None, PencilsClue::None, PencilsClue::None, PencilsClue::None, PencilsClue::Left, PencilsClue::None],
            vec![PencilsClue::Right, PencilsClue::None, PencilsClue::None, PencilsClue::None, PencilsClue::Left, PencilsClue::None],
        ]
    }

    #[test]
    fn test_pencils_problem() {
        let problem = problem_for_tests();
        let ans = solve_pencils(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        #[rustfmt::skip]
        let expected = (
            crate::puzzle::util::tests::to_option_2d([
                [PencilsAnswer::Empty, PencilsAnswer::Down, PencilsAnswer::Right, PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Empty],
                [PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Down],
                [PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Left, PencilsAnswer::Empty],
                [PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Right, PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Empty],
                [PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Left, PencilsAnswer::Empty],
                [PencilsAnswer::Right, PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Empty, PencilsAnswer::Left, PencilsAnswer::Empty],
            ]),
            graph::BoolGridEdgesIrrefutableFacts {
                horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                    [1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                ]),
                vertical: crate::puzzle::util::tests::to_option_bool_2d([
                    [1, 0, 1, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                ]),
            },
            graph::BoolInnerGridEdgesIrrefutableFacts {
                horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 1, 1],
                    [0, 1, 1, 1, 0, 0],
                ]),
                vertical: crate::puzzle::util::tests::to_option_bool_2d([
                    [0, 0, 0, 1, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                ]),
            }
        );
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_pencils_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?pencils/6/6/kgin3p2oil2njkimjk";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
