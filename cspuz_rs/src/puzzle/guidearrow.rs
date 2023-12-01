use super::util;
use crate::graph;
use crate::serializer::{
    from_base16, problem_to_url_with_context, to_base16, url_to_problem, Choice, Combinator,
    Context, ContextBasedGrid, Dict, HexInt, Size, Spaces, Tuple3,
};
use crate::solver::Solver;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GuidearrowClue {
    Up,
    Down,
    Left,
    Right,
    Unknown, // "?"
}

pub fn solve_guidearrow(
    ty: usize,
    tx: usize,
    clues: &[Vec<Option<GuidearrowClue>>],
) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);
    graph::active_vertices_connected_2d(&mut solver, !is_black);
    solver.add_expr(!is_black.conv2d_and((1, 2)));
    solver.add_expr(!is_black.conv2d_and((2, 1)));

    let rank = &solver.int_var_2d((h, w), 0, (h * w) as i32);
    solver.add_expr(rank.at((ty, tx)).eq(0));
    for y in 0..h {
        for x in 0..w {
            if (y, x) != (ty, tx) {
                solver.add_expr(
                    (!is_black.at((y, x))).imp(
                        (!is_black.four_neighbors((y, x)))
                            .imp(rank.four_neighbors((y, x)).ne(rank.at((y, x))))
                            .all()
                            & (!is_black.four_neighbors((y, x))
                                & rank.four_neighbors((y, x)).lt(rank.at((y, x))))
                            .count_true()
                            .eq(1),
                    ),
                );
            }
            if let Some(clue) = clues[y][x] {
                solver.add_expr(!is_black.at((y, x)));
                match clue {
                    GuidearrowClue::Up => {
                        if y == 0 {
                            return None;
                        }
                        solver.add_expr(!is_black.at((y - 1, x)));
                        solver.add_expr(rank.at((y - 1, x)).lt(rank.at((y, x))));
                    }
                    GuidearrowClue::Down => {
                        if y == h - 1 {
                            return None;
                        }
                        solver.add_expr(!is_black.at((y + 1, x)));
                        solver.add_expr(rank.at((y + 1, x)).lt(rank.at((y, x))));
                    }
                    GuidearrowClue::Left => {
                        if x == 0 {
                            return None;
                        }
                        solver.add_expr(!is_black.at((y, x - 1)));
                        solver.add_expr(rank.at((y, x - 1)).lt(rank.at((y, x))));
                    }
                    GuidearrowClue::Right => {
                        if x == w - 1 {
                            return None;
                        }
                        solver.add_expr(!is_black.at((y, x + 1)));
                        solver.add_expr(rank.at((y, x + 1)).lt(rank.at((y, x))));
                    }
                    _ => (),
                }
            }
        }
    }
    solver.irrefutable_facts().map(|f| f.get(is_black))
}

pub struct GuidearrowClueCombinator;

impl Combinator<Option<GuidearrowClue>> for GuidearrowClueCombinator {
    fn serialize(&self, _: &Context, input: &[Option<GuidearrowClue>]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let n = match input[0]? {
            GuidearrowClue::Up => 1,
            GuidearrowClue::Down => 2,
            GuidearrowClue::Left => 3,
            GuidearrowClue::Right => 4,
            _ => return None,
        };
        let mut n_spaces = 0;
        while n_spaces < 2 && 1 + n_spaces < input.len() && input[1 + n_spaces].is_none() {
            n_spaces += 1;
        }
        Some((1 + n_spaces, vec![to_base16(n + n_spaces as i32 * 5)]))
    }

    fn deserialize(
        &self,
        _: &Context,
        input: &[u8],
    ) -> Option<(usize, Vec<Option<GuidearrowClue>>)> {
        if input.len() == 0 {
            return None;
        }
        let c = from_base16(input[0])?;
        if c == 15 {
            return None;
        }
        let n = c % 5;
        let arrow = match n {
            0 => return None,
            1 => GuidearrowClue::Up,
            2 => GuidearrowClue::Down,
            3 => GuidearrowClue::Left,
            4 => GuidearrowClue::Right,
            _ => unreachable!(),
        };
        let spaces = c / 5;
        let mut ret = vec![Some(arrow)];
        for _ in 0..spaces {
            ret.push(None);
        }
        Some((1, ret))
    }
}

type Problem = (usize, usize, Vec<Vec<Option<GuidearrowClue>>>);

fn combinator() -> impl Combinator<(i32, i32, Vec<Vec<Option<GuidearrowClue>>>)> {
    Size::new(Tuple3::new(
        HexInt,
        HexInt,
        ContextBasedGrid::new(Choice::new(vec![
            Box::new(GuidearrowClueCombinator),
            Box::new(Dict::new(Some(GuidearrowClue::Unknown), ".")),
            Box::new(Spaces::new(None, 'g')),
        ])),
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let (h, w) = util::infer_shape(&problem.2);
    let problem = (
        problem.1 as i32 + 1,
        problem.0 as i32 + 1,
        problem.2.clone(),
    );
    problem_to_url_with_context(combinator(), "guidearrow", problem, &Context::sized(h, w))
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let res = url_to_problem(combinator(), &["guidearrow"], url)?;
    Some(((res.1 - 1) as usize, (res.0 - 1) as usize, res.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        (0, 2,
            vec![
                vec![None, None, None, None, None, Some(GuidearrowClue::Right), None],
                vec![None, Some(GuidearrowClue::Down), None, None, None, None, None],
                vec![None, None, None, None, None, None, None],
                vec![None, None, None, Some(GuidearrowClue::Left), None, None, None],
                vec![None, None, None, None, None, Some(GuidearrowClue::Unknown), None],
                vec![None, None, None, None, None, None, None],
            ]
        )
    }

    #[test]
    fn testguidearrow_problem() {
        let (ty, tx, clues) = problem_for_tests();
        let ans = solve_guidearrow(ty, tx, &clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
        ]);
        assert_eq!(ans, expected);
    }
    #[test]
    fn test_guidearrow_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?guidearrow/7/6/31kecsdl.n";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
