use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, ContextBasedGrid,
    Dict, HexInt, Map, MultiDigit, Optionalize, Size, Spaces, Tuple2,
};
use crate::solver::{count_true, Solver};

pub fn solve_herugolf(
    pond: &[Vec<bool>],
    clues: &[Vec<Option<i32>>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(&clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let mut clue_max = 0;
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                clue_max = clue_max.max(n);
            }
        }
    }
    if clue_max >= h as i32 || clue_max >= w as i32 {
        return None;
    }
    let level = &solver.int_var_2d((h, w), 0, clue_max);
    let rank = &solver.int_var_2d((h, w), 0, clue_max);
    let dir = &solver.int_var_2d((h, w), 0, 4); // 1: up, 2: down, 3: left, 4: right

    solver.add_expr(
        is_line
            .horizontal
            .iff(dir.slice((.., ..(w - 1))).eq(4) | dir.slice((.., 1..)).eq(3)),
    );
    solver.add_expr(
        is_line
            .vertical
            .iff(dir.slice((..(h - 1), ..)).eq(2) | dir.slice((1.., ..)).eq(1)),
    );

    for y in 0..h {
        for x in 0..w {
            let has_in_edge = &solver.bool_var();
            let mut in_edge_cand = vec![];
            if y > 0 {
                in_edge_cand.push(dir.at((y - 1, x)).eq(2));
            }
            if y < h - 1 {
                in_edge_cand.push(dir.at((y + 1, x)).eq(1));
            }
            if x > 0 {
                in_edge_cand.push(dir.at((y, x - 1)).eq(4));
            }
            if x < w - 1 {
                in_edge_cand.push(dir.at((y, x + 1)).eq(3));
            }
            solver.add_expr(has_in_edge.ite(1, 0).eq(count_true(in_edge_cand)));

            let d = &dir.at((y, x));
            let r = &rank.at((y, x));
            let l = &level.at((y, x));

            // rank should decrease
            if y > 0 {
                solver.add_expr(
                    d.eq(1).imp(
                        (r.gt(0)
                            .imp(rank.at((y - 1, x)).eq(r - 1) & level.at((y - 1, x)).eq(l)))
                            & (r.eq(0).imp(
                                rank.at((y - 1, x)).eq(l - 1) & level.at((y - 1, x)).eq(l - 1),
                            )),
                    ),
                );
            } else {
                solver.add_expr(d.ne(1));
            }
            if y < h - 1 {
                solver.add_expr(
                    d.eq(2).imp(
                        (r.gt(0)
                            .imp(rank.at((y + 1, x)).eq(r - 1) & level.at((y + 1, x)).eq(l)))
                            & (r.eq(0).imp(
                                rank.at((y + 1, x)).eq(l - 1) & level.at((y + 1, x)).eq(l - 1),
                            )),
                    ),
                );
            } else {
                solver.add_expr(d.ne(2));
            }
            if x > 0 {
                solver.add_expr(
                    d.eq(3).imp(
                        (r.gt(0)
                            .imp(rank.at((y, x - 1)).eq(r - 1) & level.at((y, x - 1)).eq(l)))
                            & (r.eq(0).imp(
                                rank.at((y, x - 1)).eq(l - 1) & level.at((y, x - 1)).eq(l - 1),
                            )),
                    ),
                );
            } else {
                solver.add_expr(d.ne(3));
            }
            if x < w - 1 {
                solver.add_expr(
                    d.eq(4).imp(
                        (r.gt(0)
                            .imp(rank.at((y, x + 1)).eq(r - 1) & level.at((y, x + 1)).eq(l)))
                            & (r.eq(0).imp(
                                rank.at((y, x + 1)).eq(l - 1) & level.at((y, x + 1)).eq(l - 1),
                            )),
                    ),
                );
            } else {
                solver.add_expr(d.ne(4));
            }

            // go straight except (rank == 0)
            if y > 0 {
                solver.add_expr(r.gt(0).imp(dir.at((y - 1, x)).eq(2).imp(d.eq(2))));
            }
            if y < h - 1 {
                solver.add_expr(r.gt(0).imp(dir.at((y + 1, x)).eq(1).imp(d.eq(1))));
            }
            if x > 0 {
                solver.add_expr(r.gt(0).imp(dir.at((y, x - 1)).eq(4).imp(d.eq(4))));
            }
            if x < w - 1 {
                solver.add_expr(r.gt(0).imp(dir.at((y, x + 1)).eq(3).imp(d.eq(3))));
            }

            if let Some(n) = clues[y][x] {
                if n < 0 {
                    // hole
                    solver.add_expr(has_in_edge);
                    solver.add_expr(d.eq(0));
                    solver.add_expr(r.eq(0));
                } else {
                    solver.add_expr(!has_in_edge);
                    solver.add_expr(d.ne(0));
                    solver.add_expr(l.eq(n));
                    solver.add_expr(r.eq(0));
                }
            } else {
                // (has in edge) iff (has out edge)
                solver.add_expr(has_in_edge.iff(d.ne(0)));
                solver.add_expr((!has_in_edge).imp(r.eq(0) & l.eq(0)));
            }

            if pond[y][x] {
                solver.add_expr(has_in_edge.imp(r.ne(0)));
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = (Vec<Vec<bool>>, Vec<Vec<Option<i32>>>);

fn combinator() -> impl Combinator<Problem> {
    Size::new(Tuple2::new(
        ContextBasedGrid::new(Map::new(
            MultiDigit::new(2, 5),
            |x| Some(if x { 1 } else { 0 }),
            |x| Some(x == 1),
        )),
        ContextBasedGrid::new(Choice::new(vec![
            Box::new(Optionalize::new(HexInt)),
            Box::new(Spaces::new(None, 'i')),
            Box::new(Dict::new(Some(-1), "h")),
        ])),
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let (h, w) = util::infer_shape(&problem.0);
    problem_to_url_with_context(
        combinator(),
        "herugolf",
        problem.clone(),
        &Context::sized(h, w),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["herugolf"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        (
            vec![
                vec![false, false, false, false, false, false],
                vec![false, true, false, false, false, false],
                vec![false, false, false, false, false, false],
                vec![false, false, false, true, false, false],
                vec![false, false, false, true, false, false],
                vec![false, false, false, false, false, false],
            ],
            vec![
                vec![None, None, Some(-1), None, None, Some(-1)],
                vec![None, None, None, None, None, None],
                vec![None, Some(2), None, Some(2), None, None],
                vec![None, None, None, None, None, None],
                vec![None, None, Some(-1), None, None, None],
                vec![Some(4), None, None, None, None, None],
            ],
        )
    }

    #[rustfmt::skip]
    #[test]
    fn test_herugolf_problem() {
        let (pond, clues) = problem_for_tests();
        let ans = solve_herugolf(&pond, &clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        
        let expected = graph::GridEdges {
            horizontal: vec![
                vec![Some(false), Some(false), Some(true), Some(false), Some(true)],
                vec![Some(false), Some(false), Some(false), Some(false), Some(false)],
                vec![Some(false), Some(false), Some(false), Some(false), Some(false)],
                vec![Some(false), Some(false), Some(false), Some(false), Some(false)],
                vec![Some(false), Some(true), Some(false), Some(false), Some(false)],
                vec![Some(true), Some(true), Some(true), Some(true), Some(false)],
            ],
            vertical: vec![
                vec![Some(false), Some(false), Some(false), Some(true), Some(true), Some(false)],
                vec![Some(false), Some(false), Some(false), Some(true), Some(true), Some(false)],
                vec![Some(false), Some(true), Some(false), Some(false), Some(true), Some(false)],
                vec![Some(false), Some(true), Some(false), Some(false), Some(true), Some(false)],
                vec![Some(false), Some(false), Some(false), Some(false), Some(true), Some(false)],
            ],
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_herugolf_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?herugolf/6/6/04008400jhjho2i2rhk4m";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
