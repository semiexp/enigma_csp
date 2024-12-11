use super::util;
use crate::graph;
use crate::serializer::{
    get_kudamono_url_info, kudamono_url_info_to_problem, problem_to_kudamono_url_grid, Choice,
    Combinator, DecInt, Dict, KudamonoGrid, Optionalize, PrefixAndSuffix,
};
use crate::solver::{any, Solver, FALSE};

pub fn solve_multiplication_link(
    clues: &[Vec<Option<i32>>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let is_passed = &graph::single_cycle_grid_edges(&mut solver, &is_line);

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                if n == -2 {
                    solver.add_expr(!is_passed.at((y, x)));
                    continue;
                }

                solver.add_expr(
                    is_line.horizontal.at_offset((y, x), (0, -1), FALSE)
                        ^ is_line.horizontal.at_offset((y, x), (0, 0), FALSE),
                );
                solver.add_expr(
                    is_line.vertical.at_offset((y, x), (-1, 0), FALSE)
                        ^ is_line.vertical.at_offset((y, x), (0, 0), FALSE),
                );

                let len_x = &solver.int_var(1, w as i32 - 1);
                if x > 0 {
                    solver.add_expr(
                        is_line.horizontal.at((y, x - 1)).imp(
                            len_x.eq(is_line
                                .horizontal
                                .slice_fixed_y((y, ..x))
                                .reverse()
                                .consecutive_prefix_true()),
                        ),
                    );
                }
                if x < w - 1 {
                    solver.add_expr(
                        is_line.horizontal.at((y, x)).imp(
                            len_x.eq(is_line
                                .horizontal
                                .slice_fixed_y((y, x..))
                                .consecutive_prefix_true()),
                        ),
                    );
                }

                let len_y = &solver.int_var(1, h as i32 - 1);
                if y > 0 {
                    solver.add_expr(
                        is_line.vertical.at((y - 1, x)).imp(
                            len_y.eq(is_line
                                .vertical
                                .slice_fixed_x((..y, x))
                                .reverse()
                                .consecutive_prefix_true()),
                        ),
                    );
                }
                if y < h - 1 {
                    solver.add_expr(
                        is_line.vertical.at((y, x)).imp(
                            len_y.eq(is_line
                                .vertical
                                .slice_fixed_x((y.., x))
                                .consecutive_prefix_true()),
                        ),
                    );
                }

                if n == -1 {
                    continue;
                }

                let mut cand = vec![];
                for nx in 1..=(w as i32 - 1) {
                    if n % nx != 0 {
                        continue;
                    }
                    let ny = n / nx;
                    if !(1 <= ny && ny <= h as i32 - 1) {
                        continue;
                    }

                    cand.push(len_x.eq(nx) & len_y.eq(ny));
                }

                solver.add_expr(any(cand));
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    KudamonoGrid::new(
        Optionalize::new(Choice::new(vec![
            Box::new(Dict::new(-2, "x")),
            Box::new(Dict::new(-1, "y")),
            Box::new(PrefixAndSuffix::new("(", DecInt, ")")),
        ])),
        None,
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_kudamono_url_grid(combinator(), "multiplication-link", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let info = get_kudamono_url_info(url)?;
    kudamono_url_info_to_problem(combinator(), info)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![Some(20), None, None, None, None, None],
            vec![None, Some(6), None, None, None, Some(-1)],
            vec![None, None, None, None, None, None],
            vec![None, None, None, None, Some(4), None],
            vec![None, None, None, None, None, Some(-2)],
        ]
    }

    #[test]
    fn test_multiplication_link_problem() {
        let problem = problem_for_tests();
        let ans = solve_multiplication_link(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 1, 1, 1, 1],
                [0, 1, 1, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [1, 0, 0, 0, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 1, 0],
                [1, 1, 1, 0, 1, 0],
                [1, 1, 0, 0, 0, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_multiplication_link_serializer() {
        let problem = problem_for_tests();
        let url = "https://pedros.works/paper-puzzle-player?W=6x5&L=(20)4(6)4(4)13x4y3&G=multiplication-link";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
