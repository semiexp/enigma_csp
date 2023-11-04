use super::util;
use crate::graph;
use crate::serializer::{
    get_kudamono_url_info, kudamono_url_info_to_problem, problem_to_kudamono_url_grid, AlphaToNum,
    Choice, Combinator, Dict, KudamonoGrid, Optionalize,
};
use crate::solver::{any, Solver};

pub fn solve_parrot_loop(
    clues: &[Vec<Option<i32>>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let is_passed = &graph::single_cycle_grid_edges(&mut solver, &is_line);

    let num_horizontal = solver.int_var_1d(w - 2, 0, h as i32);
    let num_vertical = solver.int_var_1d(h - 2, 0, w as i32);
    for x in 0..(w - 2) {
        solver.add_expr(
            num_horizontal
                .at(x)
                .eq((is_line.horizontal.slice_fixed_x((.., x))
                    & is_line.horizontal.slice_fixed_x((.., x + 1)))
                .count_true()),
        );
    }
    for y in 0..(h - 2) {
        solver.add_expr(
            num_vertical
                .at(y)
                .eq((is_line.vertical.slice_fixed_y((y, ..))
                    & is_line.vertical.slice_fixed_y((y + 1, ..)))
                .count_true()),
        );
    }

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                if n < 0 {
                    solver.add_expr(!is_passed.at((y, x)));
                } else {
                    let mut cand = vec![];
                    if 0 < y && y < h - 1 {
                        cand.push(
                            is_line.vertical.at((y - 1, x))
                                & is_line.vertical.at((y, x))
                                & num_vertical.at(y - 1).eq(n + 1),
                        );
                    }
                    if 0 < x && x < w - 1 {
                        cand.push(
                            is_line.horizontal.at((y, x - 1))
                                & is_line.horizontal.at((y, x))
                                & num_horizontal.at(x - 1).eq(n + 1),
                        );
                    }
                    solver.add_expr(any(cand));
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    KudamonoGrid::new(
        Optionalize::new(Choice::new(vec![
            Box::new(Dict::new(-1, "x")),
            Box::new(Dict::new(0, "z")),
            Box::new(AlphaToNum::new('a', 'w', 1)),
        ])),
        None,
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_kudamono_url_grid(combinator(), "parrot-loop", problem.clone())
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
            vec![None, None, None, Some(2), Some(0), None],
            vec![Some(1), None, Some(-1), None, None, None],
            vec![None, None, None, Some(1), None, None],
            vec![None, None, None, None, None, None],
            vec![None, Some(-1), None, None, None, None],
            vec![None, None, None, None, None, None],
        ]
    }

    #[test]
    fn test_parrot_loop_problem() {
        let problem = problem_for_tests();
        let ans = solve_parrot_loop(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [1, 1, 1, 1, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_parrot_loop_serializer() {
        // v1
        {
            let problem = problem_for_tests();
            let url =
                "https://pedros.works/paper-puzzle-player?W=5&H=5&L=a4x3x9a5b2z6&G=parrot-loop";
            assert_eq!(deserialize_problem(url), Some(problem));
        }
        // v2
        {
            let problem = problem_for_tests();
            let url = "https://pedros.works/paper-puzzle-player?W=6x6&L=a4x3x9a5b2z6&G=parrot-loop";
            util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
        }
    }
}
