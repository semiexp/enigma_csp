use super::util;
use crate::serializer::{
    get_kudamono_url_info, kudamono_url_info_to_problem, problem_to_kudamono_url_grid, AlphaToNum,
    Choice, Combinator, DecInt, KudamonoGrid, Optionalize, PrefixAndSuffix,
};
use crate::solver::Solver;

pub fn solve_soulmates(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<i32>>>> {
    let (h, w) = util::infer_shape(clues);

    let max_num = (h * w - 1) as i32;
    let mut solver = Solver::new();
    let num = &solver.int_var_2d((h, w), 0, max_num);
    solver.add_answer_key_int(num);

    let has_num = &solver.bool_var_2d((h, w));
    solver.add_expr(has_num ^ num.eq(0));

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(num.at((y, x)).eq(n));
            }
        }
    }

    let mut dists = vec![];
    for y in 0..h {
        for x in 0..w {
            let dist = &solver.int_var_2d((h, w), 0, max_num + 1);
            dists.push(dist.clone());

            for y2 in 0..h {
                for x2 in 0..w {
                    if y == y2 && x == x2 {
                        solver.add_expr(dist.at((y2, x2)).eq(0));
                    } else if y.abs_diff(y2) + x.abs_diff(x2) == 1 {
                        solver.add_expr(dist.at((y2, x2)).eq(1));
                    } else {
                        solver.add_expr(dist.at((y2, x2)).ge(2));
                    }
                }
            }

            for d in 2..=max_num {
                for y2 in 0..h {
                    for x2 in 0..w {
                        solver.add_expr(
                            dist.at((y2, x2)).le(d).iff(
                                (dist.four_neighbors((y2, x2)).le(d - 1)
                                    & !has_num.four_neighbors((y2, x2)))
                                .any()
                                    | dist.at((y2, x2)).le(d - 1),
                            ),
                        );
                    }
                }
            }

            solver.add_expr(
                has_num.at((y, x)).imp(
                    (dist.le(max_num) & num.eq(num.at((y, x))))
                        .count_true()
                        .eq(2),
                ),
            );
            solver.add_expr(
                has_num.at((y, x)).imp(
                    (dist.le(max_num) & num.eq(num.at((y, x))) & dist.eq(num.at((y, x))))
                        .count_true()
                        .eq(1),
                ),
            );
        }
    }

    solver.irrefutable_facts().map(|f| f.get(num))
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    KudamonoGrid::new(
        Optionalize::new(Choice::new(vec![
            Box::new(PrefixAndSuffix::new("(", DecInt, ")")),
            Box::new(AlphaToNum::new('a', 'w', 1)),
        ])),
        None,
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_kudamono_url_grid(combinator(), "soulmates", problem.clone())
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
            vec![Some(10), Some(1), None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![Some(3), None, None, None],
        ]
    }

    #[test]
    fn test_soulmates_problem() {
        let problem = problem_for_tests();
        let ans = solve_soulmates(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_2d([
            [10, 1, 10, 0],
            [0, 1, 0, 0],
            [0, 0, 3, 0],
            [3, 0, 0, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_parrot_loop_serializer() {
        let problem = problem_for_tests();
        let url = "https://pedros.works/paper-puzzle-player?W=3&H=3&L=(3)0(10)3(1)4&G=soulmates";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
