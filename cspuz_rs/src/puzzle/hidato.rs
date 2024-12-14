use super::util;
use crate::serializer::{
    get_kudamono_url_info, kudamono_url_info_to_problem, problem_to_kudamono_url_grid, AlphaToNum,
    Choice, Combinator, DecInt, Dict, KudamonoGrid, Optionalize, PrefixAndSuffix,
};
use crate::solver::Solver;

pub fn solve_hidato(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<i32>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut num_cells = 0;
    for y in 0..h {
        for x in 0..w {
            if clues[y][x] != Some(-1) {
                num_cells += 1;
            }
        }
    }

    let mut solver = Solver::new();
    let num = &solver.int_var_2d((h, w), 0, num_cells);
    solver.add_answer_key_int(num);

    for i in 0..(h * w) {
        for j in 0..i {
            let yi = i / w;
            let xi = i % w;
            let yj = j / w;
            let xj = j % w;

            if clues[yi][xi] != Some(-1) && clues[yj][xj] != Some(-1) {
                solver.add_expr(num.at((yi, xi)).ne(num.at((yj, xj))));
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            if clues[y][x] == Some(-1) {
                solver.add_expr(num.at((y, x)).eq(0));
                continue;
            }

            solver.add_expr(num.at((y, x)).ne(0));

            if let Some(n) = clues[y][x] {
                if n > 0 {
                    solver.add_expr(num.at((y, x)).eq(n));
                }
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            let ylo = if y == 0 { 0 } else { y - 1 };
            let yhi = if y == h - 1 { h - 1 } else { y + 1 };
            let xlo = if x == 0 { 0 } else { x - 1 };
            let xhi = if x == w - 1 { w - 1 } else { x + 1 };

            solver.add_expr(
                (num.at((y, x)).ne(0) & num.at((y, x)).ne(1)).imp(
                    num.slice((ylo..=yhi, xlo..=xhi))
                        .eq(num.at((y, x)) - 1)
                        .any(),
                ),
            );
            solver.add_expr(
                (num.at((y, x)).ne(0) & num.at((y, x)).ne(num_cells)).imp(
                    num.slice((ylo..=yhi, xlo..=xhi))
                        .eq(num.at((y, x)) + 1)
                        .any(),
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
            Box::new(Dict::new(-1, "x")),
            Box::new(PrefixAndSuffix::new("(", DecInt, ")")),
        ])),
        None,
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_kudamono_url_grid(combinator(), "hidoku", problem.clone())
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
            vec![Some(4), None, None, Some(11)],
            vec![None, None, None, None],
            vec![None, Some(6), Some(-1), Some(1)],
        ]
    }

    #[test]
    fn test_hidato_problem() {
        let problem = problem_for_tests();
        let ans = solve_hidato(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected =
            crate::puzzle::util::tests::to_option_2d([[4, 3, 9, 11], [5, 8, 2, 10], [7, 6, 0, 1]]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_hidato_serializer() {
        let problem = problem_for_tests();
        let url = "https://pedros.works/paper-puzzle-player?W=4x3&L=(4)2(6)1x3(1)3(11)2&G=hidoku";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
