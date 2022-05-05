use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::{any, Solver};

pub fn solve_nurimisaki(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_white = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_white);

    graph::active_vertices_connected_2d(&mut solver, is_white);

    for y in 0..(h - 1) {
        for x in 0..(w - 1) {
            solver.add_expr(is_white.slice((y..(y + 2), x..(x + 2))).any());
            solver.add_expr(!is_white.slice((y..(y + 2), x..(x + 2))).all());
        }
    }
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(is_white.at((y, x)));
                solver.add_expr(is_white.four_neighbors((y, x)).count_true().eq(1));
                if n > 0 {
                    let n = n as usize;
                    let mut dirs = vec![];
                    if y >= n - 1 {
                        if y == n - 1 {
                            dirs.push(is_white.slice_fixed_x(((y - (n - 1))..y, x)).all());
                        } else {
                            dirs.push(
                                is_white.slice_fixed_x(((y - (n - 1))..y, x)).all()
                                    & !is_white.at((y - n, x)),
                            );
                        }
                    }
                    if x >= n - 1 {
                        if x == n - 1 {
                            dirs.push(is_white.slice_fixed_y((y, (x - (n - 1))..x)).all());
                        } else {
                            dirs.push(
                                is_white.slice_fixed_y((y, (x - (n - 1))..x)).all()
                                    & !is_white.at((y, x - n)),
                            );
                        }
                    }
                    if h - y >= n {
                        if y == h - n {
                            dirs.push(is_white.slice_fixed_x(((y + 1)..(y + n), x)).all());
                        } else {
                            dirs.push(
                                is_white.slice_fixed_x(((y + 1)..(y + n), x)).all()
                                    & !is_white.at((y + n, x)),
                            );
                        }
                    }
                    if w - x >= n {
                        if x == w - n {
                            dirs.push(is_white.slice_fixed_y((y, (x + 1)..(x + n))).all());
                        } else {
                            dirs.push(
                                is_white.slice_fixed_y((y, (x + 1)..(x + n))).all()
                                    & !is_white.at((y, x + n)),
                            );
                        }
                    }
                    solver.add_expr(any(dirs));
                }
            } else {
                solver.add_expr(
                    is_white
                        .at((y, x))
                        .imp(is_white.four_neighbors((y, x)).count_true().ne(1)),
                );
            }
        }
    }
    solver.irrefutable_facts().map(|f| f.get(is_white))
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Optionalize::new(HexInt)),
        Box::new(Spaces::new(None, 'g')),
        Box::new(Dict::new(Some(-1), ".")),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "nurimisaki", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["nurimisaki"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        // https://twitter.com/semiexp/status/1168898897424633856
        vec![
            vec![None, None, None, None, Some(3), None, None, None, None, None],
            vec![None, Some(3), None, None, None, None, None, None, None, None],
            vec![None, None, None, None, None, None, None, None, Some(2), None],
            vec![None, None, None, None, None, None, None, None, None, None],
            vec![None, None, None, Some(2), None, None, None, None, None, None],
            vec![None, None, None, None, Some(-1), None, Some(2), None, None, None],
            vec![None, Some(2), None, None, None, None, None, None, None, None],
            vec![None, None, None, None, None, None, None, None, None, Some(2)],
            vec![None, None, None, None, None, Some(2), None, None, None, None],
            vec![None, None, None, None, Some(3), None, None, None, None, None],
        ]
    }

    #[test]
    fn test_nurimisaki_problem() {
        let problem = problem_for_tests();
        let ans = solve_nurimisaki(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = [
            [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        ];
        for y in 0..10 {
            for x in 0..10 {
                assert_eq!(
                    ans[y][x],
                    Some(expected[y][x] == 0),
                    "mismatch at ({}, {})",
                    y,
                    x
                );
            }
        }
    }

    #[test]
    fn test_nurimisaki_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?nurimisaki/10/10/j3l3v2t2p.g2j2w2k2n3k";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
