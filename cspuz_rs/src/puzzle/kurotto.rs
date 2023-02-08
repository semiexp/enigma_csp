use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::Solver;

pub fn solve_kurotto(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(!is_black.at((y, x)));
                if n < 0 {
                    continue;
                }

                let connected = &solver.bool_var_2d((h, w));
                for y2 in 0..h {
                    for x2 in 0..w {
                        if y == y2 && x == x2 {
                            solver.add_expr(connected.at((y2, x2)));
                        } else {
                            solver.add_expr(connected.at((y2, x2)).imp(is_black.at((y2, x2))));
                        }
                    }
                }
                solver.add_expr(connected.count_true().eq(n + 1));
                graph::active_vertices_connected_2d(&mut solver, connected);

                for nb in connected.four_neighbor_indices((y, x)) {
                    solver.add_expr(is_black.at(nb).imp(connected.at(nb)));
                }
                solver.add_expr(
                    (is_black.slice((1.., ..)) & is_black.slice((..(h - 1), ..))).imp(
                        connected
                            .slice((1.., ..))
                            .iff(connected.slice((..(h - 1), ..))),
                    ),
                );
                solver.add_expr(
                    (is_black.slice((.., 1..)) & is_black.slice((.., ..(w - 1)))).imp(
                        connected
                            .slice((.., 1..))
                            .iff(connected.slice((.., ..(w - 1)))),
                    ),
                );
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
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
    problem_to_url(combinator(), "kurotto", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["kurotto"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        // generated by cspuz
        // https://puzz.link/p?kurotto/6/6/3gah.m.i9.iam8h3g2
        vec![
            vec![Some(3), None, Some(10), None, None, Some(-1)],
            vec![None, None, None, None, None, None],
            vec![None, Some(-1), None, None, None, Some(9)],
            vec![Some(-1), None, None, None, Some(10), None],
            vec![None, None, None, None, None, None],
            vec![Some(8), None, None, Some(3), None, Some(2)],
        ]
    }

    #[test]
    fn test_kurotto_problem() {
        let problem = problem_for_tests();
        let ans = solve_kurotto(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_kurotto_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?kurotto/6/6/3gah.m.i9.iam8h3g2";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
