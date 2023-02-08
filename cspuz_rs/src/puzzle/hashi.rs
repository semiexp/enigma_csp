use super::util;
use crate::graph::{self, GridEdges};
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::{sum, Solver};

pub fn solve_hashi(clues: &[Vec<Option<i32>>]) -> Option<GridEdges<Vec<Vec<Option<i32>>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let answer_horizontal = &solver.int_var_2d((h, w - 1), 0, 2);
    let answer_vertical = &solver.int_var_2d((h - 1, w), 0, 2);
    solver.add_answer_key_int(answer_horizontal);
    solver.add_answer_key_int(answer_vertical);

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                let mut deg = vec![];
                if y > 0 {
                    deg.push(answer_vertical.at((y - 1, x)));
                }
                if y < h - 1 {
                    deg.push(answer_vertical.at((y, x)));
                }
                if x > 0 {
                    deg.push(answer_horizontal.at((y, x - 1)));
                }
                if x < w - 1 {
                    deg.push(answer_horizontal.at((y, x)));
                }

                if n >= 0 {
                    solver.add_expr(sum(deg).eq(n));
                } else {
                    solver.add_expr(sum(deg).gt(0));
                }
            } else {
                if y == 0 {
                    solver.add_expr(answer_vertical.at((y, x)).eq(0));
                } else if y == h - 1 {
                    solver.add_expr(answer_vertical.at((y - 1, x)).eq(0));
                } else {
                    solver.add_expr(
                        answer_vertical
                            .at((y - 1, x))
                            .eq(answer_vertical.at((y, x))),
                    );
                }
                if x == 0 {
                    solver.add_expr(answer_horizontal.at((y, x)).eq(0));
                } else if x == w - 1 {
                    solver.add_expr(answer_horizontal.at((y, x - 1)).eq(0));
                } else {
                    solver.add_expr(
                        answer_horizontal
                            .at((y, x - 1))
                            .eq(answer_horizontal.at((y, x))),
                    );
                }

                if 0 < y && y < h - 1 && 0 < x && x < w - 1 {
                    solver.add_expr(
                        !(answer_horizontal.at((y, x - 1)).gt(0)
                            & answer_vertical.at((y - 1, x)).gt(0)),
                    );
                }
            }
        }
    }

    let is_connected = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_expr(is_connected.horizontal.iff(answer_horizontal.gt(0)));
    solver.add_expr(is_connected.vertical.iff(answer_vertical.gt(0)));

    let (edges, g) = is_connected.representation();
    graph::active_vertices_connected(&mut solver, edges, &g.line_graph());

    solver.irrefutable_facts().map(|f| GridEdges {
        horizontal: f.get(answer_horizontal),
        vertical: f.get(answer_vertical),
    })
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
    problem_to_url(combinator(), "hashi", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["hashi"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![Some(3), None, Some(1), None, Some(2), Some(-1)],
            vec![None, Some(-1), None, Some(2), None, None],
            vec![Some(-1), None, Some(2), None, Some(-1), None],
            vec![None, Some(-1), None, Some(4), None, Some(3)],
            vec![None, None, Some(4), None, Some(-1), None],
            vec![Some(2), None, Some(-1), None, None, Some(2)],
        ]
    }

    #[test]
    fn test_hashi_problem() {
        let problem = problem_for_tests();
        let ans = solve_hashi(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::GridEdges {
            horizontal: crate::puzzle::util::tests::to_option_2d([
                [1, 1, 0, 0, 1],
                [0, 1, 1, 0, 0],
                [2, 2, 0, 0, 0],
                [0, 2, 2, 1, 1],
                [0, 0, 2, 2, 0],
                [1, 1, 1, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_2d([
                [2, 0, 0, 0, 1, 1],
                [2, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 2, 0, 0, 1],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_hashi_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?hashi/6/6/3g1g2.g.g2h.g2g.h.g4g3h4g.g2g.h2";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
