use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::{int_constant, Solver};

pub fn solve_aquapelago(problem: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(problem);

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);
    solver.add_expr(!is_black.conv2d_and((1, 2)));
    solver.add_expr(!is_black.conv2d_and((2, 1)));
    solver.add_expr(is_black.conv2d_or((2, 2)));
    graph::active_vertices_connected_2d(&mut solver, !is_black);

    let mut aux_graph = vec![];
    let mut aux_sizes = vec![];
    let mut aux_edges = vec![];

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = problem[y][x] {
                solver.add_expr(is_black.at((y, x)));
                if n > 0 {
                    aux_sizes.push(Some(int_constant(n)));
                } else {
                    aux_sizes.push(None);
                }
            } else {
                aux_sizes.push(None);
            }

            if y < h - 1 {
                if x < w - 1 {
                    aux_graph.push((y * w + x, (y + 1) * w + x + 1));
                    aux_edges.push(!(is_black.at((y, x)) & is_black.at((y + 1, x + 1))));
                }
                if x > 0 {
                    aux_graph.push((y * w + x, (y + 1) * w + x - 1));
                    aux_edges.push(!(is_black.at((y, x)) & is_black.at((y + 1, x - 1))));
                }
            }
        }
    }
    solver.add_graph_division(&aux_sizes, &aux_graph, &aux_edges);

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
    problem_to_url(combinator(), "aquapelago", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["aquapelago"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![None, None, Some(1), None, None, None],
            vec![None, None, None, None, None, None],
            vec![None, Some(3), None, None, None, None],
            vec![None, None, None, None, None, None],
            vec![None, None, None, None, None, None],
        ]
    }

    #[test]
    fn test_aquapelago_problem() {
        let problem = problem_for_tests();
        let ans = solve_aquapelago(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_aquapelago_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?aquapelago/6/5/h1p3v";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
