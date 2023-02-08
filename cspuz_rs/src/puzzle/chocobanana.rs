use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::{any, Solver, TRUE};

pub fn solve_chocobanana(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    let is_border = graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_expr(
        (is_black.slice((.., ..(w - 1))) ^ is_black.slice((.., 1..))).iff(&is_border.vertical),
    );
    solver.add_expr(
        (is_black.slice((..(h - 1), ..)) ^ is_black.slice((1.., ..))).iff(&is_border.horizontal),
    );

    let mut sizes = vec![];
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                let v = solver.int_var(n, n);
                sizes.push(Some(v));
            } else {
                sizes.push(None);
            }
        }
    }
    let mut edges = vec![];
    let mut edge_vars = vec![];
    for y in 0..h {
        for x in 0..w {
            if y < h - 1 {
                edges.push((y * w + x, (y + 1) * w + x));
                edge_vars.push(is_border.horizontal.at((y, x)));
            }
            if x < w - 1 {
                edges.push((y * w + x, y * w + x + 1));
                edge_vars.push(is_border.vertical.at((y, x)));
            }
        }
    }
    solver.add_graph_division(&sizes, &edges, &edge_vars);

    for y in 0..(h - 1) {
        for x in 0..(w - 1) {
            solver.add_expr(is_black.slice((y..(y + 2), x..(x + 2))).count_true().ne(3));
        }
    }

    let mut aux_graph = graph::Graph::new(h * w * 2 + 1);
    let mut aux_graph_v = vec![];
    for y in 0..h {
        for x in 0..w {
            if y < h - 1 {
                aux_graph.add_edge((y * w + x) * 2, ((y + 1) * w + x) * 2);
            }
            if x < w - 1 {
                aux_graph.add_edge((y * w + x) * 2, (y * w + x + 1) * 2);
            }
            aux_graph.add_edge((y * w + x) * 2, (y * w + x) * 2 + 1);
            aux_graph.add_edge((y * w + x) * 2 + 1, h * w * 2);

            aux_graph_v.push(!is_black.at((y, x)));
            let v = solver.bool_var();
            let mut corner = vec![];
            if y > 0 && x > 0 {
                corner.push(
                    !is_black.at((y, x))
                        & !is_black.at((y - 1, x))
                        & !is_black.at((y, x - 1))
                        & is_black.at((y - 1, x - 1)),
                );
            }
            if y > 0 && x < w - 1 {
                corner.push(
                    !is_black.at((y, x))
                        & !is_black.at((y - 1, x))
                        & !is_black.at((y, x + 1))
                        & is_black.at((y - 1, x + 1)),
                );
            }
            if y < h - 1 && x > 0 {
                corner.push(
                    !is_black.at((y, x))
                        & !is_black.at((y + 1, x))
                        & !is_black.at((y, x - 1))
                        & is_black.at((y + 1, x - 1)),
                );
            }
            if y < h - 1 && x < w - 1 {
                corner.push(
                    !is_black.at((y, x))
                        & !is_black.at((y + 1, x))
                        & !is_black.at((y, x + 1))
                        & is_black.at((y + 1, x + 1)),
                );
            }
            solver.add_expr(v.iff(any(corner)));
            aux_graph_v.push(v.expr());
        }
    }
    aux_graph_v.push(TRUE);
    graph::active_vertices_connected(&mut solver, &aux_graph_v, &aux_graph);

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
    problem_to_url(combinator(), "cbanana", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["cbanana"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![Some(3), Some(1), None, None, Some(2), None],
            vec![None, None, None, None, None, None],
            vec![None, Some(6), Some(6), None, None, None],
            vec![None, None, None, Some(8), None, None],
            vec![None, None, None, None, None, None],
            vec![None, None, None, None, None, None],
        ]
    }

    #[test]
    fn test_chocobanana_problem() {
        let problem = problem_for_tests();
        let ans = solve_chocobanana(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_chocobanana_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?cbanana/6/6/31h2n66l8t";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
