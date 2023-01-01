use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::{int_constant, BoolExpr, IntExpr, Solver};

pub fn solve_cave(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    // white cells are connected
    graph::active_vertices_connected_2d(&mut solver, !is_black);

    let mut aux_graph = graph::Graph::new(h * w + 1);
    for y in 0..h {
        for x in 0..w {
            if y == 0 || y == h - 1 || x == 0 || x == w - 1 {
                aux_graph.add_edge(y * w + x, h * w);
            }
            if y > 0 {
                aux_graph.add_edge((y - 1) * w + x, y * w + x);
            }
            if x > 0 {
                aux_graph.add_edge(y * w + (x - 1), y * w + x);
            }
        }
    }
    let mut aux_vertices = is_black.flatten().into_iter().collect::<Vec<_>>();
    let t = solver.bool_var();
    solver.add_expr(&t);
    aux_vertices.push(t);
    graph::active_vertices_connected(&mut solver, &aux_vertices, &aux_graph);

    fn consecutive_true(seq: &[BoolExpr]) -> IntExpr {
        let mut ret = int_constant(0);
        for v in seq.iter().rev() {
            ret = v.ite(ret + 1, 0);
        }
        ret
    }

    let is_white = &!is_black;

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(!is_black.at((y, x)));
                if n < 0 {
                    continue;
                }
                let up = is_white
                    .slice_fixed_x((..y, x))
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();
                let down = is_white
                    .slice_fixed_x(((y + 1).., x))
                    .into_iter()
                    .collect::<Vec<_>>();
                let left = is_white
                    .slice_fixed_y((y, ..x))
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();
                let right = is_white
                    .slice_fixed_y((y, (x + 1)..))
                    .into_iter()
                    .collect::<Vec<_>>();

                solver.add_expr(
                    (consecutive_true(&up)
                        + consecutive_true(&down)
                        + consecutive_true(&left)
                        + consecutive_true(&right)
                        + 1)
                    .eq(n),
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
    problem_to_url(combinator(), "cave", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["cave"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        // https://puzz.link/p?cave/6/6/k3h6j2l7g3g2h3n
        vec![
            vec![None, None, None, None, None, Some(3)],
            vec![None, None, Some(6), None, None, None],
            vec![None, Some(2), None, None, None, None],
            vec![None, None, Some(7), None, Some(3), None],
            vec![Some(2), None, None, Some(3), None, None],
            vec![None, None, None, None, None, None],
        ]
    }

    #[test]
    fn test_cave_problem() {
        let problem = problem_for_tests();
        let ans = solve_cave(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ];
        for y in 0..6 {
            for x in 0..6 {
                assert_eq!(
                    ans[y][x],
                    Some(expected[y][x] == 1),
                    "mismatch at ({}, {})",
                    y,
                    x
                );
            }
        }
    }

    #[test]
    fn test_kurotto_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?cave/6/6/k3h6j2l7g3g2h3n";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
