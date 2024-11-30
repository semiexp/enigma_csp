use super::util;
use crate::graph;
use crate::puzzle::slitherlink::combinator;
use crate::serializer::{problem_to_url, url_to_problem};
use crate::solver::{Solver, TRUE};

pub fn solve_litherslink(
    clues: &[Vec<Option<i32>>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(is_line.cell_neighbors((y, x)).count_true().eq(n));
            }
        }
    }

    for y in 0..=h {
        for x in 0..=w {
            let adj = is_line.vertex_neighbors((y, x));
            solver.add_expr(adj.any());
            solver.add_expr(adj.count_true().ne(2));
        }
    }

    // there are at least 2 trees
    let vertex_color = &solver.bool_var_2d((h + 1, w + 1));
    solver.add_expr(
        is_line.horizontal.imp(
            vertex_color
                .slice((.., ..w))
                .iff(vertex_color.slice((.., 1..))),
        ),
    );
    solver.add_expr(
        is_line.vertical.imp(
            vertex_color
                .slice((..h, ..))
                .iff(vertex_color.slice((1.., ..))),
        ),
    );
    solver.add_expr(vertex_color.any());
    solver.add_expr((!vertex_color).any());

    // no loop (that is, all cells are reachable to the outside of the grid)
    let mut aux_graph = graph::Graph::new(h * w + 1 + h * (w + 1) + (h + 1) * w);
    let mut indicator = vec![];

    for _ in 0..=(h * w) {
        indicator.push(TRUE);
    }

    for y in 0..=h {
        for x in 0..w {
            let v1;
            if y == 0 {
                v1 = h * w;
            } else {
                v1 = (y - 1) * w + x;
            }
            let v2;
            if y == h {
                v2 = h * w;
            } else {
                v2 = y * w + x;
            }

            let e = indicator.len();
            aux_graph.add_edge(e, v1);
            aux_graph.add_edge(e, v2);

            indicator.push(!is_line.horizontal.at((y, x)));
        }
    }

    for y in 0..h {
        for x in 0..=w {
            let v1;
            if x == 0 {
                v1 = h * w;
            } else {
                v1 = y * w + x - 1;
            }

            let v2;
            if x == w {
                v2 = h * w;
            } else {
                v2 = y * w + x;
            }

            let e = indicator.len();
            aux_graph.add_edge(e, v1);
            aux_graph.add_edge(e, v2);

            indicator.push(!is_line.vertical.at((y, x)));
        }
    }

    graph::active_vertices_connected(&mut solver, &indicator, &aux_graph);
    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<Option<i32>>>;

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "lither", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["lither"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![Some(1), None, None, Some(3)],
            vec![None, Some(3), None, None],
            vec![None, Some(1), None, Some(3)],
        ]
    }

    fn problem_for_tests2() -> Problem {
        vec![vec![Some(3), Some(3), Some(3)], vec![None, None, None]]
    }

    #[test]
    fn test_litherslink_problem() {
        let problem = problem_for_tests();
        let ans = solve_litherslink(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [1, 0, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 0, 1, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ]),
        };
        assert_eq!(ans, expected);

        // edges are not connected
        let problem = problem_for_tests2();
        let ans = solve_litherslink(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([[1, 1, 1, 1], [1, 0, 0, 1]]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_litherslink_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?lither/4/3/b8dg6d";

        // TODO: pass bidirectional test
        assert_eq!(deserialize_problem(url), Some(problem));
    }
}
