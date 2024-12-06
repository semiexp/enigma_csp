use super::util;
use crate::graph;
use crate::serializer::{
    get_kudamono_url_info, kudamono_url_info_to_problem, problem_to_kudamono_url_grid, Choice,
    Combinator, DecInt, Dict, KudamonoGrid, Optionalize, PrefixAndSuffix,
};
use crate::solver::{count_true, Solver, FALSE};

pub fn solve_spokes(
    clues: &[Vec<Option<i32>>],
) -> Option<(
    graph::BoolGridEdgesIrrefutableFacts,
    Vec<Vec<Option<bool>>>,
    Vec<Vec<Option<bool>>>,
)> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    let diagonal_dr = &solver.bool_var_2d((h - 1, w - 1));
    let diagonal_dl = &solver.bool_var_2d((h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);
    solver.add_answer_key_bool(diagonal_dr);
    solver.add_answer_key_bool(diagonal_dl);

    solver.add_expr(!(diagonal_dr & diagonal_dl));

    for y in 0..h {
        for x in 0..w {
            let mut neighbors = vec![];

            if y > 0 {
                neighbors.push(is_line.vertical.at((y - 1, x)).expr());
            } else {
                neighbors.push(FALSE);
            }
            if y < h - 1 {
                neighbors.push(is_line.vertical.at((y, x)).expr());
            } else {
                neighbors.push(FALSE);
            }

            if x > 0 {
                neighbors.push(is_line.horizontal.at((y, x - 1)).expr());
            } else {
                neighbors.push(FALSE);
            }
            if x < w - 1 {
                neighbors.push(is_line.horizontal.at((y, x)).expr());
            } else {
                neighbors.push(FALSE);
            }

            if y > 0 && x > 0 {
                neighbors.push(diagonal_dr.at((y - 1, x - 1)).expr());
            } else {
                neighbors.push(FALSE);
            }

            if y < h - 1 && x < w - 1 {
                neighbors.push(diagonal_dr.at((y, x)).expr());
            } else {
                neighbors.push(FALSE);
            }

            if y > 0 && x < w - 1 {
                neighbors.push(diagonal_dl.at((y - 1, x)).expr());
            } else {
                neighbors.push(FALSE);
            }

            if y < h - 1 && x > 0 {
                neighbors.push(diagonal_dl.at((y, x - 1)).expr());
            } else {
                neighbors.push(FALSE);
            }

            if let Some(n) = clues[y][x] {
                // -1 stands for "no constraint"
                if n >= 0 {
                    solver.add_expr(count_true(neighbors).eq(n));
                }
            } else {
                for i in 0..4 {
                    solver.add_expr(neighbors[i * 2].iff(&neighbors[i * 2 + 1]));
                }
                solver.add_expr(count_true(neighbors).le(2));
            }
        }
    }

    let mut aux_graph = graph::Graph::new(h * w);
    let mut aux_edges = vec![];

    for y in 0..h {
        for x in 0..(w - 1) {
            aux_graph.add_edge(y * w + x, y * w + x + 1);
            aux_edges.push(is_line.horizontal.at((y, x)));
        }
    }
    for y in 0..(h - 1) {
        for x in 0..w {
            aux_graph.add_edge(y * w + x, (y + 1) * w + x);
            aux_edges.push(is_line.vertical.at((y, x)));
        }
    }
    for y in 0..(h - 1) {
        for x in 0..(w - 1) {
            aux_graph.add_edge(y * w + x, (y + 1) * w + x + 1);
            aux_edges.push(diagonal_dr.at((y, x)));

            aux_graph.add_edge(y * w + x + 1, (y + 1) * w + x);
            aux_edges.push(diagonal_dl.at((y, x)));
        }
    }

    graph::active_vertices_connected(&mut solver, &aux_edges, &aux_graph.line_graph());

    solver
        .irrefutable_facts()
        .map(|f| (f.get(is_line), f.get(diagonal_dr), f.get(diagonal_dl)))
}

pub type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    KudamonoGrid::new(
        Choice::new(vec![
            Box::new(Dict::new(None, "x")),
            Box::new(Optionalize::new(PrefixAndSuffix::new("(", DecInt, ")"))),
        ]),
        Some(-1),
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_kudamono_url_grid(combinator(), "spokes", problem.clone())
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
            vec![None, Some(-1), Some(2), Some(-1), None],
            vec![Some(2), Some(1), Some(1), Some(1), Some(-1)],
            vec![Some(2), None, Some(3), Some(2), Some(-1)],
            vec![Some(3), None, Some(1), Some(4), Some(1)],
        ]
    }

    #[test]
    fn test_spokes_problem() {
        let problem = problem_for_tests();
        let ans = solve_spokes(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = (
            graph::BoolGridEdgesIrrefutableFacts {
                horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [1, 1, 0, 1],
                ]),
                vertical: crate::puzzle::util::tests::to_option_bool_2d([
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0],
                ]),
            },
            crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
            ]),
            crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ]),
        );
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_spokes_serializer() {
        let problem = problem_for_tests();
        let url =
            "https://pedros.works/paper-puzzle-player?W=5x4&L=(3)0(2)1(2)1x1x1x1(1)1(1)2(3)1(1)1(2)1(4)1(2)1(1)1(1)2x3&G=spokes";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
