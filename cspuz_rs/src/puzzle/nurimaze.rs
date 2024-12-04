use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, ContextBasedGrid,
    Dict, Rooms, Size, Spaces, Tuple2,
};
use crate::solver::{Solver, TRUE};

pub fn solve_nurimaze(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Vec<i32>],
) -> Option<Vec<Vec<Option<bool>>>> {
    /*
    0: empty
    1: start
    2: goal
    3: circle (pass)
    4: triangle (no pass)
     */
    let (h, w) = borders.base_shape();

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    // all black / all white in each region
    for y in 0..h {
        for x in 0..(w - 1) {
            if !borders.vertical[y][x] {
                solver.add_expr(is_black.at((y, x)).iff(is_black.at((y, x + 1))));
            }
        }
    }
    for y in 0..(h - 1) {
        for x in 0..w {
            if !borders.horizontal[y][x] {
                solver.add_expr(is_black.at((y, x)).iff(is_black.at((y + 1, x))));
            }
        }
    }

    // white cells are connected
    graph::active_vertices_connected_2d(&mut solver, !is_black);

    // white cells are acyclic
    let mut aux_graph = graph::Graph::new(h * w + 1);
    let mut aux_graph_vertices = vec![];
    for y in 0..h {
        for x in 0..w {
            aux_graph_vertices.push(is_black.at((y, x)).expr());

            if y == 0 || y == h - 1 || x == 0 || x == w - 1 {
                aux_graph.add_edge(y * w + x, h * w);
            }

            if y < h - 1 {
                aux_graph.add_edge(y * w + x, (y + 1) * w + x);
            }
            if x < w - 1 {
                aux_graph.add_edge(y * w + x, y * w + x + 1);
            }
            if y < h - 1 && x > 0 {
                aux_graph.add_edge(y * w + x, (y + 1) * w + x - 1);
            }
            if y < h - 1 && x < w - 1 {
                aux_graph.add_edge(y * w + x, (y + 1) * w + x + 1);
            }
        }
    }
    aux_graph_vertices.push(TRUE);
    graph::active_vertices_connected(&mut solver, aux_graph_vertices, &aux_graph);

    // no 2x2 all-black/white cells
    solver.add_expr(!(is_black.conv2d_and((2, 2))));
    solver.add_expr(is_black.conv2d_or((2, 2)));

    // cells with symbols cannot be black
    for y in 0..h {
        for x in 0..w {
            if clues[y][x] > 0 {
                solver.add_expr(!is_black.at((y, x)));
            }
        }
    }

    let the_path = &solver.bool_var_2d((h, w));
    solver.add_expr(the_path.imp(!is_black));
    graph::active_vertices_connected_2d(&mut solver, the_path);

    for y in 0..h {
        for x in 0..w {
            match clues[y][x] {
                0 => {
                    solver.add_expr(
                        the_path
                            .at((y, x))
                            .imp(the_path.four_neighbors((y, x)).count_true().eq(2)),
                    );
                }
                1 | 2 => {
                    solver.add_expr(the_path.at((y, x)));
                    solver.add_expr(the_path.four_neighbors((y, x)).count_true().eq(1));
                }
                3 => {
                    solver.add_expr(the_path.at((y, x)));
                    solver.add_expr(the_path.four_neighbors((y, x)).count_true().eq(2));
                }
                4 => {
                    solver.add_expr(!the_path.at((y, x)));
                }
                _ => panic!(),
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

pub type Problem = (graph::InnerGridEdges<Vec<Vec<bool>>>, Vec<Vec<i32>>);

pub(super) fn combinator() -> impl Combinator<Problem> {
    Size::new(Tuple2::new(
        Rooms,
        ContextBasedGrid::new(Choice::new(vec![
            Box::new(Dict::new(1, "1")),
            Box::new(Dict::new(2, "2")),
            Box::new(Dict::new(3, "3")),
            Box::new(Dict::new(4, "4")),
            Box::new(Spaces::new(0, '5')),
        ])),
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let height = problem.0.vertical.len();
    let width = problem.0.vertical[0].len() + 1;
    problem_to_url_with_context(
        combinator(),
        "nurimaze",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["nurimaze"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        let borders = graph::InnerGridEdges {
            horizontal: crate::puzzle::util::tests::to_bool_2d([
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_bool_2d([
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 0, 0, 1, 0],
            ]),
        };

        let clues = vec![
            vec![0, 3, 0, 0, 0, 0],
            vec![0, 0, 0, 4, 0, 0],
            vec![0, 0, 1, 0, 0, 0],
            vec![0, 0, 0, 0, 2, 0],
            vec![0, 0, 0, 0, 0, 0],
        ];

        (borders, clues)
    }

    #[test]
    fn test_nurimaze_problem() {
        let (borders, clues) = problem_for_tests();
        let ans = solve_nurimaze(&borders, &clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_nurimaze_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?nurimaze/6/5/ervrivfppu53b481b2b";
        crate::puzzle::util::tests::serializer_test(
            problem,
            url,
            serialize_problem,
            deserialize_problem,
        );
    }
}
