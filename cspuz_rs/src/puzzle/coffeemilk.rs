use std::vec;

use super::util;
use crate::graph;
use crate::serializer::{problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, Spaces};
use crate::solver::{count_true, Solver};

pub fn solve_coffeemilk(clues: &[Vec<i32>]) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let mut n_vertices = 0;
    let mut vertex_id: Vec<Vec<Option<usize>>> = vec![vec![None; w]; h];

    for y in 0..h {
        for x in 0..w {
            if clues[y][x] > 0 {
                vertex_id[y][x] = Some(n_vertices);
                n_vertices += 1;
            } else {
                if 0 < y && y < h - 1 && 0 < x && x < w - 1 {
                    solver.add_expr(!(is_line.horizontal.at((y, x)) & is_line.vertical.at((y, x))));
                }
            }
        }
    }

    let mut horizontal_edge_id: Vec<Vec<Option<usize>>> = vec![vec![None; w - 1]; h];
    let mut vertical_edge_id: Vec<Vec<Option<usize>>> = vec![vec![None; w]; h - 1];
    let mut edges = vec![];

    fn is_connectable(x: i32, y: i32) -> bool {
        !((x == 1 && y == 2) || (x == 2 && y == 1))
    }

    for y in 0..h {
        let mut last_x: Option<usize> = None;

        for x in 0..w {
            if clues[y][x] > 0 {
                if let Some(last_x) = last_x {
                    if is_connectable(clues[y][x], clues[y][last_x]) {
                        for x2 in last_x..x {
                            horizontal_edge_id[y][x2] = Some(edges.len());
                        }
                        edges.push((vertex_id[y][last_x].unwrap(), vertex_id[y][x].unwrap()));
                    }
                }
                last_x = Some(x);
            }
        }
    }

    for x in 0..w {
        let mut last_y: Option<usize> = None;

        for y in 0..h {
            if clues[y][x] > 0 {
                if let Some(last_y) = last_y {
                    if is_connectable(clues[y][x], clues[last_y][x]) {
                        for y2 in last_y..y {
                            vertical_edge_id[y2][x] = Some(edges.len());
                        }
                        edges.push((vertex_id[last_y][x].unwrap(), vertex_id[y][x].unwrap()));
                    }
                }
                last_y = Some(y);
            }
        }
    }

    let is_edge_connected = &solver.bool_var_1d(edges.len());

    for y in 0..h {
        for x in 0..(w - 1) {
            if let Some(edge_id) = horizontal_edge_id[y][x] {
                solver.add_expr(
                    is_line
                        .horizontal
                        .at((y, x))
                        .iff(is_edge_connected.at(edge_id)),
                );
            } else {
                solver.add_expr(!is_line.horizontal.at((y, x)));
            }
        }
    }

    for y in 0..(h - 1) {
        for x in 0..w {
            if let Some(edge_id) = vertical_edge_id[y][x] {
                solver.add_expr(
                    is_line
                        .vertical
                        .at((y, x))
                        .iff(is_edge_connected.at(edge_id)),
                );
            } else {
                solver.add_expr(!is_line.vertical.at((y, x)));
            }
        }
    }

    let mut grey_cells = vec![];
    for y in 0..h {
        for x in 0..w {
            if clues[y][x] == 3 {
                grey_cells.push(vertex_id[y][x].unwrap());
            }
        }
    }

    if grey_cells.len() == 0 {
        return None;
    }

    let mut aux_graph = graph::Graph::new(n_vertices);
    for i in 0..edges.len() {
        aux_graph.add_edge(edges[i].0, edges[i].1);
    }

    let vertex_group = &solver.int_var_1d(n_vertices, 0, grey_cells.len() as i32 - 1);

    for i in 0..is_edge_connected.len() {
        let (u, v) = edges[i];
        solver.add_expr(
            is_edge_connected
                .at(i)
                .imp(vertex_group.at(u).eq(vertex_group.at(v))),
        );
    }

    for i in 0..grey_cells.len() {
        solver.add_expr(vertex_group.at(grey_cells[i]).eq(i as i32));

        let mut white_cells_ind = vec![];
        let mut black_cells_ind = vec![];
        for y in 0..h {
            for x in 0..w {
                if clues[y][x] == 1 {
                    white_cells_ind.push(vertex_group.at(vertex_id[y][x].unwrap()).eq(i as i32));
                } else if clues[y][x] == 2 {
                    black_cells_ind.push(vertex_group.at(vertex_id[y][x].unwrap()).eq(i as i32));
                }
            }
        }

        solver.add_expr(count_true(white_cells_ind).eq(count_true(black_cells_ind)));
        graph::active_vertices_connected_via_active_edges(
            &mut solver,
            vertex_group.eq(i as i32),
            is_edge_connected,
            &aux_graph,
        );
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<i32>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Dict::new(1, "1")),
        Box::new(Dict::new(2, "2")),
        Box::new(Dict::new(3, ".")),
        Box::new(Spaces::new(0, 'a')),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "coffeemilk", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["coffeemilk"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![0, 2, 0, 0, 2, 3],
            vec![1, 3, 1, 0, 0, 0],
            vec![0, 0, 1, 0, 1, 0],
            vec![0, 0, 0, 0, 3, 2],
            vec![1, 0, 2, 0, 0, 2],
        ]
    }

    #[test]
    fn test_coffeemilk_problem() {
        let problem = problem_for_tests();
        let ans = solve_coffeemilk(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_coffeemilk_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?coffeemilk/6/5/a2b2.1.1e1a1e.21a2b2";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
