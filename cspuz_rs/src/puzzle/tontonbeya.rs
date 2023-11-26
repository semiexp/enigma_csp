use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, AlphaToNum, Choice, Combinator, Context,
    ContextBasedGrid, Optionalize, Rooms, Size, Spaces, Tuple2,
};
use crate::solver::{any, count_true, Solver};

pub fn solve_tontonbeya(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Vec<Option<i32>>],
) -> Option<Vec<Vec<Option<i32>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let ans = &solver.int_var_2d((h, w), 0, 2);
    solver.add_answer_key_int(ans);

    let rooms = graph::borders_to_rooms(borders);
    let mut room_id = vec![vec![0; w]; h];
    let mut idx_in_room = vec![vec![0; w]; h];
    for i in 0..rooms.len() {
        for j in 0..rooms[i].len() {
            let (y, x) = rooms[i][j];
            room_id[y][x] = i;
            idx_in_room[y][x] = j;
        }
    }

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(ans.at((y, x)).eq(n));
            }
        }
    }
    for i in 0..rooms.len() {
        let mut aux_graph = graph::Graph::new(rooms[i].len());
        for &(y, x) in &rooms[i] {
            if y > 0 && room_id[y - 1][x] == i {
                aux_graph.add_edge(idx_in_room[y][x], idx_in_room[y - 1][x]);
            }
            if x > 0 && room_id[y][x - 1] == i {
                aux_graph.add_edge(idx_in_room[y][x], idx_in_room[y][x - 1]);
            }
        }

        let mut adj = vec![];
        for &(y, x) in &rooms[i] {
            if y > 0 && room_id[y - 1][x] != i {
                adj.push((room_id[y - 1][x], (y, x), (y - 1, x)));
            }
            if y < h - 1 && room_id[y + 1][x] != i {
                adj.push((room_id[y + 1][x], (y, x), (y + 1, x)));
            }
            if x > 0 && room_id[y][x - 1] != i {
                adj.push((room_id[y][x - 1], (y, x), (y, x - 1)));
            }
            if x < w - 1 && room_id[y][x + 1] != i {
                adj.push((room_id[y][x + 1], (y, x), (y, x + 1)));
            }
        }
        adj.sort();

        let cnt = solver.int_var(1, rooms[i].len() as i32);
        for j in 0..3 {
            let mut indicator = vec![];
            for &(y, x) in &rooms[i] {
                indicator.push(ans.at((y, x)).eq(j));
            }
            solver.add_expr(any(&indicator).imp(count_true(&indicator).eq(&cnt)));
            graph::active_vertices_connected(&mut solver, &indicator, &aux_graph);

            let mut has_nb = vec![];
            let mut cand = vec![];
            for k in 0..adj.len() {
                let (rid, cell, cell_nb) = adj[k];
                cand.push(ans.at(cell).eq(j) & ans.at(cell_nb).eq(j));

                if k + 1 == adj.len() || rid != adj[k + 1].0 {
                    has_nb.push(any(&cand));
                    cand.clear();
                }
            }
            solver.add_expr(any(&indicator).imp(count_true(has_nb).eq(1)));
        }
    }

    solver.irrefutable_facts().map(|f| f.get(ans))
}

type Problem = (graph::InnerGridEdges<Vec<Vec<bool>>>, Vec<Vec<Option<i32>>>);

fn combinator() -> impl Combinator<Problem> {
    Size::new(Tuple2::new(
        Rooms,
        ContextBasedGrid::new(Choice::new(vec![
            Box::new(Optionalize::new(AlphaToNum::new('1', '3', 0))),
            Box::new(Spaces::new(None, 'a')),
        ])),
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let height = problem.0.vertical.len();
    let width = problem.0.vertical[0].len() + 1;
    problem_to_url_with_context(
        combinator(),
        "tontonbeya",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["tontonbeya"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        (
            graph::InnerGridEdges {
                horizontal: crate::puzzle::util::tests::to_bool_2d([
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                ]),
                vertical: crate::puzzle::util::tests::to_bool_2d([
                    [0, 1, 0, 1, 0],
                    [1, 0, 0, 1, 0],
                    [1, 1, 0, 1, 0],
                    [1, 0, 1, 1, 0],
                    [0, 0, 0, 1, 0],
                ]),
            },
            vec![
                vec![None, Some(0), None, None, None, None],
                vec![None, Some(2), None, None, None, None],
                vec![Some(1), None, None, None, None, None],
                vec![None, None, None, Some(0), None, None],
                vec![None, None, None, None, None, None],
            ]
        )
    }

    #[test]
    fn test_tontonbeya_problem() {
        let (borders, clues) = problem_for_tests();
        let ans = solve_tontonbeya(&borders, &clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_2d([
            [0, 0, 0, 0, 1, 1],
            [1, 2, 0, 0, 1, 1],
            [1, 2, 2, 1, 1, 0],
            [1, 2, 2, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_tontonbeya_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?tontonbeya/6/5/aiqm28351oa1e3d2h1h";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
