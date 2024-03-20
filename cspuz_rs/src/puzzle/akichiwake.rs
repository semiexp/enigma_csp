use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, HexInt, Optionalize,
    RoomsWithValues, Size, Spaces,
};
use crate::solver::Solver;

pub fn solve_akichiwake(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Option<i32>],
) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = borders.base_shape();

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    graph::active_vertices_connected_2d(&mut solver, !is_black);
    solver.add_expr(!is_black.conv2d_and((1, 2)));
    solver.add_expr(!is_black.conv2d_and((2, 1)));

    for y in 0..h {
        for x in 0..w {
            if y + 2 < h && borders.horizontal[y][x] {
                let mut y2 = y + 2;
                while y2 < h && !borders.horizontal[y2 - 1][x] {
                    y2 += 1;
                }
                if y2 < h {
                    solver.add_expr(is_black.slice_fixed_x((y..=y2, x)).any());
                }
            }
            if x + 2 < w && borders.vertical[y][x] {
                let mut x2 = x + 2;
                while x2 < w && !borders.vertical[y][x2 - 1] {
                    x2 += 1;
                }
                if x2 < w {
                    solver.add_expr(is_black.slice_fixed_y((y, x..=x2)).any());
                }
            }
        }
    }

    let rooms = graph::borders_to_rooms(borders);
    assert_eq!(rooms.len(), clues.len());

    let mut room_id = vec![vec![0; w]; h];
    let mut idx_in_room = vec![vec![0; w]; h];
    for i in 0..rooms.len() {
        for j in 0..rooms[i].len() {
            let (y, x) = rooms[i][j];
            room_id[y][x] = i;
            idx_in_room[y][x] = j;
        }
    }

    for i in 0..rooms.len() {
        if let Some(n) = clues[i] {
            if n == 0 {
                for &p in &rooms[i] {
                    solver.add_expr(is_black.at(p));
                }
                continue;
            }
            let sizes = &solver.int_var_1d(rooms[i].len(), 1, n);
            solver.add_expr(sizes.ge(n).any());

            let mut edges = vec![];
            let mut has_edge = vec![];
            for &(y, x) in &rooms[i] {
                if y > 0 && room_id[y - 1][x] == i {
                    edges.push((idx_in_room[y][x], idx_in_room[y - 1][x]));
                    has_edge.push(is_black.at((y, x)) | is_black.at((y - 1, x)));
                }
                if x > 0 && room_id[y][x - 1] == i {
                    edges.push((idx_in_room[y][x], idx_in_room[y][x - 1]));
                    has_edge.push(is_black.at((y, x)) | is_black.at((y, x - 1)));
                }
            }

            let sizes = sizes
                .into_iter()
                .map(|x| Some(x.expr()))
                .collect::<Vec<_>>();
            solver.add_graph_division(&sizes, &edges, has_edge);
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

pub(super) type Problem = (graph::InnerGridEdges<Vec<Vec<bool>>>, Vec<Option<i32>>);

pub(super) fn combinator() -> impl Combinator<Problem> {
    Size::new(RoomsWithValues::new(Choice::new(vec![
        Box::new(Optionalize::new(HexInt)),
        Box::new(Spaces::new(None, 'g')),
    ])))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let height = problem.0.vertical.len();
    let width = problem.0.vertical[0].len() + 1;
    problem_to_url_with_context(
        combinator(),
        "akichi",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["akichi"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        (
            graph::InnerGridEdges {
                horizontal: crate::puzzle::util::tests::to_bool_2d([
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                ]),
                vertical: crate::puzzle::util::tests::to_bool_2d([
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1],
                    [0, 1, 0, 0, 1],
                    [0, 1, 0, 0, 1],
                ]),
            },
            vec![Some(3), Some(2), Some(1), Some(3), None, Some(5)],
        )
    }

    #[test]
    fn test_akichiwake_problem() {
        let (borders, clues) = problem_for_tests();
        let ans = solve_akichiwake(&borders, &clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [1, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_akichiwake_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?akichi/6/5/455993g7o03213g5";
        crate::puzzle::util::tests::serializer_test(
            problem,
            url,
            serialize_problem,
            deserialize_problem,
        );
    }
}
