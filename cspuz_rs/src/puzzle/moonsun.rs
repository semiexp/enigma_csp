use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Combinator, Context, ContextBasedGrid, MultiDigit,
    Rooms, Size, Tuple2,
};
use crate::solver::{count_true, Solver};

pub fn solve_moonsun(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Vec<i32>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let is_passed = &graph::single_cycle_grid_edges(&mut solver, is_line);
    let rooms = graph::borders_to_rooms(borders);
    let room_mode = &solver.bool_var_1d(rooms.len()); // false: 1, true: 2
    let mut room_id = vec![vec![0; w]; h];

    for i in 0..rooms.len() {
        let mut has_one = false;
        let mut has_two = false;
        for &(y, x) in &rooms[i] {
            let n = clues[y][x];
            if n != 0 {
                if n == 1 {
                    has_one = true;
                }
                if n == 2 {
                    has_two = true;
                }
                solver.add_expr(is_passed.at((y, x)).iff(room_mode.at(i).iff(n == 2)));
            }
        }
        if !(has_one || has_two) {
            return None;
        }
        if !has_one {
            solver.add_expr(room_mode.at(i));
        }
        if !has_two {
            solver.add_expr(!room_mode.at(i));
        }
    }

    for i in 0..rooms.len() {
        for &(y, x) in &rooms[i] {
            room_id[y][x] = i;
        }
    }

    let mut room_entrance = vec![vec![]; rooms.len()];
    for y in 0..h {
        for x in 0..w {
            if y < h - 1 && room_id[y][x] != room_id[y + 1][x] {
                solver.add_expr(
                    is_line
                        .vertical
                        .at((y, x))
                        .imp(room_mode.at(room_id[y][x]) ^ room_mode.at(room_id[y + 1][x])),
                );
                room_entrance[room_id[y][x]].push(is_line.vertical.at((y, x)));
                room_entrance[room_id[y + 1][x]].push(is_line.vertical.at((y, x)));
            }
            if x < w - 1 && room_id[y][x] != room_id[y][x + 1] {
                solver.add_expr(
                    is_line
                        .horizontal
                        .at((y, x))
                        .imp(room_mode.at(room_id[y][x]) ^ room_mode.at(room_id[y][x + 1])),
                );
                room_entrance[room_id[y][x]].push(is_line.horizontal.at((y, x)));
                room_entrance[room_id[y][x + 1]].push(is_line.horizontal.at((y, x)));
            }
        }
    }
    for i in 0..rooms.len() {
        solver.add_expr(count_true(&room_entrance[i]).eq(2));
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = (graph::InnerGridEdges<Vec<Vec<bool>>>, Vec<Vec<i32>>);

fn combinator() -> impl Combinator<Problem> {
    Size::new(Tuple2::new(
        Rooms,
        ContextBasedGrid::new(MultiDigit::new(3, 3)),
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let height = problem.0.vertical.len();
    let width = problem.0.vertical[0].len() + 1;
    problem_to_url_with_context(
        combinator(),
        "moonsun",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["moonsun"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        (
            graph::InnerGridEdges {
                horizontal: vec![
                    vec![false, false, true, false, true, false],
                    vec![true, true, false, true, true, false],
                    vec![false, true, false, true, true, false],
                    vec![false, true, true, false, true, false],
                    vec![false, true, false, false, false, false],
                ],
                vertical: vec![
                    vec![false, true, false, true, false],
                    vec![false, true, true, false, true],
                    vec![false, true, true, false, false],
                    vec![true, false, true, false, true],
                    vec![false, true, false, true, false],
                    vec![true, false, false, true, false],
                ]
            },
            vec![
                vec![1, 0, 0, 0, 0, 0],
                vec![0, 1, 0, 2, 1, 0],
                vec![1, 0, 0, 0, 0, 1],
                vec![0, 2, 0, 2, 0, 0],
                vec![0, 2, 1, 0, 1, 0],
                vec![0, 0, 0, 0, 2, 0],
            ]
        )
    }

    #[test]
    fn test_moonsun_problem() {
        let (borders, clues) = problem_for_tests();
        let ans = solve_moonsun(&borders, &clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        #[rustfmt::skip]
        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: vec![
                vec![Some(true), Some(true), Some(true), Some(false), Some(false)],
                vec![Some(true), Some(true), Some(false), Some(false), Some(false)],
                vec![Some(true), Some(false), Some(false), Some(true), Some(true)],
                vec![Some(false), Some(true), Some(false), Some(true), Some(false)],
                vec![Some(false), Some(false), Some(false), Some(false), Some(true)],
                vec![Some(true), Some(true), Some(true), Some(false), Some(false)],
            ],
            vertical: vec![
                vec![Some(true), Some(false), Some(false), Some(true), Some(false), Some(false)],
                vec![Some(false), Some(false), Some(true), Some(true), Some(false), Some(false)],
                vec![Some(true), Some(true), Some(true), Some(false), Some(false), Some(true)],
                vec![Some(true), Some(false), Some(false), Some(true), Some(true), Some(true)],
                vec![Some(true), Some(false), Some(false), Some(true), Some(false), Some(false)],
            ],
        };
        assert_eq!(ans, expected);
    }
    #[test]
    fn test_moonsun_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?moonsun/6/6/adclai5dipkg903l916i7306";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
