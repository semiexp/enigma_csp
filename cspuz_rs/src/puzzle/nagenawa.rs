use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, HexInt, Optionalize,
    RoomsWithValues, Size, Spaces,
};
use crate::solver::{count_true, Solver, FALSE};

pub fn solve_nagenawa(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Option<i32>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = borders.base_shape();

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let horizontal_y = solver.int_var_2d((h, w), 0, (h - 1) as i32);
    let horizontal_x = solver.int_var_2d((h, w), 0, (w - 1) as i32);
    let horizontal_h = solver.int_var_2d((h, w), 0, (h - 1) as i32);
    let horizontal_w = solver.int_var_2d((h, w), 0, (w - 1) as i32);
    let vertical_y = solver.int_var_2d((h, w), 0, (h - 1) as i32);
    let vertical_x = solver.int_var_2d((h, w), 0, (w - 1) as i32);
    let vertical_h = solver.int_var_2d((h, w), 0, (h - 1) as i32);
    let vertical_w = solver.int_var_2d((h, w), 0, (w - 1) as i32);

    for y in 0..h {
        for x in 0..w {
            if 0 < y {
                solver.add_expr(is_line.vertical.at((y - 1, x)).imp(
                    vertical_h.at((y - 1, x)).eq(vertical_h.at((y, x)))
                        & vertical_w.at((y - 1, x)).eq(vertical_w.at((y, x)))
                        & vertical_y.at((y - 1, x)).eq(vertical_y.at((y, x)) - 1)
                        & vertical_x.at((y - 1, x)).eq(vertical_x.at((y, x))),
                ));
            }
            if 0 < x {
                solver.add_expr(is_line.horizontal.at((y, x - 1)).imp(
                    horizontal_h.at((y, x - 1)).eq(horizontal_h.at((y, x)))
                        & horizontal_w.at((y, x - 1)).eq(horizontal_w.at((y, x)))
                        & horizontal_y.at((y, x - 1)).eq(horizontal_y.at((y, x)))
                        & horizontal_x.at((y, x - 1)).eq(horizontal_x.at((y, x)) - 1),
                ));
            }

            let is_corner = &solver.bool_var();
            solver.add_expr(is_corner.iff(
                is_line.vertical.at_offset((y, x), (-1, 0), FALSE)
                    ^ is_line.vertical.at_offset((y, x), (0, 0), FALSE),
            ));
            solver.add_expr(is_corner.iff(
                is_line.horizontal.at_offset((y, x), (0, -1), FALSE)
                    ^ is_line.horizontal.at_offset((y, x), (0, 0), FALSE),
            ));
            solver.add_expr(
                (is_corner & !is_line.vertical.at_offset((y, x), (-1, 0), FALSE))
                    .imp(vertical_y.at((y, x)).eq(0)),
            );
            solver.add_expr(
                (is_corner & !is_line.vertical.at_offset((y, x), (0, 0), FALSE))
                    .imp(vertical_y.at((y, x)).eq(vertical_h.at((y, x)))),
            );
            solver.add_expr(
                (is_corner & !is_line.horizontal.at_offset((y, x), (0, -1), FALSE))
                    .imp(horizontal_x.at((y, x)).eq(0)),
            );
            solver.add_expr(
                (is_corner & !is_line.horizontal.at_offset((y, x), (0, 0), FALSE))
                    .imp(horizontal_x.at((y, x)).eq(horizontal_w.at((y, x)))),
            );
            solver.add_expr(is_corner.imp(
                horizontal_y.at((y, x)).eq(vertical_y.at((y, x)))
                    & horizontal_x.at((y, x)).eq(vertical_x.at((y, x)))
                    & horizontal_h.at((y, x)).eq(vertical_h.at((y, x)))
                    & horizontal_w.at((y, x)).eq(vertical_w.at((y, x))),
            ));
        }
    }

    let rooms = graph::borders_to_rooms(borders);
    assert_eq!(rooms.len(), clues.len());

    for i in 0..rooms.len() {
        if let Some(n) = clues[i] {
            let mut cells = vec![];
            for &pt in &rooms[i] {
                cells.push(is_line.vertex_neighbors(pt).any());
            }
            solver.add_expr(count_true(cells).eq(n));
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = (graph::InnerGridEdges<Vec<Vec<bool>>>, Vec<Option<i32>>);

fn combinator() -> impl Combinator<Problem> {
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
        "nagenawa",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["nagenawa"], url)
}

#[cfg(test)]
mod tests {
    use super::super::util;
    use super::*;

    fn problem_for_tests() -> Problem {
        let borders = graph::InnerGridEdges {
            horizontal: crate::puzzle::util::tests::to_bool_2d([
                [1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_bool_2d([
                [0, 0, 0, 1, 0],
                [1, 1, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [1, 0, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [0, 1, 1, 0, 0],
            ]),
        };
        let clues = vec![
            Some(3),
            None,
            None,
            Some(0),
            Some(4),
            Some(1),
            Some(1),
            Some(1),
            Some(2),
        ];
        (borders, clues)
    }

    #[test]
    fn test_nagenawa_problem() {
        let (borders, clues) = problem_for_tests();
        let ans = solve_nagenawa(&borders, &clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 1, 1, 1, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1],
                [0, 1, 0, 1, 1, 1],
                [0, 1, 0, 0, 1, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_nagenawa_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?nagenawa/6/6/2u6mucu440hn3h041112";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
