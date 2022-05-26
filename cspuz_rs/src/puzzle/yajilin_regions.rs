use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, HexInt, Optionalize,
    RoomsWithValues, Size, Spaces,
};
use crate::solver::{count_true, Solver};

pub fn solve_yajilin_regions(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Option<i32>],
) -> Option<(graph::BoolGridEdgesIrrefutableFacts, Vec<Vec<Option<bool>>>)> {
    let h = borders.vertical.len();
    assert!(h > 0);
    let w = borders.vertical[0].len() + 1;

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let is_passed = &graph::single_cycle_grid_edges(&mut solver, is_line);
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);
    solver.add_expr(is_passed ^ is_black);
    solver.add_expr(!(is_black.slice((..(h - 1), ..)) & is_black.slice((1.., ..))));
    solver.add_expr(!(is_black.slice((.., ..(w - 1))) & is_black.slice((.., 1..))));

    let rooms = graph::borders_to_rooms(borders);
    assert_eq!(rooms.len(), clues.len());

    for i in 0..rooms.len() {
        if let Some(n) = clues[i] {
            let mut cells = vec![];
            for &pt in &rooms[i] {
                cells.push(is_black.at(pt));
            }
            solver.add_expr(count_true(cells).eq(n));
        }
    }

    solver
        .irrefutable_facts()
        .map(|f| (f.get(is_line), f.get(is_black)))
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
        "yajilin-regions",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["yajilin-regions"], url)
}

#[cfg(test)]
mod tests {
    use super::super::util;
    use super::*;

    fn problem_for_tests() -> Problem {
        let borders = graph::InnerGridEdges {
            horizontal: vec![
                vec![false, false, false, false, false, false],
                vec![false, true, true, true, false, false],
                vec![false, false, false, false, true, true],
                vec![false, true, false, false, false, false],
                vec![false, true, false, false, false, false],
            ],
            vertical: vec![
                vec![true, false, false, true, false],
                vec![true, false, false, true, false],
                vec![false, false, false, false, false],
                vec![false, false, false, true, false],
                vec![true, true, false, true, false],
                vec![false, false, false, true, false],
            ],
        };
        let clues = vec![None, Some(2), Some(2), Some(1)];
        (borders, clues)
    }

    #[test]
    fn test_yajilin_regions_problem() {
        let (borders, clues) = problem_for_tests();
        let ans = solve_yajilin_regions(&borders, &clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected_base = [
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ];
        let expected =
            expected_base.map(|row| row.iter().map(|&n| Some(n == 1)).collect::<Vec<_>>());
        assert_eq!(ans.1, expected);
    }

    #[test]
    fn test_yajilin_regions_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?yajilin-regions/6/6/ii02q2070d0gg221";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
