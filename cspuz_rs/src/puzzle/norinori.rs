use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Combinator, Context, Rooms, Size,
};
use crate::solver::{count_true, Solver};

pub fn solve_norinori(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = borders.base_shape();

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    let rooms = graph::borders_to_rooms(borders);

    for room in &rooms {
        let cells = room.iter().map(|&p| is_black.at(p)).collect::<Vec<_>>();
        solver.add_expr(count_true(cells).eq(2));
    }
    for y in 0..h {
        for x in 0..w {
            solver.add_expr(
                is_black
                    .at((y, x))
                    .imp(count_true(is_black.four_neighbors((y, x))).eq(1)),
            );
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

type Problem = graph::InnerGridEdges<Vec<Vec<bool>>>;

fn combinator() -> impl Combinator<Problem> {
    Size::new(Rooms)
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let height = problem.vertical.len();
    let width = problem.vertical[0].len() + 1;
    problem_to_url_with_context(
        combinator(),
        "norinori",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["norinori"], url)
}

#[cfg(test)]
mod tests {
    use super::super::util;
    use super::*;

    fn problem_for_tests() -> Problem {
        graph::InnerGridEdges {
            horizontal: vec![
                vec![false, true, true, false, false, false],
                vec![false, false, false, true, false, false],
                vec![false, false, true, true, true, true],
                vec![false, false, true, true, false, false],
                vec![true, true, true, false, true, true],
            ],
            vertical: vec![
                vec![true, false, true, true, false],
                vec![false, true, false, true, false],
                vec![false, true, true, false, false],
                vec![false, false, false, true, false],
                vec![false, true, false, true, false],
                vec![false, false, true, false, false],
            ],
        }
    }

    #[test]
    fn test_norinori_problem() {
        let problem = problem_for_tests();
        let ans = solve_norinori(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_norinori_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?norinori/6/6/mac2a4c11spr";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
