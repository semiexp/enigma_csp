use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, HexInt, Optionalize,
    RoomsWithValues, Size, Spaces,
};
use crate::solver::{count_true, Solver};

pub fn solve_aqre(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Option<i32>],
) -> Option<Vec<Vec<Option<bool>>>> {
    let h = borders.vertical.len();
    assert!(h > 0);
    let w = borders.vertical[0].len() + 1;

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    graph::active_vertices_connected_2d(&mut solver, is_black);

    solver.add_expr(!is_black.conv2d_and((1, 4)));
    solver.add_expr(is_black.conv2d_or((1, 4)));
    solver.add_expr(!is_black.conv2d_and((4, 1)));
    solver.add_expr(is_black.conv2d_or((4, 1)));

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

    solver.irrefutable_facts().map(|f| f.get(is_black))
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
        "aqre",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["aqre"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aqre_problem() {
        let url = "https://puzz.link/p?aqre/6/6/8a41dd1t0re00g300g";
        let problem = deserialize_problem(url);
        assert!(problem.is_some());
        let problem = problem.unwrap();
        assert_eq!(serialize_problem(&problem), Some(String::from(url)));
        let (borders, clues) = problem;

        let ans = solve_aqre(&borders, &clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]);
        assert_eq!(ans, expected);
    }
}
