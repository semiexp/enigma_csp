use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, HexInt, Optionalize,
    RoomsWithValues, Size, Spaces,
};
use crate::solver::{count_true, BoolVarArray2D, Solver};

pub fn solve_heyawake(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Option<i32>],
) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = borders.base_shape();

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    add_constraints(&mut solver, is_black, borders, clues);

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

pub fn enumerate_answers_heyawake(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Option<i32>],
    num_max_answers: usize,
) -> Vec<Vec<Vec<bool>>> {
    let h = borders.vertical.len();
    assert!(h > 0);
    let w = borders.vertical[0].len() + 1;

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    add_constraints(&mut solver, is_black, borders, clues);

    solver
        .answer_iter()
        .take(num_max_answers)
        .map(|f| f.get_unwrap(is_black))
        .collect()
}

pub(super) fn add_constraints(
    solver: &mut Solver,
    is_black: &BoolVarArray2D,
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Option<i32>],
) {
    let h = borders.vertical.len();
    assert!(h > 0);
    let w = borders.vertical[0].len() + 1;

    graph::active_vertices_connected_2d(solver, !is_black);
    solver.add_expr(!(is_black.slice((..(h - 1), ..)) & is_black.slice((1.., ..))));
    solver.add_expr(!(is_black.slice((.., ..(w - 1))) & is_black.slice((.., 1..))));

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

    for i in 0..rooms.len() {
        if let Some(n) = clues[i] {
            let mut cells = vec![];
            for &pt in &rooms[i] {
                cells.push(is_black.at(pt));
            }
            solver.add_expr(count_true(cells).eq(n));
        }
    }
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
        "heyawake",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["heyawake"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heyawake_problem() {
        // https://puzz.link/p?heyawake/6/6/aa66aapv0fu0g2i3k
        let url = "https://puzz.link/p?heyawake/6/6/aa66aapv0fu0g2i3k";
        let problem = deserialize_problem(url);
        assert!(problem.is_some());
        let problem = problem.unwrap();
        assert_eq!(serialize_problem(&problem), Some(String::from(url)));
        let (borders, clues) = problem;

        let ans = solve_heyawake(&borders, &clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected_base = [
            [1, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 0],
        ];
        let expected =
            expected_base.map(|row| row.iter().map(|&n| Some(n == 1)).collect::<Vec<_>>());
        assert_eq!(ans, expected);
    }
}
