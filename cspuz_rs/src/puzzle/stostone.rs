use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, HexInt, Optionalize,
    RoomsWithValues, Size, Spaces,
};
use crate::solver::{any, count_true, Solver};

pub fn solve_stostone(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Option<i32>],
) -> Option<Vec<Vec<Option<bool>>>> {
    let h = borders.vertical.len();
    assert!(h > 0);
    let w = borders.vertical[0].len() + 1;

    if h % 2 != 0 {
        return None;
    }

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    for y in 0..h {
        for x in 0..w {
            if y < h - 1 && borders.horizontal[y][x] {
                solver.add_expr(!(is_black.at((y, x)) & is_black.at((y + 1, x))));
            }
            if x < w - 1 && borders.vertical[y][x] {
                solver.add_expr(!(is_black.at((y, x)) & is_black.at((y, x + 1))));
            }
        }
    }
    let rooms = graph::borders_to_rooms(borders);
    assert_eq!(rooms.len(), clues.len());

    for i in 0..rooms.len() {
        graph::active_vertices_connected_2d_region(&mut solver, &is_black, &rooms[i]);
        let mut cells = vec![];
        for &pt in &rooms[i] {
            cells.push(is_black.at(pt));
        }
        if let Some(n) = clues[i] {
            solver.add_expr(count_true(cells).eq(n));
        } else {
            solver.add_expr(any(cells));
        }
    }

    let h2 = (h / 2) as i32;
    let rank = solver.int_var_2d((h + 1, w), 0, h2);
    solver.add_expr(
        rank.slice((1.., ..))
            .eq(rank.slice((..h, ..)) + is_black.ite(1, 0)),
    );
    solver.add_expr(rank.slice_fixed_y((0, ..)).eq(0));
    solver.add_expr(rank.slice_fixed_y((h, ..)).eq(h2));
    for i in 0..rooms.len() {
        let lift = &solver.int_var(0, h2);
        for &(y, x) in &rooms[i] {
            solver.add_expr(
                is_black
                    .at((y, x))
                    .imp((rank.at((y, x)) + lift).eq(y as i32)),
            );
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
        "stostone",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["stostone"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        (
            graph::InnerGridEdges {
                horizontal: crate::puzzle::util::tests::to_bool_2d([
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1],
                    [1, 1, 0, 1, 1, 1],
                    [0, 0, 1, 0, 0, 0],
                ]),
                vertical: crate::puzzle::util::tests::to_bool_2d([
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                ]),
            },
            vec![
                Some(3),
                Some(1),
                None,
                None,
                Some(3),
                None,
                Some(2),
                Some(3),
            ],
        )
    }

    #[test]
    fn test_cocktail_problem() {
        let (borders, clues) = problem_for_tests();
        let ans = solve_stostone(&borders, &clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 1],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_moonsun_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?stostone/6/6/222ac4vg1ve831h3g23";
        crate::puzzle::util::tests::serializer_test(
            problem,
            url,
            serialize_problem,
            deserialize_problem,
        );
    }
}
