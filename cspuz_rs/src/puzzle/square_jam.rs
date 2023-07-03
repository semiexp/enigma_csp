use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::Solver;

pub fn solve_square_jam(
    clues: &[Vec<Option<i32>>],
) -> Option<graph::BoolInnerGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_border = graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&is_border.horizontal);
    solver.add_answer_key_bool(&is_border.vertical);

    let mut cell_id = vec![vec![None; w]; h];
    let mut id_last = 0usize;
    for y in 0..h {
        for x in 0..w {
            if clues[y][x] != Some(-2) {
                cell_id[y][x] = Some(id_last);
                id_last += 1;
            }
        }
    }

    let num_up = &solver.int_var_2d((h, w), 0, h as i32 - 1);
    solver.add_expr(num_up.slice_fixed_y((0, ..)).eq(0));
    solver.add_expr(
        num_up.slice((1.., ..)).eq(is_border
            .horizontal
            .ite(0, num_up.slice((..(h - 1), ..)) + 1)),
    );
    let num_down = &solver.int_var_2d((h, w), 0, h as i32 - 1);
    solver.add_expr(num_down.slice_fixed_y((h - 1, ..)).eq(0));
    solver.add_expr(
        num_down
            .slice((..(h - 1), ..))
            .eq(is_border.horizontal.ite(0, num_down.slice((1.., ..)) + 1)),
    );
    let num_left = &solver.int_var_2d((h, w), 0, w as i32 - 1);
    solver.add_expr(num_left.slice_fixed_x((.., 0)).eq(0));
    solver.add_expr(
        num_left.slice((.., 1..)).eq(is_border
            .vertical
            .ite(0, num_left.slice((.., ..(w - 1))) + 1)),
    );
    let num_right = &solver.int_var_2d((h, w), 0, w as i32 - 1);
    solver.add_expr(num_right.slice_fixed_x((.., w - 1)).eq(0));
    solver.add_expr(
        num_right
            .slice((.., ..(w - 1)))
            .eq(is_border.vertical.ite(0, num_right.slice((.., 1..)) + 1)),
    );

    for y in 0..h {
        for x in 0..w {
            solver.add_expr(
                (num_up.at((y, x)) + num_down.at((y, x)))
                    .eq(num_left.at((y, x)) + num_right.at((y, x))),
            );
            if let Some(n) = clues[y][x] {
                solver.add_expr((num_up.at((y, x)) + num_down.at((y, x))).eq(n - 1));
                solver.add_expr((num_left.at((y, x)) + num_right.at((y, x))).eq(n - 1));
            }
        }
    }

    for y in 1..h {
        for x in 1..w {
            let left = &is_border.horizontal.at((y - 1, x - 1));
            let right = &is_border.horizontal.at((y - 1, x));
            let up = &is_border.vertical.at((y - 1, x - 1));
            let down = &is_border.vertical.at((y, x - 1));
            solver.add_expr(!(left & right & up & down));
            solver.add_expr(!((left ^ right) & (up ^ down)));
        }
    }

    solver.irrefutable_facts().map(|f| f.get(&is_border))
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Spaces::new(None, 'g')),
        Box::new(Optionalize::new(HexInt)),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "squarejam", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["squarejam"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![None, Some(2), None, None, None, None],
            vec![None, None, None, None, None, None],
            vec![None, Some(1), None, None, Some(2), None],
            vec![None, None, None, None, None, None],
            vec![None, None, None, None, None, None],
            vec![None, None, None, None, None, None],
            vec![None, None, Some(1), None, None, None],
        ]
    }

    #[test]
    fn test_square_jam_problem() {
        let problem = problem_for_tests();
        let ans = solve_square_jam(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        #[rustfmt::skip]
        let expected = graph::BoolInnerGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 0, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_square_jam_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?squarejam/6/7/g2q1h2zg1i";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
