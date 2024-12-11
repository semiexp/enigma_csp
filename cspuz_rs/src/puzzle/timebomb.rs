use super::util;
use crate::serializer::{
    problem_to_url_with_context_and_site, url_to_problem, Choice, Combinator, Context, Dict, Grid,
    HexInt, Optionalize, Spaces,
};
use crate::solver::Solver;

pub fn solve_timebomb(
    clues: &[Vec<Option<i32>>],
) -> Option<(Vec<Vec<Option<bool>>>, Vec<Vec<Option<i32>>>)> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let has_number = &solver.bool_var_2d((h, w));
    let num = &solver.int_var_2d((h, w), -1, (h * w) as i32);
    solver.add_answer_key_bool(has_number);
    solver.add_answer_key_int(num);

    let mut bombs = vec![];
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                if n >= -1 {
                    bombs.push((y, x, n));
                } else {
                    solver.add_expr(!has_number.at((y, x)));
                }
            }
        }
    }

    let bomb_id = &solver.int_var_2d((h, w), 0, bombs.len() as i32);
    solver.add_expr(has_number.iff(bomb_id.ne(bombs.len() as i32)));
    solver.add_expr(has_number.iff(num.ne(-1)));

    solver.add_expr(
        has_number
            .conv2d_and((1, 2))
            .imp(bomb_id.slice((.., 1..)).eq(bomb_id.slice((.., ..w - 1)))),
    );
    solver.add_expr(
        has_number
            .conv2d_and((2, 1))
            .imp(bomb_id.slice((1.., ..)).eq(bomb_id.slice((..h - 1, ..)))),
    );

    for i in 0..bombs.len() {
        let (y, x, n) = bombs[i];
        solver.add_expr(bomb_id.at((y, x)).eq(i as i32));

        if n >= 0 {
            solver.add_expr(num.at((y, x)).eq(n));
            solver.add_expr(bomb_id.eq(i as i32).imp(num.le(n)));
            for j in 0..=n {
                solver.add_expr((bomb_id.eq(i as i32) & num.eq(j)).count_true().eq(1));
            }
        } else {
            for j in 0..=(h * w) {
                solver.add_expr((bomb_id.eq(i as i32) & num.eq(j as i32)).count_true().le(1));
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            if clues[y][x].is_none() {
                solver.add_expr(
                    has_number.at((y, x)).imp(
                        num.four_neighbors((y, x))
                            .eq(num.at((y, x)) + 1)
                            .count_true()
                            .eq(1),
                    ),
                );
            }
            solver.add_expr(
                num.at((y, x)).ge(1).imp(
                    num.four_neighbors((y, x))
                        .eq(num.at((y, x)) - 1)
                        .count_true()
                        .eq(1),
                ),
            );
        }
    }

    for y in 0..h {
        for x in 0..w {
            if clues[y][x] == Some(-2) {
                let y0 = (y as i32 - 1).max(0) as usize;
                let y1 = (y + 2).min(h);
                let x0 = (x as i32 - 1).max(0) as usize;
                let x1 = (x + 2).min(w);
                solver.add_expr(num.slice((y0..y1, x0..x1)).eq(0).any());
            }
        }
    }

    solver
        .irrefutable_facts()
        .map(|f| (f.get(has_number), f.get(num)))
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Dict::new(Some(-2), "0")),
        Box::new(Dict::new(Some(-1), ".")),
        Box::new(Optionalize::new(HexInt)),
        Box::new(Spaces::new(None, 'g')),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let height = problem.len();
    let width = problem.len();
    problem_to_url_with_context_and_site(
        combinator(),
        "timebomb",
        "https://pzprxs.vercel.app/p?",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["timebomb"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![Some(5), None, None, None, None, Some(-2)],
            vec![None, None, Some(-2), None, None, Some(-2)],
            vec![Some(-1), None, None, None, None, None],
            vec![Some(-2), None, Some(-2), Some(1), None, Some(2)],
            vec![None, Some(-2), None, None, None, None],
        ]
    }

    #[test]
    fn test_timebomb_problem() {
        let problem = problem_for_tests();
        let ans = solve_timebomb(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = (
            crate::puzzle::util::tests::to_option_bool_2d([
                [1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 1],
            ]),
            crate::puzzle::util::tests::to_option_2d([
                [5, 4, 3, 2, 1, -1],
                [-1, -1, -1, -1, 0, -1],
                [2, 1, -1, 0, -1, -1],
                [-1, 0, -1, 1, -1, 2],
                [-1, -1, -1, -1, 0, 1],
            ]),
        );
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_timebomb_serializer() {
        let problem = problem_for_tests();
        let url = "https://pzprxs.vercel.app/p?timebomb/6/5/5j0h0h0.k0g01g2g0j";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
