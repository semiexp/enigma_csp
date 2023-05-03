use super::util;
use crate::graph;
use crate::puzzle::slitherlink::SlitherlinkClueCombinator;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, ContextBasedGrid,
    Size, Spaces,
};
use crate::solver::Solver;

pub fn solve_creek(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h1, w1) = util::infer_shape(clues);
    let h = h1 - 1;
    let w = w1 - 1;

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    for y in 0..=h {
        for x in 0..=w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(
                    is_black
                        .slice((
                            (y.max(1) - 1)..((y + 1).min(h)),
                            (x.max(1) - 1)..((x + 1).min(h)),
                        ))
                        .count_true()
                        .eq(n),
                );
            }
        }
    }

    graph::active_vertices_connected_2d(&mut solver, !is_black);

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Size::with_offset(
        ContextBasedGrid::new(Choice::new(vec![
            Box::new(SlitherlinkClueCombinator),
            Box::new(Spaces::new(None, 'g')),
        ])),
        1,
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let (h, w) = util::infer_shape(problem);
    problem_to_url_with_context(
        combinator(),
        "creek",
        problem.clone(),
        &Context::sized(h, w),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["creek"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        vec![
            vec![None, None, None, None, None, None, None],
            vec![None, None, None, None, Some(2), Some(2), None],
            vec![None, None, Some(2), None, None, None, None],
            vec![None, None, Some(1), Some(3), None, Some(2), None],
            vec![None, Some(3), None, None, None, None, None],
            vec![None, None, None, None, Some(3), Some(2), None],
            vec![None, Some(3), None, Some(3), None, Some(2), None],
            vec![None, None, None, None, None, None, None],
        ]
    }

    #[test]
    fn test_creek_problem() {
        let problem = problem_for_tests();
        let ans = solve_creek(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_creek_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?creek/6/7/q2cgcj18cdm3c88cl";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
