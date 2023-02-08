use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::Solver;

pub fn solve_fillomino(
    clues: &[Vec<Option<i32>>],
) -> Option<(
    Vec<Vec<Option<i32>>>,
    graph::BoolInnerGridEdgesIrrefutableFacts,
)> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let num = &solver.int_var_2d((h, w), 1, (h * w) as i32);
    solver.add_answer_key_int(num);

    let is_border = graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&is_border.horizontal);
    solver.add_answer_key_bool(&is_border.vertical);
    solver.add_expr(
        num.slice((.., ..(w - 1)))
            .ne(num.slice((.., 1..)))
            .iff(&is_border.vertical),
    );
    solver.add_expr(
        num.slice((..(h - 1), ..))
            .ne(num.slice((1.., ..)))
            .iff(&is_border.horizontal),
    );

    graph::graph_division_2d(&mut solver, num, &is_border);

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(num.at((y, x)).eq(n));
            }
        }
    }

    solver
        .irrefutable_facts()
        .map(|f| (f.get(num), f.get(&is_border)))
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Optionalize::new(HexInt)),
        Box::new(Spaces::new(None, 'g')),
        Box::new(Dict::new(Some(-1), ".")),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "fillomino", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["fillomino"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![None, Some(1), None, None, None],
            vec![None, None, Some(3), Some(4), None],
            vec![Some(2), None, None, Some(5), None],
            vec![None, Some(4), None, None, None],
            vec![None, None, None, None, None],
        ]
    }

    #[test]
    fn test_fillomino_problem() {
        let problem = problem_for_tests();
        let ans = solve_fillomino(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = crate::puzzle::util::tests::to_option_2d([
            [6, 1, 3, 3, 4],
            [6, 6, 3, 4, 4],
            [2, 6, 6, 5, 4],
            [2, 4, 6, 5, 5],
            [4, 4, 4, 5, 5],
        ]);
        assert_eq!(ans.0, expected);
    }

    #[test]
    fn test_fillomino_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?fillomino/5/5/g1k34g2h5h4n";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
