use super::util;
use crate::graph;
use crate::serializer::{
    from_base16, problem_to_url, to_base16, url_to_problem, Choice, Combinator, Context, Grid,
    Spaces,
};
use crate::solver::Solver;

pub fn solve_slitherlink(
    clues: &[Vec<Option<i32>>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    graph::single_cycle_grid_edges(&mut solver, &is_line);

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(is_line.cell_neighbors((y, x)).count_true().eq(n));
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

struct SlitherlinkClueCombinator;

impl Combinator<Option<i32>> for SlitherlinkClueCombinator {
    fn serialize(&self, _: &Context, input: &[Option<i32>]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let n = input[0]?;
        let mut n_spaces = 0;
        while n_spaces < 2 && 1 + n_spaces < input.len() && input[1 + n_spaces].is_none() {
            n_spaces += 1;
        }
        Some((1 + n_spaces, vec![to_base16(n + n_spaces as i32 * 5)]))
    }

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<Option<i32>>)> {
        if input.len() == 0 {
            return None;
        }
        let c = from_base16(input[0])?;
        if c == 15 {
            return None;
        }
        let n = c % 5;
        let spaces = c / 5;
        let mut ret = vec![Some(n)];
        for _ in 0..spaces {
            ret.push(None);
        }
        Some((1, ret))
    }
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(SlitherlinkClueCombinator),
        Box::new(Spaces::new(None, 'g')),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "slither", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["slither", "slitherlink"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_slitherlink_problem() {
        // original example: http://pzv.jp/p.html?slither/4/4/dgdh2c7b
        let problem = vec![
            vec![Some(3), None, None, None],
            vec![Some(3), None, None, None],
            vec![None, Some(2), Some(2), None],
            vec![None, Some(2), None, Some(1)],
        ];
        assert_eq!(serialize_problem(&problem), Some(String::from("https://puzz.link/p?slither/4/4/dgdh2c71")));
        assert_eq!(problem, deserialize_problem("https://puzz.link/p?slither/4/4/dgdh2c71").unwrap());
        let ans = solve_slitherlink(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: vec![
                vec![Some(true), Some(true), Some(true), Some(true)],
                vec![Some(true), Some(false), Some(true), Some(false)],
                vec![Some(true), Some(false), Some(false), Some(false)],
                vec![Some(false), Some(true), Some(false), Some(true)],
                vec![Some(true), Some(false), Some(false), Some(false)],
            ],
            vertical: vec![
                vec![Some(true), Some(false), Some(false), Some(false), Some(true)],
                vec![Some(false), Some(true), Some(true), Some(true), Some(true)],
                vec![Some(true), Some(false), Some(true), Some(true), Some(true)],
                vec![Some(true), Some(true), Some(false), Some(false), Some(false)],
            ],
        };
        assert_eq!(ans, expected);
    }
}
